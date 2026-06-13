"""KSL 모델 트레이너.

모든 stage가 독립 실행 가능하도록 설계한다.
config 파일로 재현 가능한 실험을 보장한다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.signals import NMS_DETAIL_GROUPS

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    stage: str = "C"
    max_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 8
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every_n_epochs: int = 5
    val_every_n_epochs: int = 1
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    device: str = "cpu"
    use_amp: bool = False

    # Loss 가중치
    loss_weight_gloss: float = 1.0
    loss_weight_nms: float = 0.5
    loss_weight_nms_detail: float = 0.3
    loss_weight_intent: float = 0.3
    loss_weight_boundary: float = 0.5
    loss_weight_draft: float = 1.0     # Stage C


class KSLTrainer:
    """KSL 모델 학습기.

    Args:
        model: KSLModel
        config: TrainerConfig
        train_loader: 학습 DataLoader
        val_loader: 검증 DataLoader (없으면 None)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs
        )

        self.ckpt_dir = Path(config.checkpoint_dir) / config.stage
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.global_step = 0
        self.best_val_loss = float("inf")

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """배치 출력에 대해 loss를 계산한다."""
        c = self.config
        losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device)

        # Gloss CTC
        if "gloss_logits" in outputs and "gloss_ids" in batch:
            log_probs = outputs["gloss_logits"].log_softmax(dim=-1)  # [B, T, V]
            log_probs = log_probs.permute(1, 0, 2)                    # [T, B, V]
            seq_len = batch["seq_len"].to(self.device)
            gloss_ids = batch["gloss_ids"]
            tgt_lengths = torch.tensor([g.shape[0] for g in gloss_ids], dtype=torch.long)
            tgt_flat = torch.cat(gloss_ids).to(self.device)
            gloss_loss = self.ctc_loss(log_probs, tgt_flat, seq_len, tgt_lengths)
            losses["gloss"] = gloss_loss
            total = total + c.loss_weight_gloss * gloss_loss

        # Intent CE
        if "intent_logits" in outputs and "intent_label" in batch:
            intent_logits = outputs["intent_logits"]
            intent_labels = batch["intent_label"].to(self.device)
            intent_loss = self.ce_loss(intent_logits, intent_labels)
            losses["intent"] = intent_loss
            total = total + c.loss_weight_intent * intent_loss

        # NMS multi-label BCE
        if "nms_logits" in outputs and "nms_label" in batch and "nms_mask" in batch:
            nms_logits = outputs["nms_logits"]                  # [B, T, C]
            nms_labels = batch["nms_label"].to(self.device)     # [B, C]
            nms_mask = batch["nms_mask"].to(self.device)        # [B, C]
            seq_len = batch["seq_len"].to(self.device)

            B, T, _ = nms_logits.shape
            valid = (
                torch.arange(T, device=self.device).unsqueeze(0) < seq_len.unsqueeze(1)
            ).float()
            pooled_logits = (nms_logits * valid.unsqueeze(-1)).sum(dim=1)
            pooled_logits = pooled_logits / valid.sum(dim=1, keepdim=True).clamp(min=1.0)

            if nms_mask.sum() > 0:
                nms_loss_raw = F.binary_cross_entropy_with_logits(
                    pooled_logits,
                    nms_labels,
                    reduction="none",
                )
                nms_loss = (nms_loss_raw * nms_mask).sum() / nms_mask.sum().clamp(min=1.0)
                losses["nms"] = nms_loss
                total = total + c.loss_weight_nms * nms_loss

        # Fine-grained NMS categorical CE.
        if "nms_detail_label" in batch and "nms_detail_mask" in batch:
            detail_labels = batch["nms_detail_label"].to(self.device)      # [B, G]
            detail_mask = batch["nms_detail_mask"].to(self.device).bool()  # [B, G]
            seq_len = batch["seq_len"].to(self.device)
            detail_losses = []

            for group_idx, group in enumerate(NMS_DETAIL_GROUPS):
                key = f"nms_{group}_logits"
                if key not in outputs:
                    continue
                logits = outputs[key]  # [B, T, C_group]
                B, T, _ = logits.shape
                valid = (
                    torch.arange(T, device=self.device).unsqueeze(0) < seq_len.unsqueeze(1)
                ).float()
                pooled_logits = (logits * valid.unsqueeze(-1)).sum(dim=1)
                pooled_logits = pooled_logits / valid.sum(dim=1, keepdim=True).clamp(min=1.0)

                supervised = detail_mask[:, group_idx]
                if supervised.any():
                    detail_losses.append(
                        F.cross_entropy(
                            pooled_logits[supervised],
                            detail_labels[supervised, group_idx],
                        )
                    )

            if detail_losses:
                nms_detail_loss = torch.stack(detail_losses).mean()
                losses["nms_detail"] = nms_detail_loss
                total = total + c.loss_weight_nms_detail * nms_detail_loss

        # Boundary CE
        if "boundary_logits" in outputs and "activity" in batch:
            bnd_logits = outputs["boundary_logits"]          # [B, T, 3]
            activity = batch["activity"].to(self.device)      # [B, T]
            seq_len = batch["seq_len"].to(self.device)
            B, T, _ = bnd_logits.shape
            valid = torch.arange(T, device=self.device).unsqueeze(0) < seq_len.unsqueeze(1)
            bnd_loss = self.ce_loss(bnd_logits[valid], activity[valid])
            losses["boundary"] = bnd_loss
            total = total + c.loss_weight_boundary * bnd_loss

        # Draft CE (Stage C)
        if "draft_logits" in outputs and "draft_labels" in batch:
            draft_logits = outputs["draft_logits"]   # [B, L, vocab]
            draft_labels = batch["draft_labels"].to(self.device)   # [B, L]
            B, L, V = draft_logits.shape
            draft_loss = self.ce_loss(draft_logits.view(B * L, V), draft_labels.view(B * L))
            losses["draft"] = draft_loss
            total = total + c.loss_weight_draft * draft_loss

        losses["total"] = total
        return losses

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self._forward(batch)
                losses = self.compute_loss(batch, outputs)
            loss = losses["total"]
            if not torch.isfinite(loss):
                loss_parts = {
                    k: (float(v.detach().cpu()) if torch.isfinite(v.detach()).all() else "non-finite")
                    for k, v in losses.items()
                }
                raise FloatingPointError(f"Non-finite training loss: {loss_parts}")

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def val_epoch(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self._forward(batch)
                losses = self.compute_loss(batch, outputs)
            total_loss += losses["total"].item()
        return total_loss / max(len(self.val_loader), 1)

    def fit(self, resume: str | None = None) -> None:
        """전체 학습 루프."""
        import re

        start_epoch = 1
        epochs_without_improvement = 0

        if resume:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            saved_epoch = ckpt.get("epoch")
            if saved_epoch is None:
                m = re.match(r"epoch_(\d+)", Path(resume).stem)
                saved_epoch = int(m.group(1)) if m else 0
            start_epoch = saved_epoch + 1
            # scheduler를 재개 epoch까지 fast-forward한다
            for _ in range(saved_epoch):
                self.scheduler.step()
            logger.info(
                "Resumed from %s — next epoch=%d, best_val_loss=%.4f",
                resume, start_epoch, self.best_val_loss,
            )

        logger.info(f"Starting Stage {self.config.stage} training from epoch {start_epoch}")
        for epoch in range(start_epoch, self.config.max_epochs + 1):
            train_loss = self.train_epoch()
            self.scheduler.step()

            if epoch % self.config.val_every_n_epochs == 0:
                val_loss = self.val_epoch()
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

                improved = val_loss < self.best_val_loss - self.config.early_stop_min_delta
                if improved:
                    self.best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save_checkpoint("best.pt", epoch=epoch)
                    self.save_checkpoint("best_by_val_loss.pt", epoch=epoch)
                else:
                    epochs_without_improvement += 1
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch=epoch)

            if (
                self.config.early_stop_patience > 0
                and epoch % self.config.val_every_n_epochs == 0
                and epochs_without_improvement >= self.config.early_stop_patience
            ):
                logger.info(
                    "Early stopping at epoch %s: best_val_loss=%.4f, patience=%s",
                    epoch,
                    self.best_val_loss,
                    self.config.early_stop_patience,
                )
                break

        self.save_checkpoint("last.pt", epoch=self.config.max_epochs)

    def save_checkpoint(self, name: str, epoch: int = 0) -> None:
        path = self.ckpt_dir / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        logger.info(f"Loaded checkpoint from {path} (strict={strict})")

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                result[k] = [t.to(self.device, non_blocking=self.device.type == "cuda") for t in v]
            else:
                result[k] = v
        return result

    def _forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """배치 딕셔너리에서 모델 forward를 호출한다."""
        # padding mask: seq_len으로부터 생성
        B = batch["pose"].shape[0]
        T = batch["pose"].shape[1]
        seq_len = batch["seq_len"]  # [B]
        mask = torch.arange(T, device=self.device).unsqueeze(0) >= seq_len.unsqueeze(1)  # True=pad

        kwargs: dict[str, Any] = {
            "pose": batch["pose"],
            "left_hand": batch["left_hand"],
            "right_hand": batch["right_hand"],
            "face_blendshape": batch["face_blendshape"],
            "face_key_subset": batch.get("face_key_subset"),
            "presence_mask": batch.get("presence_mask"),
            "src_key_padding_mask": mask,
        }
        if "left_hand_crop" in batch:
            kwargs["left_hand_crop"] = batch["left_hand_crop"]
            kwargs["right_hand_crop"] = batch["right_hand_crop"]
        if "tgt_tokens" in batch:
            kwargs["tgt_tokens"] = batch["tgt_tokens"]
            kwargs["tgt_padding"] = batch.get("tgt_padding")

        return self.model(**kwargs)
