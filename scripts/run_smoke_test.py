#!/usr/bin/env python3
"""빠른 CPU smoke test 스크립트.

데이터가 없어도 전체 파이프라인이 동작하는지 확인한다.
사용법: python scripts/run_smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dummy import DummyDataset, collate_fn, make_dummy_batch
from src.eval.evaluator import KSLEvaluator
from src.infer.pipeline import InferencePipeline
from src.llm.corrector import ContextCorrector
from src.models.decoder import DecoderConfig
from src.models.fusion import FusionConfig
from src.models.ksl_model import KSLModel, ModelConfig
from src.train.trainer import KSLTrainer, TrainerConfig


def run():
    print("=" * 60)
    print("KSL Pipeline Smoke Test")
    print("=" * 60)

    for stage in ["C"]:
        print(f"\n[Stage {stage}]")

        # Dataset
        train_ds = DummyDataset(split_group="train", num_samples=4, max_len=16)
        val_ds = DummyDataset(split_group="valid", num_samples=4, max_len=16)
        train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn)

        # Model (cross-attention fusion: E1=Query, E2&E3=Key/Value)
        model = KSLModel(ModelConfig(
            stage=stage,
            fusion=FusionConfig(method="cross_attention"),
            decoder=DecoderConfig(max_len=16),
        ))
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

        # Train 1 epoch
        trainer = KSLTrainer(
            model,
            TrainerConfig(stage=stage, max_epochs=1, device="cpu"),
            train_loader,
            val_loader,
        )
        train_loss = trainer.train_epoch()
        val_loss = trainer.val_epoch()
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Inference
        pipeline = InferencePipeline(model, device="cpu")
        batch = make_dummy_batch(batch_size=1, max_len=16)
        result = pipeline.infer(batch, domain="hospital")
        print(f"  infer confidence={result.confidence:.3f}  final_text='{result.final_text}'")

        # Eval
        evaluator = KSLEvaluator(model, device="cpu")
        eval_result = evaluator.evaluate(val_loader, split="valid")
        print(f"  intent_acc={eval_result.intent_accuracy:.3f}")

    print("\n" + "=" * 60)
    print("All stages passed!")
    print("=" * 60)


if __name__ == "__main__":
    run()
