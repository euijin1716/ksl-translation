"""End-to-end smoke test.

실제 데이터나 MediaPipe 없이 CPU에서 동작해야 한다.
dummy sample 구조는 실제 schema와 동일하다.
"""

import pytest
import torch
from torch.utils.data import DataLoader


# ── 1. Schema / Data ──────────────────────────────────────────────────────────

class TestSchema:
    def test_ksl_sample_valid(self):
        from src.data.schema import KSLSample, NMSLabels
        s = KSLSample(
            sample_id="test_0001",
            dataset_name="test",
            domain="hospital",
            scenario_id="sc_001",
            turn_id=0,
            utterance_id="utt_0001",
            signer_id="S001",
            split_group="train",
            video_path="data/test/001.mp4",
            fps=25.0,
            num_frames=60,
            korean_text="두통이 있습니다.",
            gloss_tokens=["머리", "아프다"],
            nms_labels=NMSLabels(eyebrow_raise=True),
            intent="hospital",
            intent_source="gold",
        )
        assert s.sample_id == "test_0001"
        assert s.domain == "hospital"

    def test_ksl_sample_invalid_domain(self):
        from src.data.schema import KSLSample
        with pytest.raises(ValueError, match="domain"):
            KSLSample(
                sample_id="x", dataset_name="x", domain="invalid_domain",
                scenario_id="x", turn_id=0, utterance_id="x",
                signer_id="S001", split_group="train",
                video_path="x", fps=25.0, num_frames=30,
                korean_text="x", gloss_tokens=None, nms_labels=None,
                intent=None, intent_source="gold",
            )

    def test_nms_labels_to_dict(self):
        from src.data.schema import NMSLabels
        nms = NMSLabels(eyebrow_raise=True, mouth_open=False)
        d = nms.to_dict()
        assert "eyebrow_raise" in d
        assert d["eyebrow_raise"] is True

    def test_nms_labels_roundtrip(self):
        from src.data.schema import NMSLabels
        nms = NMSLabels(eyebrow_raise=True, head_nod=False, mouth_shape="아")
        restored = NMSLabels.from_dict(nms.to_dict())
        assert restored.eyebrow_raise is True
        assert restored.mouth_shape == "아"

    def test_nms_detail_labels(self):
        from src.data.schema import NMSLabels
        from src.data.signals import NMS_DETAIL_CLASSES, encode_nms_detail_labels

        labels, mask = encode_nms_detail_labels(
            NMSLabels(
                eyebrow_raise=True,
                eye_squint=True,
                mouth_shape="a",
                head_shake=True,
                gaze_direction="left",
            )
        )

        assert labels.shape == (5,)
        assert mask.shape == (5,)
        assert mask.sum().item() == 5
        assert labels[0].item() == NMS_DETAIL_CLASSES["eyebrow"].index("raise")
        assert labels[1].item() == NMS_DETAIL_CLASSES["eye"].index("squint")
        assert labels[2].item() == NMS_DETAIL_CLASSES["mouth_shape"].index("a")
        assert labels[3].item() == NMS_DETAIL_CLASSES["head_movement"].index("shake")
        assert labels[4].item() == NMS_DETAIL_CLASSES["gaze_direction"].index("left")


# ── 2. Dummy Dataset / Collator ───────────────────────────────────────────────

class TestDummyDataset:
    def test_dummy_dataset_len(self):
        from src.data.dummy import DummyDataset
        ds = DummyDataset(split_group="train", num_samples=20)
        assert len(ds) > 0

    def test_dummy_dataset_item_keys(self):
        from src.data.dummy import DummyDataset
        ds = DummyDataset(split_group="train", num_samples=20)
        item = ds[0]
        required_keys = [
            "sample_id", "pose", "left_hand", "right_hand",
            "face_blendshape", "face_key_subset", "presence_mask",
            "left_hand_crop", "right_hand_crop",
            "gloss_ids", "nms_label", "nms_mask", "nms_detail_label",
            "nms_detail_mask", "intent_label", "activity", "seq_len",
        ]
        for k in required_keys:
            assert k in item, f"Missing key: {k}"

    def test_dummy_dataset_shapes(self):
        from src.data.dummy import DummyDataset
        ds = DummyDataset(split_group="train", num_samples=20, max_len=32)
        item = ds[0]
        T = item["seq_len"].item()
        assert item["pose"].shape == (T, 25, 3)
        assert item["left_hand"].shape == (T, 21, 3)
        assert item["right_hand"].shape == (T, 21, 3)
        assert item["face_blendshape"].shape == (T, 52)
        assert item["face_key_subset"].shape == (T, 68, 3)
        assert item["left_hand_crop"].shape == (T, 3, 112, 112)
        assert item["right_hand_crop"].shape == (T, 3, 112, 112)
        assert item["nms_label"].shape == (12,)
        assert item["nms_mask"].shape == (12,)
        assert item["nms_detail_label"].shape == (5,)
        assert item["nms_detail_mask"].shape == (5,)

    def test_signer_independent_split(self):
        from src.data.dummy import DummyDataset
        from src.data.splits import check_signer_leakage
        from src.data.adapters.dummy_adapter import DummyAdapter
        from src.data.splits import make_signer_independent_split

        adapter = DummyAdapter(num_samples=20, seed=42)
        samples, manifest = make_signer_independent_split(list(adapter.iter_samples()), dataset_name="dummy")
        leakage = check_signer_leakage(samples)
        assert leakage == [], f"Signer leakage found: {leakage}"

    def test_collate_fn(self):
        from src.data.dummy import DummyDataset, collate_fn
        ds = DummyDataset(split_group="train", num_samples=20)
        batch = collate_fn([ds[0], ds[1]])
        assert "pose" in batch
        assert batch["pose"].shape[0] == 2
        assert batch["nms_detail_label"].shape == (2, 5)
        assert batch["nms_detail_mask"].shape == (2, 5)

    def test_collate_fn_draft_tokens(self):
        """tgt_tokens / draft_labels / tgt_padding이 올바르게 패딩되는지 확인."""
        from src.data.dummy import DummyDataset, collate_fn
        ds = DummyDataset(split_group="train", num_samples=20,
                          decoder_vocab_size=32000, decoder_bos_id=0,
                          decoder_eos_id=2, decoder_pad_id=1)
        batch = collate_fn([ds[0], ds[1]])
        assert "tgt_tokens" in batch, "tgt_tokens missing from batch"
        assert "draft_labels" in batch, "draft_labels missing from batch"
        assert "tgt_padding" in batch, "tgt_padding missing from batch"
        B, L = batch["tgt_tokens"].shape
        assert B == 2
        assert batch["draft_labels"].shape == (B, L)
        assert batch["tgt_padding"].shape == (B, L)
        assert batch["tgt_padding"].dtype == torch.bool
        # BOS 토큰이 첫 위치에 있는지 확인
        assert (batch["tgt_tokens"][:, 0] == 0).all(), "First token must be BOS(0)"

    def test_decoder_config_from_tokenizer(self):
        """DecoderConfig.from_tokenizer가 특수토큰 ID를 올바르게 가져오는지 확인."""
        from src.models.decoder import DecoderConfig

        class _FakeTok:
            vocab_size = 32000
            bos_token_id = 0
            eos_token_id = 2
            pad_token_id = 1
            cls_token_id = None
            sep_token_id = None

        cfg = DecoderConfig.from_tokenizer(_FakeTok(), d_model=128)
        assert cfg.vocab_size == 32000
        assert cfg.bos_token_id == 0
        assert cfg.eos_token_id == 2
        assert cfg.pad_token_id == 1
        assert cfg.d_model == 128


# ── 3. Manifest ───────────────────────────────────────────────────────────────

class TestManifest:
    def test_write_read_manifest(self, tmp_path):
        from src.data.adapters.dummy_adapter import DummyAdapter
        from src.data.manifest import write_manifest, read_manifest
        adapter = DummyAdapter(num_samples=5)
        samples = list(adapter.iter_samples())
        path = tmp_path / "test_manifest.jsonl"
        write_manifest(samples, path)
        loaded = list(read_manifest(path))
        assert len(loaded) == 5
        assert loaded[0].sample_id == samples[0].sample_id


# ── 4. Extractor (dummy mode) ─────────────────────────────────────────────────

class TestExtractor:
    def test_mediapipe_extractor_dummy_mode(self):
        from src.preprocess.extractors.mediapipe_extractor import MediaPipeExtractor
        extractor = MediaPipeExtractor()
        result = extractor.extract("nonexistent_video.mp4")
        # dummy mode에서는 None이 아닌 numpy 배열이 반환되어야 함
        assert result.pose is not None
        assert result.left_hand is not None
        assert result.right_hand is not None
        assert result.face_blendshape is not None
        assert result.face_blendshape.shape[-1] == 52

    def test_extractor_shapes(self):
        from src.preprocess.extractors.mediapipe_extractor import MediaPipeExtractor
        extractor = MediaPipeExtractor()
        result = extractor.extract("test.mp4")
        T = result.pose.shape[0]
        assert result.pose.shape == (T, 25, 3)
        assert result.left_hand.shape == (T, 21, 3)
        assert result.right_hand.shape == (T, 21, 3)


# ── 5. Normalizer ─────────────────────────────────────────────────────────────

class TestNormalizer:
    def test_normalize_bbox(self):
        import numpy as np
        from src.preprocess.normalizers.coordinate_normalizer import normalize_landmarks
        lm = np.random.rand(10, 21, 3).astype("float32")
        out, meta = normalize_landmarks(lm, method="bbox")
        assert out.shape == lm.shape
        assert meta["scale_by"] == "bbox"

    def test_normalize_none(self):
        import numpy as np
        from src.preprocess.normalizers.coordinate_normalizer import normalize_landmarks
        lm = np.random.rand(5, 21, 3).astype("float32")
        out, meta = normalize_landmarks(lm, method="none")
        assert out.shape == lm.shape


# ── 6. Validator ──────────────────────────────────────────────────────────────

class TestValidator:
    def test_validate_pass(self):
        import numpy as np
        from src.preprocess.extractors.base_extractor import ExtractionResult
        from src.preprocess.validators.shape_validator import validate_extraction_result
        result = ExtractionResult(
            pose=np.zeros((30, 25, 3), dtype="float32"),
            left_hand=np.zeros((30, 21, 3), dtype="float32"),
            right_hand=np.zeros((30, 21, 3), dtype="float32"),
            face_blendshape=np.zeros((30, 52), dtype="float32"),
            presence_mask=np.ones((30, 4), dtype=bool),
        )
        report = validate_extraction_result("test_001", result)
        assert report.passed

    def test_validate_fail_shape(self):
        import numpy as np
        from src.preprocess.extractors.base_extractor import ExtractionResult
        from src.preprocess.validators.shape_validator import validate_extraction_result
        result = ExtractionResult(
            pose=np.zeros((30, 33, 3), dtype="float32"),  # wrong: 33 instead of 25
        )
        report = validate_extraction_result("test_002", result)
        assert not report.passed


# ── 7. Model Forward Pass ─────────────────────────────────────────────────────

class TestModel:
    def _make_batch(self, B=2, T=16):
        return {
            "pose": torch.randn(B, T, 25, 3),
            "left_hand": torch.randn(B, T, 21, 3),
            "right_hand": torch.randn(B, T, 21, 3),
            "face_blendshape": torch.randn(B, T, 52),
            "face_key_subset": torch.randn(B, T, 68, 3),
            "presence_mask": torch.ones(B, T, 4),
        }

    def test_stage_c_forward(self):
        from src.models.ksl_model import KSLModel, ModelConfig
        B, T, L = 2, 16, 8
        from src.models.decoder import DecoderConfig
        model = KSLModel(ModelConfig(stage="C", decoder=DecoderConfig(max_len=8)))
        model.eval()
        batch = self._make_batch(B, T)
        batch["left_hand_crop"] = torch.randn(B, T, 3, 112, 112)
        batch["right_hand_crop"] = torch.randn(B, T, 3, 112, 112)
        batch["tgt_tokens"] = torch.randint(1, 100, (B, L))
        with torch.no_grad():
            out = model(**batch)
        assert "gloss_logits" in out
        assert "nms_logits" in out
        assert "nms_eyebrow_logits" in out
        assert "nms_eye_logits" in out
        assert "nms_mouth_shape_logits" in out
        assert "nms_head_movement_logits" in out
        assert "nms_gaze_direction_logits" in out
        assert "intent_logits" in out
        assert "boundary_logits" in out
        assert "draft_logits" in out
        assert out["gloss_logits"].shape == (B, T, 1001)
        assert out["nms_logits"].shape == (B, T, 12)
        assert out["nms_eyebrow_logits"].shape == (B, T, 4)
        assert out["nms_eye_logits"].shape == (B, T, 4)
        assert out["nms_mouth_shape_logits"].shape == (B, T, 11)
        assert out["nms_head_movement_logits"].shape == (B, T, 5)
        assert out["nms_gaze_direction_logits"].shape == (B, T, 7)
        assert out["intent_logits"].shape == (B, 7)
        assert out["boundary_logits"].shape == (B, T, 3)
        assert out["draft_logits"].shape == (B, L, 32000)


# ── 8. Trainer Smoke Test ─────────────────────────────────────────────────────

class TestTrainer:
    def test_trainer_one_step(self):
        from src.data.dummy import DummyDataset, collate_fn
        from src.models.ksl_model import KSLModel, ModelConfig
        from src.train.trainer import KSLTrainer, TrainerConfig

        dataset = DummyDataset(split_group="train", num_samples=4, max_len=16)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        from src.models.decoder import DecoderConfig
        model = KSLModel(ModelConfig(stage="C", decoder=DecoderConfig(max_len=8)))
        config = TrainerConfig(stage="C", max_epochs=1, device="cpu")
        trainer = KSLTrainer(model, config, loader)
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN


# ── 9. LLM Corrector ──────────────────────────────────────────────────────────

class TestLLMCorrector:
    def test_dummy_corrector(self):
        from src.llm.corrector import ContextCorrector
        corrector = ContextCorrector()
        result = corrector.correct(
            korean_draft="두통이 있습니다.",
            gloss_hypotheses=["머리", "아프다"],
            nms_summary={"eyebrow_raise": 0.8},
            confidence=0.85,
            domain="hospital",
        )
        assert result.final_text == "두통이 있습니다."
        assert not result.retry_or_clarify

    def test_response_parser_valid_json(self):
        from src.llm.response_parser import parse_response
        raw = '{"final_text": "테스트", "retry_or_clarify": false, "uncertain_spans": []}'
        out = parse_response(raw, fallback_text="fallback")
        assert out.final_text == "테스트"

    def test_response_parser_fallback(self):
        from src.llm.response_parser import parse_response
        out = parse_response("invalid json @@@ !!!", fallback_text="fallback text")
        assert out.final_text == "fallback text"

    def test_prompt_builder(self):
        from src.llm.provider import LLMInput
        from src.llm.prompt_builder import build_prompt
        llm_input = LLMInput(
            korean_draft="배가 아픕니다.",
            top_k_gloss=["배", "아프다"],
            nms_summary={"mouth_open": 0.7},
            confidence=0.75,
            previous_turns=["두통이 있습니다."],
            domain="hospital",
        )
        prompt = build_prompt(llm_input)
        assert "배가 아픕니다." in prompt
        assert "hospital" in prompt


# ── 10. Eval Metrics ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_bleu_perfect(self):
        from src.eval.metrics import compute_bleu
        hyps = ["두통이 있습니다"]
        refs = ["두통이 있습니다"]
        result = compute_bleu(hyps, refs)
        assert result["bleu"] > 0

    def test_wer_perfect(self):
        from src.eval.metrics import compute_wer
        assert compute_wer([["a", "b"]], [["a", "b"]]) == 0.0

    def test_wer_all_wrong(self):
        from src.eval.metrics import compute_wer
        assert compute_wer([["x"]], [["a", "b"]]) > 0.0

    def test_accuracy(self):
        from src.eval.metrics import compute_accuracy
        assert compute_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
        assert compute_accuracy([0, 0, 0], [0, 1, 2]) == pytest.approx(1/3)

    def test_f1(self):
        from src.eval.metrics import compute_f1
        result = compute_f1([0, 1, 0, 1], [0, 1, 1, 0], num_classes=2)
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0


# ── 11. Inference Pipeline ────────────────────────────────────────────────────

class TestInferencePipeline:
    def test_pipeline_infer(self):
        from src.data.dummy import make_dummy_batch
        from src.infer.pipeline import InferencePipeline
        from src.models.decoder import DecoderConfig
        from src.models.ksl_model import KSLModel, ModelConfig

        model = KSLModel(ModelConfig(stage="C", decoder=DecoderConfig(max_len=8)))
        pipeline = InferencePipeline(model, device="cpu")
        batch = make_dummy_batch(batch_size=1, max_len=16)
        # list of tensors → stack for inference
        batch["gloss_ids"] = batch["gloss_ids"][0]

        result = pipeline.infer(batch, domain="hospital")
        assert isinstance(result.final_text, str)
        assert 0.0 <= result.confidence <= 1.0
        assert result.activity_state in ("idle", "ongoing", "ended")


# ── 12. Streaming State Machine ───────────────────────────────────────────────

class TestStreamingStateMachine:
    def test_state_transitions(self):
        from src.infer.streaming import StreamingStateMachine, ActivityState, StreamingConfig
        sm = StreamingStateMachine(StreamingConfig(onset_threshold=0.6, offset_threshold=0.4, min_sign_frames=3))

        # push idle frames
        for _ in range(3):
            r = sm.push_frame(None, 0.2)
        assert sm.state == ActivityState.IDLE

        # onset
        r = sm.push_frame(None, 0.7)
        assert sm.state in (ActivityState.ONSET, ActivityState.ONGOING)

    def test_reset(self):
        from src.infer.streaming import StreamingStateMachine, ActivityState
        sm = StreamingStateMachine()
        sm.push_frame(None, 0.9)
        sm.reset()
        assert sm.state == ActivityState.IDLE


# ── 13. End-to-End Smoke Test ─────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_smoke(self):
        """dummy data → data loader → model forward → LLM correct → final output."""
        from src.data.dummy import DummyDataset, collate_fn, make_dummy_batch
        from src.eval.evaluator import KSLEvaluator
        from src.infer.pipeline import InferencePipeline
        from src.models.decoder import DecoderConfig
        from src.models.ksl_model import KSLModel, ModelConfig
        from src.train.trainer import KSLTrainer, TrainerConfig

        # 1. Dataset
        dataset = DummyDataset(split_group="train", num_samples=4, max_len=16)
        assert len(dataset) > 0

        # 2. DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        # 3. Model
        model = KSLModel(ModelConfig(stage="C", decoder=DecoderConfig(max_len=8)))

        # 4. Trainer - 1 epoch
        trainer = KSLTrainer(
            model,
            TrainerConfig(stage="C", max_epochs=1, device="cpu"),
            loader,
        )
        loss = trainer.train_epoch()
        assert not (loss != loss), "Loss is NaN"

        # 5. Evaluator
        evaluator = KSLEvaluator(model, device="cpu")
        eval_result = evaluator.evaluate(loader, split="train")
        assert eval_result.num_samples >= 0

        # 6. Inference
        pipeline = InferencePipeline(model, device="cpu")
        batch = make_dummy_batch(batch_size=1, max_len=16)
        infer_result = pipeline.infer(batch, domain="hospital")
        assert isinstance(infer_result.final_text, str)

        print(f"\n[Smoke Test] loss={loss:.4f} | intent_acc={eval_result.intent_accuracy:.3f}")


# ── 10. Real Dataset Adapters (데이터 없을 때는 빈 결과 확인) ─────────────────

class TestAdapters:
    """데이터가 없어도 adapter가 경고만 출력하고 빈 이터레이터를 반환하는지 확인."""

    def test_niasl2021_missing_root(self, tmp_path):
        from src.data.adapters import NIASL2021Adapter
        adapter = NIASL2021Adapter(root=tmp_path / "nonexistent_niasl2021")
        samples = list(adapter.iter_samples())
        assert samples == [], "데이터 없을 때 빈 리스트여야 한다"

    def test_aihub_sign_missing_root(self, tmp_path):
        from src.data.adapters import AIHubSignAdapter
        adapter = AIHubSignAdapter(root=tmp_path / "nonexistent_aihub_sign")
        samples = list(adapter.iter_samples())
        assert samples == []

    def test_aihub_disaster_missing_root(self, tmp_path):
        from src.data.adapters import AIHubDisasterAdapter
        adapter = AIHubDisasterAdapter(root=tmp_path / "nonexistent_aihub_disaster")
        samples = list(adapter.iter_samples())
        assert samples == []

    def test_niasl2021_manifest_json(self, tmp_path):
        """train.json 매니페스트 포맷을 올바르게 파싱하는지 확인."""
        import json
        from src.data.adapters import NIASL2021Adapter

        # 최소 유효 레코드
        records = [
            {
                "id": "N2021_T_000001",
                "signer": "P01",
                "korean_text": "서울 날씨가 맑겠습니다.",
                "gloss": "서울 날씨 맑다",
                "domain": "weather",
                "fps": 25,
                "num_frames": 125,
            },
            {
                "id": "N2021_T_000002",
                "signer": "P02",
                "korean_text": "긴급 대피 하세요.",
                "gloss": "긴급 대피",
                "domain": "emergency",
                "fps": 25,
                "num_frames": 80,
            },
        ]
        (tmp_path / "train.json").write_text(json.dumps(records), encoding="utf-8")
        (tmp_path / "videos").mkdir()

        adapter = NIASL2021Adapter(root=tmp_path, splits=["train"])
        samples = list(adapter.iter_samples())

        assert len(samples) == 2, f"2개 샘플을 기대했지만 {len(samples)}개"
        assert samples[0].signer_id == "P01"
        assert samples[0].korean_text == "서울 날씨가 맑겠습니다."
        assert samples[0].gloss_tokens == ["서울", "날씨", "맑다"]
        assert samples[0].domain == "directions"      # weather → directions 매핑
        assert samples[1].domain == "help"            # emergency → help 매핑
        # signer_id가 서로 다름 (signer-independent split 가능 조건)
        assert samples[0].signer_id != samples[1].signer_id

    def test_niasl2021_missing_korean_text_skipped(self, tmp_path):
        """korean_text가 없는 레코드는 건너뛰는지 확인."""
        import json
        from src.data.adapters import NIASL2021Adapter

        records = [
            {"id": "N001", "signer": "P01", "korean_text": "안녕하세요.", "num_frames": 50},
            {"id": "N002", "signer": "P02"},  # korean_text 없음 → 건너뜀
        ]
        (tmp_path / "train.json").write_text(json.dumps(records), encoding="utf-8")

        adapter = NIASL2021Adapter(root=tmp_path, splits=["train"])
        samples = list(adapter.iter_samples())
        assert len(samples) == 1

    def test_aihub_sign_json_parsing(self, tmp_path):
        """AI Hub DataInfo + Annotation JSON 포맷을 올바르게 파싱하는지 확인."""
        import json
        from src.data.adapters import AIHubSignAdapter

        label_dir = tmp_path / "Training" / "라벨링데이터" / "인사"
        label_dir.mkdir(parents=True)
        video_dir = tmp_path / "Training" / "원천데이터" / "인사"
        video_dir.mkdir(parents=True)

        label_data = {
            "DataInfo": {
                "VideoName": "P001_감사합니다_001.mp4",
                "FrameRate": 30.0,
                "TotalFrame": 90,
                "SignerID": "P001",
                "Gender": "F",
                "Category": "인사",
                "KoreanText": "감사합니다",
            },
            "Annotation": [
                {"SignGloss": "감사하다", "StartFrame": 5, "EndFrame": 85}
            ],
        }
        (label_dir / "P001_감사합니다_001.json").write_text(
            json.dumps(label_data, ensure_ascii=False), encoding="utf-8"
        )
        # 더미 영상 파일 (빈 파일)
        (video_dir / "P001_감사합니다_001.mp4").touch()

        adapter = AIHubSignAdapter(root=tmp_path)
        samples = list(adapter.iter_samples())

        assert len(samples) == 1
        s = samples[0]
        assert s.signer_id == "P001"
        assert s.korean_text == "감사합니다"
        assert s.gloss_tokens == ["감사하다"]
        assert s.domain == "unknown"          # 인사 → unknown
        assert "aihub_sign" in s.sample_id
        # video_path가 절대경로가 아닌지 확인
        assert not s.video_path.startswith("/")

    def test_aihub_disaster_nms_parsing(self, tmp_path):
        """재난안전 JSON에서 NMS가 올바르게 파싱되는지 확인."""
        import json
        from src.data.adapters import AIHubDisasterAdapter

        label_dir = tmp_path / "Training" / "라벨링데이터" / "화재"
        label_dir.mkdir(parents=True)

        label_data = {
            "DataInfo": {
                "VideoName": "P001_화재_대피_001.mp4",
                "FrameRate": 30.0,
                "TotalFrame": 120,
                "SignerID": "P001",
                "Category": "화재",
                "KoreanText": "불이 났습니다. 빨리 대피하세요.",
            },
            "Annotation": [
                {"SignGloss": "불", "StartFrame": 5, "EndFrame": 30},
                {"SignGloss": "대피", "StartFrame": 35, "EndFrame": 80},
            ],
            "NMS": {
                "facial_expression": "urgent",
                "head_movement": "shake",
            },
        }
        (label_dir / "P001_화재_대피_001.json").write_text(
            json.dumps(label_data, ensure_ascii=False), encoding="utf-8"
        )

        adapter = AIHubDisasterAdapter(root=tmp_path)
        samples = list(adapter.iter_samples())

        assert len(samples) == 1
        s = samples[0]
        assert s.domain == "help"             # 화재 → help
        assert s.gloss_tokens == ["불", "대피"]
        assert s.nms_labels is not None, "NMS가 파싱되어야 한다"
        assert s.nms_labels.head_shake is True
        assert s.metadata["has_nms"] is True
