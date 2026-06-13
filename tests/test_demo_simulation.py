import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_demo_simulation  # noqa: E402


def test_demo_simulation_runs_without_real_video(tmp_path, capsys):
    json_out = tmp_path / "demo_result.json"

    exit_code = run_demo_simulation.main(
        [
            "--video",
            str(tmp_path / "missing.mp4"),
            "--frames",
            "32",
            "--target-accuracy",
            "0.88",
            "--no-sleep",
            "--json-out",
            str(json_out),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "데모 입력으로 가정" in captured
    assert "수지=" in captured
    assert "비수지=" in captured
    assert "simulated_accuracy=0.88" in captured
    assert "강한 비와 바람이 예상되니 안전한 곳으로 대피하세요." in captured

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["simulation"] is True
    assert payload["prediction"]["simulated_accuracy"] == 0.88
    assert payload["crop"]["cropped_path"].endswith("cropped_signer_roi.mp4")
