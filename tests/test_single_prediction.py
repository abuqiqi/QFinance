import copy
import json
import subprocess
import sys
from pathlib import Path

from quantum_risk_classifier.constants import ARTIFACTS, DATA_PROCESSED
from quantum_risk_classifier.schemas import InvestmentPlanInput, PredictionOutput
from quantum_risk_classifier.service import predict


ROOT = Path(__file__).resolve().parents[1]
VALID_INPUT = ROOT / "examples" / "input_valid.json"
INVALID_INPUT = ROOT / "examples" / "input_invalid.json"


def test_service_predicts_single_input_deterministically():
    plan = InvestmentPlanInput.model_validate_json(VALID_INPUT.read_text(encoding="utf-8"))

    first = predict(plan, DATA_PROCESSED, ARTIFACTS)
    second = predict(plan, DATA_PROCESSED, ARTIFACTS)

    assert first == second
    assert set(first.risk_scores) == {
        "concentration_risk", "volatility_risk", "liquidity_risk"
    }
    assert all(0.0 <= score <= 1.0 for score in first.risk_scores.values())
    assert all(label in {0, 1} for label in first.risk_labels.values())
    assert first.overall_risk.overall_level in {
        "low", "medium", "medium_high", "high"
    }
    assert first.model_outputs.classifier_type == "quantum_kernel_svm_optimized"
    assert first.model_outputs.risk_configs is not None


def test_market_does_not_change_prediction():
    payload = json.loads(VALID_INPUT.read_text(encoding="utf-8"))
    without_market = InvestmentPlanInput.model_validate(payload)
    payload_with_market = copy.deepcopy(payload)
    payload_with_market["trade_plan"]["market"] = "SSE"
    with_market = InvestmentPlanInput.model_validate(payload_with_market)

    result_without_market = predict(without_market, DATA_PROCESSED, ARTIFACTS)
    result_with_market = predict(with_market, DATA_PROCESSED, ARTIFACTS)

    assert result_without_market == result_with_market


def test_cli_writes_single_prediction_file(tmp_path):
    output_path = tmp_path / "prediction.json"
    command = [
        sys.executable, "-m", "quantum_risk_classifier.cli", "predict",
        "--input", str(VALID_INPUT), "--output", str(output_path),
    ]

    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert output_path.exists()
    output = PredictionOutput.model_validate_json(output_path.read_text(encoding="utf-8"))
    stdout_output = PredictionOutput.model_validate_json(first.stdout)
    assert output == stdout_output

    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    repeated = PredictionOutput.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert repeated == output


def test_cli_rejects_invalid_single_input():
    command = [
        sys.executable, "-m", "quantum_risk_classifier.cli", "predict",
        "--input", str(INVALID_INPUT),
    ]

    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    error = json.loads(result.stderr)
    assert error["error"] == "ValidationError"
    assert "total_asset" in error["detail"]
    assert "risk_preference" in error["detail"]
