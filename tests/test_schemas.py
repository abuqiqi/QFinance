import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quantum_risk_classifier.schemas import InvestmentPlanInput


ROOT = Path(__file__).resolve().parents[1]


def test_valid_input():
    plan = InvestmentPlanInput.model_validate_json((ROOT / "examples/input_valid.json").read_text())
    assert plan.trade_plan.stock_code == "600006"
    assert plan.trade_plan.market is None


@pytest.mark.parametrize("market", ["SSE", "SZSE"])
def test_market_is_optional_but_valid_when_supplied(market):
    payload = json.loads((ROOT / "examples/input_valid.json").read_text())
    payload["trade_plan"]["market"] = market
    plan = InvestmentPlanInput.model_validate(payload)
    assert plan.trade_plan.market == market


def test_invalid_market_is_rejected():
    payload = json.loads((ROOT / "examples/input_valid.json").read_text())
    payload["trade_plan"]["market"] = "NYSE"
    with pytest.raises(ValidationError, match="market"):
        InvestmentPlanInput.model_validate(payload)


def test_invalid_input_reports_fields():
    with pytest.raises(ValidationError) as error:
        InvestmentPlanInput.model_validate_json((ROOT / "examples/input_invalid.json").read_text())
    assert "total_asset" in str(error.value)
    assert "risk_preference" in str(error.value)
