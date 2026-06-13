from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserProfile(StrictModel):
    user_id: str = Field(min_length=1)
    total_asset: float = Field(gt=0)
    cash_available: float = Field(ge=0)
    risk_preference: Literal["conservative", "moderate", "aggressive"]
    current_stock_position_value: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_finances(self):
        if self.cash_available > self.total_asset:
            raise ValueError("cash_available cannot exceed total_asset")
        if self.current_stock_position_value > self.total_asset:
            raise ValueError("current_stock_position_value cannot exceed total_asset")
        return self


class TradePlan(StrictModel):
    action: Literal["buy"]
    market: Optional[Literal["SSE", "SZSE"]] = None
    stock_code: str = Field(pattern=r"^\d{6}$")
    trade_time: str = Field(min_length=19)
    volume_lot: int = Field(gt=0)
    estimated_price: float = Field(gt=0)


class StockFeatures(StrictModel):
    volatility_20d: float = Field(ge=0, le=2)
    return_5d: float = Field(ge=-1, le=1)
    avg_volume_20d: float = Field(gt=0)
    turnover_rate: float = Field(ge=0, le=1)


class InvestmentPlanInput(StrictModel):
    sample_id: str = Field(min_length=1)
    user_profile: UserProfile
    trade_plan: TradePlan
    stock_features: StockFeatures


class InputSummary(StrictModel):
    stock_code: str
    action: str
    volume_lot: int
    volume_share: int
    estimated_amount: float


class OverallRisk(StrictModel):
    overall_score: float = Field(ge=0, le=1)
    overall_level: Literal["low", "medium", "medium_high", "high"]


class ModelOutputs(StrictModel):
    classifier_type: str
    feature_map: str
    num_qubits: int
    feature_dimension: int
    risk_configs: Optional[Dict[str, Dict[str, Any]]] = None


class PredictionOutput(StrictModel):
    sample_id: str
    model_version: str
    input_summary: InputSummary
    risk_scores: Dict[str, float]
    risk_labels: Dict[str, int]
    overall_risk: OverallRisk
    confidence: float = Field(ge=0, le=1)
    model_outputs: ModelOutputs
