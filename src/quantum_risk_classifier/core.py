import math
from typing import Dict, Tuple

import numpy as np

from .constants import (
    CONCENTRATION_THRESHOLDS,
    FEATURE_NAMES,
    OVERALL_WEIGHTS,
    RISK_PREFERENCE_CODES,
)
from .schemas import InvestmentPlanInput


def derive_values(plan: InvestmentPlanInput) -> Dict[str, float]:
    volume_share = plan.trade_plan.volume_lot * 100
    estimated_amount = volume_share * plan.trade_plan.estimated_price
    profile = plan.user_profile
    stock = plan.stock_features
    current_position_ratio = profile.current_stock_position_value / profile.total_asset
    return {
        "volume_share": volume_share,
        "estimated_amount": estimated_amount,
        "estimated_amount_ratio": estimated_amount / profile.total_asset,
        "current_position_ratio": current_position_ratio,
        "holding_ratio_after_trade": (
            profile.current_stock_position_value + estimated_amount
        )
        / profile.total_asset,
        "abs_return_5d": abs(stock.return_5d),
        "trade_volume_to_avg_volume": volume_share / stock.avg_volume_20d,
    }


def label_from_values(
    risk_preference: str,
    holding_ratio_after_trade: float,
    volatility_20d: float,
    abs_return_5d: float,
    trade_volume_to_avg_volume: float,
) -> Dict[str, int]:
    return {
        "concentration_risk": int(
            holding_ratio_after_trade >= CONCENTRATION_THRESHOLDS[risk_preference]
        ),
        "volatility_risk": int(volatility_20d >= 0.35 or abs_return_5d >= 0.15),
        "liquidity_risk": int(trade_volume_to_avg_volume >= 0.10),
    }


def overall_level(labels: Dict[str, int]) -> str:
    return ["low", "medium", "medium_high", "high"][sum(labels.values())]


def overall_score(scores: Dict[str, float]) -> float:
    return float(sum(scores[name] * weight for name, weight in OVERALL_WEIGHTS.items()))


def confidence_from_scores(scores: Dict[str, float]) -> float:
    return float(np.mean([min(1.0, abs(score - 0.5) * 2) for score in scores.values()]))


def feature_vector(plan: InvestmentPlanInput) -> Tuple[np.ndarray, Dict[str, float]]:
    values = derive_values(plan)
    stock = plan.stock_features
    values.update(
        {
            "risk_preference_code": RISK_PREFERENCE_CODES[plan.user_profile.risk_preference],
            "volatility_20d": stock.volatility_20d,
            "turnover_rate": stock.turnover_rate,
        }
    )
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float), values


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def model_scores(model, values):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(values)[:, 1]
    return sigmoid(model.decision_function(values))
