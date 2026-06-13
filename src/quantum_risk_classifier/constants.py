from pathlib import Path

MODEL_VERSION = "qksvm_demo_v1"
SEED = 20260613

FEATURE_NAMES = [
    "estimated_amount_ratio",
    "holding_ratio_after_trade",
    "current_position_ratio",
    "risk_preference_code",
    "volatility_20d",
    "abs_return_5d",
    "turnover_rate",
    "trade_volume_to_avg_volume",
]
RISK_LABELS = ["concentration_risk", "volatility_risk", "liquidity_risk"]
RISK_PREFERENCE_CODES = {"conservative": 0, "moderate": 1, "aggressive": 2}
CONCENTRATION_THRESHOLDS = {"conservative": 0.20, "moderate": 0.30, "aggressive": 0.40}
OVERALL_WEIGHTS = {
    "concentration_risk": 0.40,
    "volatility_risk": 0.35,
    "liquidity_risk": 0.25,
}

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
EXAMPLES = ROOT / "examples"

