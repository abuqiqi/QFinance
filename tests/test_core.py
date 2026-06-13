import math

from quantum_risk_classifier.core import label_from_values, overall_level, overall_score


def test_label_thresholds_are_inclusive():
    labels = label_from_values("moderate", 0.30, 0.35, 0.01, 0.10)
    assert labels == {"concentration_risk": 1, "volatility_risk": 1, "liquidity_risk": 1}


def test_overall_rules():
    labels = {"concentration_risk": 1, "volatility_risk": 1, "liquidity_risk": 0}
    scores = {"concentration_risk": 1.0, "volatility_risk": 0.5, "liquidity_risk": 0.0}
    assert overall_level(labels) == "medium_high"
    assert math.isclose(overall_score(scores), 0.575)

