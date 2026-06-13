import json

import joblib
import numpy as np
import pandas as pd

from quantum_risk_classifier.constants import FEATURE_NAMES
from quantum_risk_classifier.data_generation import generate_dataset, write_dataset
from quantum_risk_classifier.preprocessing import prepare_data
from quantum_risk_classifier.classical import train_classical


def test_generation_is_reproducible():
    assert generate_dataset(10, 7) == generate_dataset(10, 7)


def test_generated_derived_values_match_raw_fields():
    record = generate_dataset(1, 9)[0]
    profile = record["user_profile"]
    trade = record["trade_plan"]
    derived = record["derived_features"]
    assert trade["volume_share"] == trade["volume_lot"] * 100
    assert np.isclose(trade["estimated_amount"], trade["volume_share"] * trade["estimated_price"], atol=0.01)
    assert np.isclose(
        derived["holding_ratio_after_trade"],
        (profile["current_stock_position_value"] + trade["estimated_amount"]) / profile["total_asset"],
        atol=1e-6,
    )


def test_write_and_prepare(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    records = generate_dataset(160, 7)
    profile = write_dataset(records, raw)
    metadata = prepare_data(raw / "investment_samples.csv", processed, 7)
    assert profile["sample_count"] == 160
    ids = [set(metadata["splits"][name]["sample_ids"]) for name in ["train", "val", "test"]]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    train = pd.read_csv(processed / "train.csv")
    assert train[FEATURE_NAMES].to_numpy().min() >= -1e-12
    assert train[FEATURE_NAMES].to_numpy().max() <= np.pi + 1e-12
    assert joblib.load(processed / "scaler.joblib").n_features_in_ == 8


def test_classical_pipeline_integration(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    write_dataset(generate_dataset(160, 11), raw)
    prepare_data(raw / "investment_samples.csv", processed, 11)
    metrics = train_classical(processed, tmp_path / "artifacts", tmp_path / "results")
    assert set(metrics["risk"]) == {
        "concentration_risk", "volatility_risk", "liquidity_risk"
    }
    assert metrics["f1"].between(0, 1).all()
