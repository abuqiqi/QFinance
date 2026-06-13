import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .constants import CONCENTRATION_THRESHOLDS, RISK_PREFERENCE_CODES, SEED
from .core import overall_level


def _sample_record(index: int, rng: np.random.Generator) -> Dict:
    preference = rng.choice(["conservative", "moderate", "aggressive"], p=[0.3, 0.5, 0.2])
    threshold = CONCENTRATION_THRESHOLDS[preference]
    concentration_target = bool(rng.random() < 0.36)
    volatility_target = bool(rng.random() < 0.35)
    liquidity_target = bool(rng.random() < 0.25)

    total_asset = float(rng.integers(200_000, 2_000_001))
    current_ratio = float(rng.uniform(0.0, min(0.18, threshold * 0.55)))
    if concentration_target:
        holding_ratio = float(rng.uniform(threshold, min(0.85, threshold + 0.25)))
    else:
        holding_ratio = float(rng.uniform(current_ratio + 0.01, max(current_ratio + 0.011, threshold - 0.005)))
    amount_ratio = holding_ratio - current_ratio
    current_value = total_asset * current_ratio
    estimated_amount = total_asset * amount_ratio
    price = round(float(rng.uniform(4.0, 80.0)), 4)
    volume_lot = max(1, int(round(estimated_amount / price / 100)))
    volume_share = volume_lot * 100
    estimated_amount = volume_share * price
    amount_ratio = estimated_amount / total_asset
    holding_ratio = current_ratio + amount_ratio

    ratio = float(rng.uniform(0.10, 0.30) if liquidity_target else rng.uniform(0.005, 0.095))
    avg_volume = volume_share / ratio
    if volatility_target:
        if rng.random() < 0.65:
            volatility = float(rng.uniform(0.35, 0.65))
            return_5d = float(rng.uniform(-0.22, 0.22))
        else:
            volatility = float(rng.uniform(0.12, 0.34))
            return_5d = float(rng.choice([-1, 1]) * rng.uniform(0.15, 0.28))
    else:
        volatility = float(rng.uniform(0.10, 0.345))
        return_5d = float(rng.uniform(-0.145, 0.145))

    concentration = int(holding_ratio >= threshold)
    volatility_label = int(volatility >= 0.35 or abs(return_5d) >= 0.15)
    liquidity = int(ratio >= 0.10)
    labels = {
        "concentration_risk": concentration,
        "volatility_risk": volatility_label,
        "liquidity_risk": liquidity,
    }
    timestamp = datetime(2026, 1, 5, 9, 30) + timedelta(minutes=int(rng.integers(0, 180)))
    stock_code = f"{int(rng.integers(1, 999999)):06d}"
    market = "SSE" if stock_code.startswith("6") else "SZSE"
    return {
        "sample_id": f"S{index:06d}",
        "user_profile": {
            "user_id": f"U{int(rng.integers(1, 201)):04d}",
            "total_asset": round(total_asset, 2),
            "cash_available": round(min(total_asset, estimated_amount * rng.uniform(1.0, 1.5)), 2),
            "risk_preference": preference,
            "current_stock_position_value": round(current_value, 2),
        },
        "trade_plan": {
            "action": "buy",
            "market": market,
            "stock_code": stock_code,
            "trade_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "volume_lot": volume_lot,
            "volume_share": volume_share,
            "estimated_price": price,
            "estimated_amount": round(estimated_amount, 2),
        },
        "stock_features": {
            "volatility_20d": round(volatility, 6),
            "return_5d": round(return_5d, 6),
            "abs_return_5d": round(abs(return_5d), 6),
            "avg_volume_20d": round(avg_volume, 2),
            "turnover_rate": round(float(rng.uniform(0.002, 0.12)), 6),
        },
        "derived_features": {
            "estimated_amount_ratio": round(amount_ratio, 8),
            "current_position_ratio": round(current_ratio, 8),
            "holding_ratio_after_trade": round(holding_ratio, 8),
            "trade_volume_to_avg_volume": round(ratio, 8),
        },
        "labels": {**labels, "overall_risk_level": overall_level(labels)},
    }


def flatten(record: Dict) -> Dict:
    row = {"sample_id": record["sample_id"]}
    for section in ["user_profile", "trade_plan", "stock_features", "derived_features", "labels"]:
        row.update(record[section])
    row["risk_preference_code"] = RISK_PREFERENCE_CODES[row["risk_preference"]]
    row["trade_hour"] = int(row["trade_time"][11:13])
    return row


def generate_dataset(count: int = 600, seed: int = SEED) -> List[Dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return [_sample_record(i + 1, rng) for i in range(count)]


def write_dataset(records: List[Dict], raw_dir: Path) -> Dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = raw_dir / "investment_samples.jsonl"
    csv_path = raw_dir / "investment_samples.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    frame = pd.DataFrame([flatten(record) for record in records])
    frame.to_csv(csv_path, index=False)
    profile = {
        "sample_count": len(frame),
        "missing_values": int(frame.isna().sum().sum()),
        "label_positive_rates": {
            label: float(frame[label].mean())
            for label in ["concentration_risk", "volatility_risk", "liquidity_risk"]
        },
        "numeric_ranges": {
            column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
            for column in [
                "estimated_amount_ratio", "holding_ratio_after_trade", "volatility_20d",
                "abs_return_5d", "trade_volume_to_avg_volume",
            ]
        },
    }
    (raw_dir / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile
