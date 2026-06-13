import json
import math
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .constants import FEATURE_NAMES, RISK_LABELS, SEED


def _stratification_key(frame: pd.DataFrame) -> pd.Series:
    return frame[RISK_LABELS].astype(str).agg("".join, axis=1)


def prepare_data(raw_csv: Path, processed_dir: Path, seed: int = SEED) -> Dict:
    frame = pd.read_csv(raw_csv)
    key = _stratification_key(frame)
    train_val, test = train_test_split(frame, test_size=0.20, random_state=seed, stratify=key)
    train_key = _stratification_key(train_val)
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=seed, stratify=train_key
    )
    scaler = MinMaxScaler(feature_range=(0.0, math.pi), clip=True)
    scaler.fit(train[FEATURE_NAMES])
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame[["sample_id", *FEATURE_NAMES, *RISK_LABELS]].to_csv(
        processed_dir / "features.csv", index=False
    )
    outputs = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        output = split.copy()
        output[FEATURE_NAMES] = scaler.transform(split[FEATURE_NAMES])
        output.to_csv(processed_dir / f"{name}.csv", index=False)
        outputs[name] = {
            "size": len(output),
            "sample_ids": sorted(output["sample_id"].tolist()),
            "positive_rates": {label: float(output[label].mean()) for label in RISK_LABELS},
        }
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    metadata = {"features": FEATURE_NAMES, "splits": outputs, "seed": seed}
    (processed_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
