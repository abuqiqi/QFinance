import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from .constants import FEATURE_NAMES, RISK_LABELS, SEED
from .evaluation import metric_row, save_confusion


def train_classical(processed_dir: Path, artifacts_dir: Path, results_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    fit = pd.concat([train, val], ignore_index=True)
    model_dir = artifacts_dir / "models" / "classical"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selection = {}
    for risk in RISK_LABELS:
        best_c = 1.0
        best_f1 = -1.0
        for candidate_c in [0.5, 1.0, 2.0]:
            candidate = SVC(kernel="rbf", C=candidate_c, gamma="scale", random_state=SEED)
            candidate.fit(train[FEATURE_NAMES], train[risk])
            candidate_predictions = candidate.decision_function(val[FEATURE_NAMES]) >= 0.0
            candidate_f1 = f1_score(val[risk], candidate_predictions, zero_division=0)
            if candidate_f1 > best_f1:
                best_c, best_f1 = candidate_c, candidate_f1
        selection[risk] = {"C": best_c, "validation_f1": best_f1}
        started = time.perf_counter()
        base = SVC(kernel="rbf", C=best_c, gamma="scale", random_state=SEED)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(fit[FEATURE_NAMES], fit[risk])
        elapsed = time.perf_counter() - started
        scores = model.predict_proba(test[FEATURE_NAMES])[:, 1]
        rows.append(metric_row("classical_rbf_svm", risk, test[risk], scores, elapsed))
        joblib.dump(model, model_dir / f"{risk}.joblib")
        save_confusion(test[risk], scores, risk, results_dir / f"confusion_classical_{risk}.png")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "metrics_classical.csv", index=False)
    joblib.dump(selection, model_dir / "selection.joblib")
    return metrics
