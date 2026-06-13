import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from .constants import FEATURE_NAMES, RISK_LABELS, SEED
from .core import model_scores, sigmoid
from .evaluation import metric_row


RISK_FEATURES = {
    "concentration_risk": [
        "estimated_amount_ratio", "holding_ratio_after_trade",
        "current_position_ratio", "risk_preference_code",
    ],
    "volatility_risk": ["volatility_20d", "abs_return_5d"],
    "liquidity_risk": ["turnover_rate", "trade_volume_to_avg_volume"],
}


def build_quantum_kernel(
    feature_dimension: int = len(FEATURE_NAMES), reps: int = 2,
    entanglement: str = "linear",
):
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    feature_map = zz_feature_map(
        feature_dimension=feature_dimension, reps=reps, entanglement=entanglement
    )
    return FidelityQuantumKernel(feature_map=feature_map)


def _signature(values: np.ndarray, config: Dict) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values).tobytes()
        + json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def kernel_statistics(matrix: np.ndarray) -> Dict[str, float]:
    mask = ~np.eye(len(matrix), dtype=bool)
    off = matrix[mask]
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    weights = np.clip(eigenvalues, 0, None)
    weights = weights / max(weights.sum(), 1e-15)
    effective_rank = float(np.exp(-np.sum(weights * np.log(np.clip(weights, 1e-15, None)))))
    return {
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "offdiag_max": float(off.max()),
        "effective_rank": effective_rank,
    }


def compute_full_kernel(
    values: np.ndarray, cache_dir: Path, config: Dict,
) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "kernel.npy"
    metadata_path = cache_dir / "cache_metadata.json"
    signature = _signature(values, config)
    if matrix_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            matrix = np.load(matrix_path)
            return matrix, float(metadata["elapsed_seconds"]), True, metadata["statistics"]
    kernel = build_quantum_kernel(
        feature_dimension=values.shape[1], reps=int(config["reps"]),
        entanglement=config.get("entanglement", "linear"),
    )
    started = time.perf_counter()
    matrix = kernel.evaluate(x_vec=values * float(config.get("scale", 1.0)))
    elapsed = time.perf_counter() - started
    statistics = kernel_statistics(matrix)
    np.save(matrix_path, matrix)
    metadata_path.write_text(
        json.dumps({
            "signature": signature, "config": config, "shape": list(matrix.shape),
            "elapsed_seconds": elapsed, "statistics": statistics,
        }, indent=2), encoding="utf-8",
    )
    return matrix, elapsed, False, statistics


def compute_kernel_matrices(x_train, x_test, cache_dir: Path, reps: int = 2):
    """Backward-compatible helper used by existing callers and tests."""
    values = np.vstack([x_train, x_test])
    matrix, elapsed, _, _ = compute_full_kernel(
        values, cache_dir, {"reps": reps, "entanglement": "linear", "scale": 1.0}
    )
    boundary = len(x_train)
    return matrix[:boundary, :boundary], matrix[boundary:, :boundary], elapsed


def _subset(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(frame):
        return frame.copy().reset_index(drop=True)
    key = frame[RISK_LABELS].astype(str).agg("".join, axis=1)
    selected, _ = train_test_split(frame, train_size=size, random_state=seed, stratify=key)
    return selected.reset_index(drop=True)


def _load_splits(processed_dir: Path):
    return (
        pd.read_csv(processed_dir / "train.csv"),
        pd.read_csv(processed_dir / "val.csv"),
        pd.read_csv(processed_dir / "test.csv"),
    )


def assert_disjoint_sample_ids(fit: pd.DataFrame, test: pd.DataFrame) -> None:
    overlap = set(fit["sample_id"]) & set(test["sample_id"])
    if overlap:
        raise ValueError(f"training and test samples overlap: {sorted(overlap)[:3]}")


def _save_experiment_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_quantum(
    processed_dir: Path, artifacts_dir: Path, results_dir: Path,
    subset_size: int = 180, reps: int = 2, full_data: bool = False,
    experiment: str = "subset_v1",
) -> pd.DataFrame:
    train, val, test = _load_splits(processed_dir)
    fit = pd.concat([train, val], ignore_index=True)
    assert_disjoint_sample_ids(fit, test)
    if full_data:
        q_train, q_test = fit, test.reset_index(drop=True)
        result_name = "metrics_quantum_full_baseline.csv"
    else:
        train_size = max(24, int(subset_size * 0.75))
        test_size = max(8, subset_size - train_size)
        q_train = _subset(fit, train_size, SEED)
        q_test = _subset(test, min(test_size, len(test)), SEED)
        result_name = "metrics_quantum.csv"
    all_samples = pd.concat([q_train, q_test], ignore_index=True)
    experiment_dir = artifacts_dir / "quantum_experiments" / experiment
    cache_dir = experiment_dir / "shared_kernel"
    config = {
        "features": FEATURE_NAMES, "reps": reps, "entanglement": "linear",
        "scale": 1.0, "architecture": "shared",
    }
    matrix, kernel_seconds, cache_hit, statistics = compute_full_kernel(
        all_samples[FEATURE_NAMES].to_numpy(), cache_dir, config
    )
    boundary = len(q_train)
    k_train = matrix[:boundary, :boundary]
    k_test = matrix[boundary:, :boundary]
    model_dir = experiment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for risk in RISK_LABELS:
        started = time.perf_counter()
        model = SVC(kernel="precomputed", C=1.0, random_state=SEED)
        model.fit(k_train, q_train[risk])
        scores = model_scores(model, k_test)
        rows.append(metric_row(
            "quantum_kernel_svm_full" if full_data else "quantum_kernel_svm",
            risk, q_test[risk], scores, time.perf_counter() - started,
        ))
        joblib.dump(model, model_dir / f"{risk}.joblib")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / result_name, index=False)
    if full_data:
        classical = pd.read_csv(results_dir / "metrics_classical.csv") if (results_dir / "metrics_classical.csv").exists() else pd.DataFrame()
        pd.concat([metrics, classical], ignore_index=True).to_csv(
            results_dir / "comparison_full.csv", index=False
        )
    np.save(experiment_dir / "train_features.npy", q_train[FEATURE_NAMES].to_numpy())
    _save_experiment_manifest(experiment_dir / "manifest.json", {
        "experiment": experiment, "full_data": full_data,
        "train_sample_ids": q_train.sample_id.tolist(),
        "test_sample_ids": q_test.sample_id.tolist(),
        "train_size": len(q_train), "test_size": len(q_test),
        "kernel_seconds": kernel_seconds, "cache_hit": cache_hit,
        "kernel_statistics": statistics, "config": config,
    })
    return metrics


def _best_threshold(y_true: np.ndarray, decisions: np.ndarray) -> Tuple[float, float]:
    candidates = np.unique(np.quantile(decisions, np.linspace(0.02, 0.98, 97)))
    best = max(
        ((f1_score(y_true, decisions >= threshold, zero_division=0), float(threshold))
         for threshold in candidates),
        key=lambda item: (item[0], -abs(item[1])),
    )
    return best[1], best[0]


def _cross_validated_search(kernel: np.ndarray, labels: np.ndarray) -> Dict:
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    best = None
    for c_value in [0.1, 1.0, 10.0, 100.0]:
        for class_weight in [None, "balanced"]:
            decisions = np.zeros(len(labels), dtype=float)
            for train_index, val_index in folds.split(kernel, labels):
                model = SVC(kernel="precomputed", C=c_value, class_weight=class_weight, random_state=SEED)
                model.fit(kernel[np.ix_(train_index, train_index)], labels[train_index])
                decisions[val_index] = model.decision_function(kernel[np.ix_(val_index, train_index)])
            threshold, score = _best_threshold(labels, decisions)
            auc = roc_auc_score(labels, decisions)
            candidate = {
                "C": c_value, "class_weight": class_weight, "threshold": threshold,
                "cv_f1": score, "cv_roc_auc": auc,
            }
            if best is None or (score, auc) > (best["cv_f1"], best["cv_roc_auc"]):
                best = candidate
    return best


def tune_quantum(
    processed_dir: Path, artifacts_dir: Path, results_dir: Path,
    budget_minutes: float = 60.0, subset_size: int = 180,
) -> pd.DataFrame:
    train, val, test = _load_splits(processed_dir)
    fit = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    assert_disjoint_sample_ids(fit, test)
    tuning = _subset(fit, min(subset_size, len(fit)), SEED)
    search_root = artifacts_dir / "quantum_experiments" / "tuning"
    search_rows: List[Dict] = []
    started_all = time.perf_counter()
    best_configs = {}
    for risk in RISK_LABELS:
        features = RISK_FEATURES[risk]
        risk_best = None
        for scale in [0.125, 0.25, 0.5, 1.0]:
            for reps in [1, 2]:
                if (time.perf_counter() - started_all) / 60 > budget_minutes:
                    break
                config = {
                    "risk": risk, "features": features, "scale": scale, "reps": reps,
                    "entanglement": "linear", "architecture": "risk_specific",
                }
                cache = search_root / risk / f"scale_{scale}_reps_{reps}"
                matrix, elapsed, cache_hit, statistics = compute_full_kernel(
                    tuning[features].to_numpy(), cache, config
                )
                hyper = _cross_validated_search(matrix, tuning[risk].to_numpy())
                row = {**config, **hyper, **statistics, "kernel_seconds": elapsed, "cache_hit": cache_hit}
                row["features"] = json.dumps(features)
                search_rows.append(row)
                if risk_best is None or (hyper["cv_f1"], hyper["cv_roc_auc"]) > (
                    risk_best["cv_f1"], risk_best["cv_roc_auc"]
                ):
                    risk_best = {**config, **hyper, "kernel_statistics": statistics}
            if (time.perf_counter() - started_all) / 60 > budget_minutes:
                break
        if risk_best is None:
            raise RuntimeError(f"tuning budget exhausted before evaluating {risk}")
        best_configs[risk] = risk_best
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).to_csv(results_dir / "quantum_search_results.csv", index=False)

    final_root = artifacts_dir / "models" / "quantum_optimized"
    final_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for risk, config in best_configs.items():
        features = config["features"]
        all_samples = pd.concat([fit, test], ignore_index=True)
        cache = artifacts_dir / "quantum_experiments" / "optimized_full" / risk
        matrix, elapsed, cache_hit, statistics = compute_full_kernel(
            all_samples[features].to_numpy(), cache, config
        )
        boundary = len(fit)
        train_kernel, test_kernel = matrix[:boundary, :boundary], matrix[boundary:, :boundary]
        model = SVC(
            kernel="precomputed", C=float(config["C"]),
            class_weight=config["class_weight"], random_state=SEED,
        )
        model.fit(train_kernel, fit[risk])
        decisions = model.decision_function(test_kernel)
        scores = sigmoid(decisions - float(config["threshold"]))
        metric_rows.append(metric_row(
            "quantum_kernel_svm_optimized", risk, test[risk], scores,
            elapsed, threshold=0.5,
        ))
        risk_dir = final_root / risk
        risk_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, risk_dir / "model.joblib")
        np.save(risk_dir / "train_features.npy", fit[features].to_numpy())
        final_config = {
            **config, "feature_dimension": len(features), "num_qubits": len(features),
            "train_size": len(fit), "test_size": len(test), "kernel_seconds": elapsed,
            "cache_hit": cache_hit, "full_kernel_statistics": statistics,
        }
        (risk_dir / "config.json").write_text(json.dumps(final_config, indent=2), encoding="utf-8")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "metrics_quantum_optimized.csv", index=False)
    comparison_parts = [metrics]
    for result_file in ["metrics_quantum_full_baseline.csv", "metrics_classical.csv"]:
        path = results_dir / result_file
        if path.exists():
            comparison_parts.append(pd.read_csv(path))
    pd.concat(comparison_parts, ignore_index=True).to_csv(
        results_dir / "comparison_full.csv", index=False
    )
    _save_experiment_manifest(final_root / "manifest.json", {
        "architecture": "risk_specific", "train_size": len(fit), "test_size": len(test),
        "tuning_sample_ids": tuning.sample_id.tolist(), "best_configs": best_configs,
        "budget_minutes": budget_minutes,
    })
    return metrics
