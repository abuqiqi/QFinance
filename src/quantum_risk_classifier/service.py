import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .constants import FEATURE_NAMES, MODEL_VERSION, RISK_LABELS
from .core import confidence_from_scores, feature_vector, model_scores, overall_level, overall_score
from .quantum import build_quantum_kernel
from .schemas import InvestmentPlanInput, PredictionOutput


def predict(plan: InvestmentPlanInput, processed_dir: Path, artifacts_dir: Path) -> PredictionOutput:
    scaler = joblib.load(processed_dir / "scaler.joblib")
    vector, _ = feature_vector(plan)
    scaled = scaler.transform(pd.DataFrame([vector], columns=FEATURE_NAMES))
    optimized_dir = artifacts_dir / "models" / "quantum_optimized"
    kernel_manifest = artifacts_dir / "kernels" / "manifest.json"
    quantum_models = artifacts_dir / "models" / "quantum"
    scores = {}
    risk_configs = None
    if (optimized_dir / "manifest.json").exists():
        scaled_frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
        risk_configs = {}
        for risk in RISK_LABELS:
            risk_dir = optimized_dir / risk
            config = json.loads((risk_dir / "config.json").read_text(encoding="utf-8"))
            train = np.load(risk_dir / "train_features.npy")
            features = config["features"]
            values = scaled_frame[features].to_numpy() * float(config["scale"])
            kernel = build_quantum_kernel(
                feature_dimension=len(features), reps=int(config["reps"]),
                entanglement=config["entanglement"],
            )
            model_input = kernel.evaluate(
                x_vec=values, y_vec=train * float(config["scale"])
            )
            model = joblib.load(risk_dir / "model.joblib")
            decision = float(model.decision_function(model_input)[0])
            scores[risk] = float(1.0 / (1.0 + np.exp(-(decision - float(config["threshold"])))))
            risk_configs[risk] = {
                "features": features, "feature_dimension": len(features),
                "reps": config["reps"], "scale": config["scale"],
                "C": config["C"], "class_weight": config["class_weight"],
                "decision_threshold": config["threshold"],
            }
        classifier_type = "quantum_kernel_svm_optimized"
        feature_dimension = max(len(config["features"]) for config in risk_configs.values())
        num_qubits = feature_dimension
    elif kernel_manifest.exists() and quantum_models.exists():
        manifest = json.loads(kernel_manifest.read_text(encoding="utf-8"))
        train = np.load(artifacts_dir / "kernels" / "train_features.npy") if (artifacts_dir / "kernels" / "train_features.npy").exists() else None
        if train is None:
            raise FileNotFoundError("quantum training features are missing; rerun train-quantum")
        kernel = build_quantum_kernel(reps=int(manifest["reps"]))
        model_input = kernel.evaluate(x_vec=scaled, y_vec=train)
        model_dir = quantum_models
        classifier_type = "quantum_kernel_svm"
        feature_dimension = len(FEATURE_NAMES)
        num_qubits = len(FEATURE_NAMES)
    else:
        model_input = scaled
        model_dir = artifacts_dir / "models" / "classical"
        classifier_type = "classical_rbf_svm"
        feature_dimension = len(FEATURE_NAMES)
        num_qubits = 0
    if not scores:
        for risk in RISK_LABELS:
            model = joblib.load(model_dir / f"{risk}.joblib")
            scores[risk] = float(model_scores(model, model_input)[0])
    labels = {name: int(score >= 0.5) for name, score in scores.items()}
    volume_share = plan.trade_plan.volume_lot * 100
    estimated_amount = volume_share * plan.trade_plan.estimated_price
    return PredictionOutput.model_validate(
        {
            "sample_id": plan.sample_id,
            "model_version": MODEL_VERSION,
            "input_summary": {
                "stock_code": plan.trade_plan.stock_code,
                "action": plan.trade_plan.action,
                "volume_lot": plan.trade_plan.volume_lot,
                "volume_share": volume_share,
                "estimated_amount": round(estimated_amount, 2),
            },
            "risk_scores": scores,
            "risk_labels": labels,
            "overall_risk": {
                "overall_score": overall_score(scores),
                "overall_level": overall_level(labels),
            },
            "confidence": confidence_from_scores(scores),
            "model_outputs": {
                "classifier_type": classifier_type,
                "feature_map": "ZZFeatureMap" if classifier_type.startswith("quantum") else "RBF",
                "num_qubits": num_qubits,
                "feature_dimension": feature_dimension,
                "risk_configs": risk_configs,
            },
        }
    )
