import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .classical import train_classical
from .constants import ARTIFACTS, DATA_PROCESSED, DATA_RAW, EXAMPLES, RESULTS, ROOT, SEED
from .data_generation import generate_dataset, write_dataset
from .preprocessing import prepare_data
from .quantum import train_quantum, tune_quantum
from .schemas import InvestmentPlanInput, PredictionOutput
from .service import predict


def _read_plan(path: Path) -> InvestmentPlanInput:
    return InvestmentPlanInput.model_validate_json(path.read_text(encoding="utf-8"))


def command_validate(args):
    plan = _read_plan(args.input)
    print(json.dumps({"valid": True, "sample_id": plan.sample_id}))


def command_generate(args):
    profile = write_dataset(generate_dataset(args.count, args.seed), DATA_RAW)
    print(json.dumps(profile, indent=2))


def command_prepare(args):
    metadata = prepare_data(DATA_RAW / "investment_samples.csv", DATA_PROCESSED, args.seed)
    print(json.dumps({name: value["size"] for name, value in metadata["splits"].items()}))


def command_classical(args):
    print(train_classical(DATA_PROCESSED, ARTIFACTS, RESULTS).to_string(index=False))


def command_quantum(args):
    print(train_quantum(
        DATA_PROCESSED, ARTIFACTS, RESULTS, args.subset_size, args.reps,
        args.full_data, args.experiment,
    ).to_string(index=False))


def command_tune_quantum(args):
    print(tune_quantum(
        DATA_PROCESSED, ARTIFACTS, RESULTS, args.budget_minutes, args.subset_size,
    ).to_string(index=False))


def command_evaluate(args):
    status = {}
    for name in [
        "metrics_classical.csv", "metrics_quantum.csv", "comparison.csv",
        "metrics_quantum_full_baseline.csv", "quantum_search_results.csv",
        "metrics_quantum_optimized.csv", "comparison_full.csv",
    ]:
        path = RESULTS / name
        status[name] = "ready" if path.exists() else "missing"
        print(f"{name}: {status[name]}")
    manifest = {
        "model_version": "qksvm_demo_v1",
        "seed": SEED,
        "artifacts": status,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "qiskit", "qiskit-machine-learning"]
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def command_predict(args):
    output = predict(_read_plan(args.input), DATA_PROCESSED, ARTIFACTS)
    rendered = output.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "predictions.jsonl").write_text(
        output.model_dump_json() + "\n", encoding="utf-8"
    )
    print(rendered)


def command_schemas(args):
    target = ROOT / "schemas"
    target.mkdir(parents=True, exist_ok=True)
    (target / "input_schema.json").write_text(json.dumps(InvestmentPlanInput.model_json_schema(), indent=2), encoding="utf-8")
    (target / "output_schema.json").write_text(json.dumps(PredictionOutput.model_json_schema(), indent=2), encoding="utf-8")
    print(str(target))


def command_demo(args):
    command_generate(argparse.Namespace(count=args.count, seed=args.seed))
    command_prepare(argparse.Namespace(seed=args.seed))
    command_classical(args)
    if not args.skip_quantum:
        command_quantum(argparse.Namespace(
            subset_size=args.subset_size, reps=2, full_data=False,
            experiment="demo_subset",
        ))
    command_predict(argparse.Namespace(input=EXAMPLES / "input_valid.json", output=RESULTS / "prediction.json"))
    command_evaluate(args)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="quantum-risk")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-input")
    validate.add_argument("--input", type=Path, required=True); validate.set_defaults(func=command_validate)
    generate = commands.add_parser("generate-data")
    generate.add_argument("--count", type=int, default=600); generate.add_argument("--seed", type=int, default=SEED); generate.set_defaults(func=command_generate)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--seed", type=int, default=SEED); prepare.set_defaults(func=command_prepare)
    classical = commands.add_parser("train-classical"); classical.set_defaults(func=command_classical)
    quantum = commands.add_parser("train-quantum")
    quantum.add_argument("--subset-size", type=int, default=180)
    quantum.add_argument("--reps", type=int, default=2)
    quantum.add_argument("--full-data", action="store_true")
    quantum.add_argument("--experiment", default="subset_v1")
    quantum.set_defaults(func=command_quantum)
    tune = commands.add_parser("tune-quantum")
    tune.add_argument("--budget-minutes", type=float, default=60.0)
    tune.add_argument("--subset-size", type=int, default=180)
    tune.set_defaults(func=command_tune_quantum)
    evaluate = commands.add_parser("evaluate"); evaluate.set_defaults(func=command_evaluate)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--input", type=Path, required=True); prediction.add_argument("--output", type=Path); prediction.set_defaults(func=command_predict)
    schemas = commands.add_parser("export-schemas"); schemas.set_defaults(func=command_schemas)
    demo = commands.add_parser("run-demo")
    demo.add_argument("--count", type=int, default=600); demo.add_argument("--seed", type=int, default=SEED)
    demo.add_argument("--subset-size", type=int, default=180); demo.add_argument("--skip-quantum", action="store_true"); demo.set_defaults(func=command_demo)
    return root


def main() -> None:
    try:
        args = parser().parse_args()
        args.func(args)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
