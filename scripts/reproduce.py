from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
CANONICAL_POLICY_ID = "aggressive_q4_k_m_default_q8_protected_layers_3_11_14_26"

REQUIRED_RESULT_FILES = [
    "01_model_exploration.json",
    "02_weight_stats.json",
    "03_baseline_evaluation.json",
    "04_manual_quantization.json",
    "05_ptq_artifacts_summary.json",
    "06_benchmarking_perf_snapshot.json",
    "07_full_baseline_humaneval.json",
    "07_layer_int4_screening_summary.json",
    "07_selected_layer_int4_full_followup_runs.json",
    "08_mixed_precision_policy.json",
]


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"expected object at top level in {path}")
    return data


def require_key(data: dict[str, Any], key: str, source: Path) -> Any:
    if key not in data:
        raise RuntimeError(f"{source} is missing required key: {key}")
    return data[key]


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_percent(value: Any) -> str:
    return f"{fmt_float(value)}%"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def validate_required_files(results_dir: Path) -> list[Path]:
    missing = [results_dir / name for name in REQUIRED_RESULT_FILES if not (results_dir / name).is_file()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"missing required result artifact(s):\n{joined}")
    return [results_dir / name for name in REQUIRED_RESULT_FILES]


def find_canonical_validation(results_dir: Path, policy_id: str) -> tuple[Path, dict[str, Any]]:
    runs_dir = results_dir / "mixed_precision" / "runs"
    candidates: list[tuple[datetime, Path, dict[str, Any]]] = []

    for path in sorted(runs_dir.glob("*.validation.json")):
        data = load_json(path)
        combined_rows = data.get("comparison", {}).get("combined_rows", [])
        if not isinstance(combined_rows, list):
            continue

        policy_rows = [
            row
            for row in combined_rows
            if isinstance(row, dict) and row.get("model_key") == policy_id
        ]
        if not policy_rows:
            continue

        row = policy_rows[0]
        complete = (
            row.get("quality_source") == "fresh_09_humaneval"
            and row.get("perf_source") == "fresh_09_perf"
            and row.get("quality_status") == "complete"
            and row.get("perf_status") == "complete"
        )
        if not complete:
            continue

        timestamp = parse_timestamp(data.get("updated_at_utc") or data.get("timestamp_utc"))
        candidates.append((timestamp, path, data))

    if not candidates:
        raise RuntimeError(
            "could not find a complete fresh Notebook 09 validation run for "
            f"policy {policy_id!r}"
        )

    _, path, data = max(candidates, key=lambda item: item[0])
    return path, data


def artifact_status(recorded_path: Any, project_root: Path) -> tuple[str, str]:
    if not isinstance(recorded_path, str) or not recorded_path:
        return "missing path", "n/a"

    path = Path(recorded_path)
    checked_paths = [path]
    if path.is_absolute():
        checked_paths.append(project_root / "exports" / path.name)

    for candidate in checked_paths:
        if candidate.exists():
            size_gib = candidate.stat().st_size / (1024**3)
            return "present", f"{size_gib:.2f} GiB"
    return "missing", path.name


def summarize_results(results_dir: Path, check_artifacts: bool) -> None:
    validate_required_files(results_dir)

    model = load_json(results_dir / "01_model_exploration.json")
    ptq = load_json(results_dir / "05_ptq_artifacts_summary.json")
    policy = load_json(results_dir / "08_mixed_precision_policy.json")

    recommended = require_key(policy, "recommendation", results_dir / "08_mixed_precision_policy.json")
    if not isinstance(recommended, dict):
        raise RuntimeError("08_mixed_precision_policy.json recommendation must be an object")

    policy_id = recommended.get("recommended_policy_id")
    if policy_id != CANONICAL_POLICY_ID:
        raise RuntimeError(
            "unexpected recommended policy: "
            f"{policy_id!r}; expected {CANONICAL_POLICY_ID!r}"
        )

    validation_path, validation = find_canonical_validation(results_dir, CANONICAL_POLICY_ID)
    combined_rows = validation["comparison"]["combined_rows"]
    rows_by_key = {row["model_key"]: row for row in combined_rows if isinstance(row, dict)}

    expected_keys = ["fp16", "q8_0", "q4_k_m", CANONICAL_POLICY_ID]
    missing_keys = [key for key in expected_keys if key not in rows_by_key]
    if missing_keys:
        raise RuntimeError(f"canonical validation is missing row(s): {missing_keys}")

    print("Quantization Study Reproduction")
    print("===============================")
    print(f"Results directory: {results_dir}")
    print(f"Model: {model.get('model_id', validation.get('model_id', 'unknown'))}")
    print(f"Canonical policy: {policy_id}")
    print(f"Validation artifact: {validation_path}")
    print()

    print("Model Architecture")
    print("------------------")
    print(f"Total parameters: {model.get('total_params', 'n/a')}")
    print(f"Layers: {model.get('num_layers', 'n/a')}")
    print(f"Dense layers: {model.get('dense_layers', 'n/a')}")
    print(f"MoE layers: {model.get('moe_layers', 'n/a')}")
    print(f"Routed experts: {model.get('n_routed_experts', 'n/a')}")
    print(f"Experts per token: {model.get('num_experts_per_tok', 'n/a')}")
    print()

    print("Artifact Compression")
    print("--------------------")
    artifacts = require_key(ptq, "artifacts", results_dir / "05_ptq_artifacts_summary.json")
    artifact_rows = []
    artifact_labels = [
        ("f16_source", "FP16 GGUF", "100.0%"),
        ("q8_0", "Q8_0 GGUF", fmt_percent(float(ptq.get("compression_q8_vs_f16", 0.0)) * 100)),
        ("q4_k_m", "Q4_K_M GGUF", fmt_percent(float(ptq.get("compression_q4_vs_f16", 0.0)) * 100)),
    ]
    for key, label, relative in artifact_labels:
        artifact = artifacts.get(key, {}) if isinstance(artifacts, dict) else {}
        artifact_rows.append(
            [
                label,
                fmt_float(artifact.get("size_gib_binary")),
                relative,
            ]
        )
    print_table(["Artifact", "Size GiB", "Relative Size"], artifact_rows)
    print()

    print("Strict Notebook 09 Validation")
    print("-----------------------------")
    validation_rows = []
    labels = {
        "fp16": "FP16 GGUF",
        "q8_0": "Q8_0 GGUF",
        "q4_k_m": "Q4_K_M GGUF",
        CANONICAL_POLICY_ID: "Mixed 3,11,14,26",
    }
    for key in expected_keys:
        row = rows_by_key[key]
        validation_rows.append(
            [
                labels[key],
                fmt_float(row.get("size_gib")),
                fmt_percent(row.get("pass_at_1_percent")),
                str(row.get("passed", "n/a")),
                fmt_float(row.get("microbench_tps")),
                fmt_float(row.get("microbench_latency"), 3),
                str(row.get("quality_source", "n/a")),
                str(row.get("perf_source", "n/a")),
            ]
        )
    print_table(
        [
            "Model",
            "Size GiB",
            "HumanEval",
            "Passed",
            "Tok/s",
            "Latency s",
            "Quality Source",
            "Perf Source",
        ],
        validation_rows,
    )
    print()

    fp16 = rows_by_key["fp16"]
    q4 = rows_by_key["q4_k_m"]
    mixed = rows_by_key[CANONICAL_POLICY_ID]
    recovered = float(mixed["pass_at_1_percent"]) - float(q4["pass_at_1_percent"])
    speedup = float(mixed["microbench_tps"]) / float(fp16["microbench_tps"])
    print("Headline Check")
    print("--------------")
    print(f"Mixed quality recovery over Q4_K_M: +{recovered:.2f} pass@1 points")
    print(f"Mixed throughput speedup over FP16: {speedup:.2f}x")
    print(
        "Protected layers: "
        + ", ".join(str(layer) for layer in recommended.get("protected_layers", []))
    )
    print()

    if check_artifacts:
        print("Local GGUF Artifact Check")
        print("-------------------------")
        manifest = validation.get("model_manifest", {})
        artifact_check_rows = []
        for key in expected_keys:
            cfg = manifest.get(key, {}) if isinstance(manifest, dict) else {}
            status, detail = artifact_status(cfg.get("artifact_path"), PROJECT_ROOT)
            artifact_check_rows.append([labels[key], status, detail])
        print_table(["Model", "Status", "Detail"], artifact_check_rows)
        print()

    print("Reproduction status: passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the repository's published headline results from checked-in "
            "JSON artifacts."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing result JSON artifacts.",
    )
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Also check whether local GGUF files referenced by validation metadata exist.",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help=(
            "Explain the expensive full-validation path. This script intentionally "
            "does not run model benchmarks by default."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.full_validation:
        print(
            "Full validation is intentionally not hidden behind the fast reproduction command.\n"
            "To rerun the expensive benchmark, open notebooks/09_mixed_precision_validation.ipynb,\n"
            "set RUN_FULL_HUMANEVAL, RUN_FRESH_PERF_BENCHMARKS, and REEVALUATE_EXISTING to True,\n"
            "then execute the notebook with the local GGUF artifacts present in exports/."
        )
        return 0

    try:
        summarize_results(args.results_dir.resolve(), args.check_artifacts)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
