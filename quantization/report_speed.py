#!/usr/bin/env python3
"""Print mean/P95 TTFT, TPOT, and E2E from a completed vLLM benchmark result."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

METRICS = (
    ("TTFT Mean (ms)", "mean_ttft_ms"),
    ("TTFT P95 (ms)", "p95_ttft_ms"),
    ("Per Token Mean (ms/token)", "mean_tpot_ms"),
    ("Per Token P95 (ms/token)", "p95_tpot_ms"),
    ("E2E Mean (ms)", "mean_e2el_ms"),
    ("E2E P95 (ms)", "p95_e2el_ms"),
)


def add_e2e_metrics(result):
    """Backfill old runs from per-request seconds, using NumPy's linear P95."""
    if all(key in result for key in ("mean_e2el_ms", "p95_e2el_ms")):
        return
    latencies = result.get("latencies")
    if (not isinstance(latencies, list) or not latencies
            or len(latencies) != result["completed"]
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or value < 0 for value in latencies)):
        raise ValueError("Missing/invalid per-request latencies for E2E metrics")
    ordered = sorted(latencies)
    rank = (len(ordered) - 1) * 0.95
    lower, upper = math.floor(rank), math.ceil(rank)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
    result.setdefault("mean_e2el_ms", fmean(latencies) * 1000)
    result.setdefault("p95_e2el_ms", p95 * 1000)


def report(path, label):
    result = json.loads(path.read_text(encoding="utf-8"))
    completed, failed = result.get("completed"), result.get("failed")
    requested = result.get("num_prompts")
    if failed != 0 or not completed or completed != requested:
        raise ValueError(
            f"Incomplete benchmark: {completed}/{requested} completed, {failed} failed; "
            "no summary reported. Inspect the benchmark log and raw JSON."
        )
    add_e2e_metrics(result)
    values = []
    for _, key in METRICS:
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing/invalid metric {key}; inspect {path}")
        values.append(f"{value:.2f}")
    print(f"{label} — {result['request_rate']} requests/sec target")
    print()
    print("| " + " | ".join(header for header, _ in METRICS) + " |")
    print("| " + " | ".join("---" for _ in METRICS) + " |")
    print("| " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    try:
        report(args.input, args.label)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
