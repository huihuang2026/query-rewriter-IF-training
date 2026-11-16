#!/usr/bin/env python3
"""Print only mean/P95 TTFT and TPOT from a completed vLLM benchmark result."""

import argparse
import json
import math
from pathlib import Path

METRICS = (
    ("TTFT Mean (ms)", "mean_ttft_ms"),
    ("TTFT P95 (ms)", "p95_ttft_ms"),
    ("Per Token Mean (ms/token)", "mean_tpot_ms"),
    ("Per Token P95 (ms/token)", "p95_tpot_ms"),
)


def report(path, label):
    result = json.loads(path.read_text(encoding="utf-8"))
    completed, failed = result.get("completed"), result.get("failed")
    requested = result.get("num_prompts")
    if failed != 0 or not completed or completed != requested:
        raise ValueError(
            f"Incomplete benchmark: {completed}/{requested} completed, {failed} failed; "
            "no summary reported. Inspect the benchmark log and raw JSON."
        )
    values = []
    for _, key in METRICS:
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing/invalid metric {key}; inspect {path}")
        values.append(f"{value:.3f}")
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
