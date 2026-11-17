#!/usr/bin/env python3
"""Run the installed vLLM benchmark with isolated warmup/measurement cache scopes.

Each invocation starts a fresh prefix-cache namespace. Requests in the measured
run can share prefixes, but cannot reuse warmup or previous-run cache entries.
vLLM still owns request scheduling, streaming, timing, and metric aggregation.
"""

import json
import secrets
import sys
from dataclasses import replace
from pathlib import Path


def main(argv=None):
    import vllm.benchmarks.serve as serving
    from vllm.entrypoints.cli.main import main as vllm_main

    argv = list(sys.argv[1:] if argv is None else argv)
    original_request = serving.ASYNC_REQUEST_FUNCS["openai"]
    original_metrics = serving.fetch_diffusion_metrics
    original_argv = sys.argv
    namespace = secrets.token_urlsafe(32)
    phase = "warmup"
    counts = {"warmup": 0, "measurement": 0}
    boundary_calls = 0

    async def scoped_request(request_func_input, session, pbar=None):
        extra = dict(request_func_input.extra_body or {})
        if "cache_salt" in extra:
            raise ValueError("benchmark_speed.py manages cache_salt; remove it from --extra-body")
        extra["cache_salt"] = f"{namespace}:{phase}"
        counts[phase] += 1
        return await original_request(
            request_func_input=replace(request_func_input, extra_body=extra),
            session=session, pbar=pbar,
        )

    async def measurement_boundary(base_url, session):
        nonlocal phase, boundary_calls
        result = await original_metrics(base_url, session)
        # In pinned vLLM 0.31.0 this hook runs after warmup, before the timer,
        # and again after the measured interval. No timed work is added here.
        phase = "measurement"
        boundary_calls += 1
        return result

    serving.ASYNC_REQUEST_FUNCS["openai"] = scoped_request
    serving.fetch_diffusion_metrics = measurement_boundary
    sys.argv = ["vllm", "bench", "serve", *argv]
    try:
        vllm_main()
        if boundary_calls != 2 or not counts["measurement"]:
            raise RuntimeError("vLLM benchmark boundaries changed; cache isolation was not verified")
        if "--save-result" in argv and "--result-filename" in argv:
            folder = Path(argv[argv.index("--result-dir") + 1]) if "--result-dir" in argv else Path(".")
            path = folder / argv[argv.index("--result-filename") + 1]
            result = json.loads(path.read_text())
            result["prefix_cache_policy"] = {
                "policy": "fresh namespace per run; warmup isolated from measurement",
                "request_counts": counts,
                "boundary_calls": boundary_calls,
            }
            path.write_text(json.dumps(result) + "\n")
    finally:
        serving.ASYNC_REQUEST_FUNCS["openai"] = original_request
        serving.fetch_diffusion_metrics = original_metrics
        sys.argv = original_argv


if __name__ == "__main__":
    main()
