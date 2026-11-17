#!/usr/bin/env python3
"""Start a serving script, wait for its model, run a client, and clean up."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def interrupt(signum, _frame):
    raise Interrupted(signum)


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_process(process):
    """Stop only the process group created for this server or client."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 15
    while group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()  # Reap the group leader while its children finish.
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def wait_ready(server, base_url, model, timeout):
    # Local readiness checks must not pass through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited before becoming ready (status {server.returncode})")
        try:
            with opener.open(base_url + "/health", timeout=2) as response:
                healthy = response.status == 200
            with opener.open(base_url + "/v1/models", timeout=2) as response:
                models = json.load(response)
            if healthy:
                if model not in [item.get("id") for item in models.get("data", [])]:
                    raise RuntimeError(f"Server at {base_url} does not expose expected model {model}")
                if server.poll() is not None:
                    raise RuntimeError("Server exited during readiness checks")
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Server did not become ready within {timeout:g} seconds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/quantization"))
    parser.add_argument("client", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.client[1:] if args.client[:1] == ["--"] else args.client
    if not command or args.startup_timeout <= 0:
        parser.error("Provide a client command after -- and a positive startup timeout")
    if not args.server_script.is_file():
        parser.error(f"Serving script does not exist: {args.server_script}")

    # Refuse an occupied port instead of accidentally benchmarking another server.
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
    except OSError as error:
        print(f"Cannot start server on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"benchmark_{args.server_script.stem}_{time.time_ns()}.log"
    server = client = None
    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log_path.open("w") as log:
            print(f"Starting {args.server_script.name}. Server log: {log_path}", flush=True)
            server = subprocess.Popen(["bash", str(args.server_script)], stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            wait_ready(server, f"http://{args.host}:{args.port}", args.model, args.startup_timeout)
            print("Model ready. Starting benchmark.", flush=True)
            env = os.environ.copy()
            env["BENCHMARK_SERVER_LOG"] = str(log_path.resolve())
            client = subprocess.Popen(command, env=env, start_new_session=True)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during benchmark (status {server.returncode})")
                time.sleep(0.2)
            return client.returncode if client.returncode >= 0 else 128 - client.returncode
    except Interrupted as error:
        print("Benchmark interrupted; stopping its server and client.", file=sys.stderr, flush=True)
        return 128 + error.signum
    except (OSError, RuntimeError) as error:
        print(f"{error}. Server log: {log_path}", file=sys.stderr)
        if log_path.exists():
            with log_path.open(errors="replace") as log:
                print("".join(deque(log, maxlen=20)), file=sys.stderr, end="")
        return 1
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if server is not None:
            print("Stopping benchmark server.", flush=True)
        stop_process(client)
        stop_process(server)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
