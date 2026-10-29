#!/usr/bin/env bash
# Download the supplied Keywords AI export URLs; completed files are reused.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$script_dir" "$@" <<'PY'
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit

root = Path(sys.argv[1])
parser = argparse.ArgumentParser(description="Download and validate Keywords AI JSONL exports. No AWS login needed.")
parser.add_argument("--urls", type=Path, default=root / "export-job-Retell_AI_query_rewriter_13_call_ids_2026-10-13_to_10-15-urls.txt")
parser.add_argument("--output", type=Path, default=root / "keyword_exports")
args = parser.parse_args(sys.argv[2:])
if not shutil.which("curl"):
    parser.error("curl is required")
urls = list(dict.fromkeys(line.strip() for line in args.urls.read_text().splitlines() if line.strip()))
downloads = []
names = set()
for url in urls:
    parsed = urlsplit(url)
    name = unquote(parsed.path.rsplit("/", 1)[-1])
    if parsed.scheme != "https" or not parsed.netloc or not name.endswith(".jsonl") or "/" in name or "\\" in name:
        parser.error("Expected one HTTPS JSONL download URL per line")
    if name in names:
        parser.error(f"Different URLs have the same filename: {name}")
    names.add(name)
    downloads.append((name, url))
if not downloads:
    parser.error("No download URLs found")


def validate(path):
    count = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except ValueError:
                    raise ValueError(f"Invalid JSON in {path.name}, line {number}") from None
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a JSON object in {path.name}, line {number}")
                count += 1
    return count


args.output.mkdir(parents=True, exist_ok=True)
with (args.output / ".download.lock").open("a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another Keywords export download is already running")
    failures = total_rows = 0
    for number, (name, url) in enumerate(downloads, 1):
        final = args.output / name
        partial = args.output / (name + ".partial")
        prefix = f"[{number}/{len(downloads)}] {name}"
        try:
            if final.exists():
                rows = validate(final)
                print(f"{prefix}: cached, {rows:,} records", flush=True)
            else:
                print(f"{prefix}: {'resuming' if partial.exists() else 'downloading'}", flush=True)
                command = ["curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "4", "--retry-delay", "2", "--retry-all-errors",
                    "--connect-timeout", "20", "--max-time", "300",
                    "--continue-at", "-", "--output", str(partial), url]
                with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as process:
                    started = time.monotonic()
                    while True:
                        try:
                            _, error = process.communicate(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            size = partial.stat().st_size if partial.exists() else 0
                            print(f"  {size / 1048576:.2f} MiB downloaded | {time.monotonic() - started:.0f}s", flush=True)
                        except BaseException:
                            process.kill()
                            process.communicate()
                            raise
                    if process.returncode:
                        safe_error = re.sub(r"https?://\S+", "<download URL>", error.strip())
                        raise RuntimeError(safe_error or "Download failed")
                rows = validate(partial)
                partial.replace(final)
                print(f"{prefix}: saved, {rows:,} records, {final.stat().st_size:,} bytes", flush=True)
            total_rows += rows
        except (OSError, ValueError, RuntimeError) as error:
            failures += 1
            print(f"{prefix}: FAILED: {error}", flush=True)
    print(f"{len(downloads) - failures}/{len(downloads)} files ready | {total_rows:,} records | {args.output}", flush=True)
    if failures:
        print("Rerun to resume. HTTP 403 may require fresh export URLs.", flush=True)
        raise SystemExit(1)
PY
