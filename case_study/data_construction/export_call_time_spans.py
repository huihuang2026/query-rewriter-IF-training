#!/usr/bin/env python3
"""Export call time spans and padded Keywords AI windows from local call records."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SHANGHAI = ZoneInfo("Asia/Shanghai")


def from_milliseconds(record, field):
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Missing or invalid {field}")
    return datetime.fromtimestamp(value / 1000, timezone.utc)


def utc_text(value):
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def export(call_ids, calls_dir, output, padding_seconds):
    ids = list(dict.fromkeys(call_ids.read_text().split()))
    if not ids or any(not re.fullmatch(r"call_[A-Za-z0-9_-]+", sid) for sid in ids):
        raise ValueError("Expected one call_... ID per line")
    if output.resolve() == call_ids.resolve():
        raise ValueError("Output must differ from the input call-ID file")
    padding = timedelta(seconds=padding_seconds)
    sections = [
        "Call time spans for Keywords AI exports",
        "UTC timestamps end in Z; Shanghai timestamps use UTC+08:00.",
        f"Suggested export windows include {padding_seconds} seconds before and after each call.",
        "Filter exported logs by call ID and metadata.use_case = query_rewriter.",
        "Missing timestamps are reported explicitly; no time span is guessed.",
        "",
    ]
    complete = 0
    for sid in ids:
        path = calls_dir / sid / "call.json"
        sections.append(f"Call ID: {sid}")
        try:
            record = json.loads(path.read_text())
            if record.get("callId") != sid:
                raise ValueError("Call ID differs from the requested ID")
            sections.append(f"Organization: {record.get('orgId', 'unknown')}")
            start = from_milliseconds(record, "startTimestamp")
            sections.append(f"Call start (UTC):      {utc_text(start)}")
            sections.append(f"Call start (Shanghai): {start.astimezone(SHANGHAI).isoformat(timespec='milliseconds')}")
            end = from_milliseconds(record, "endTimestamp")
            if end < start:
                raise ValueError("Call end precedes call start")
            sections.append(f"Call end   (UTC):      {utc_text(end)}")
            sections.append(f"Call end   (Shanghai): {end.astimezone(SHANGHAI).isoformat(timespec='milliseconds')}")
            sections.append(f"Duration: {(end - start).total_seconds():.3f} seconds")
            sections.append(f"Export from (UTC): {utc_text(start - padding)}")
            sections.append(f"Export to   (UTC): {utc_text(end + padding)}")
            complete += 1
        except (OSError, ValueError, TypeError, OverflowError) as error:
            sections.append(f"UNAVAILABLE: {error}")
        sections.append("")
    sections.append(f"Complete time spans: {complete}/{len(ids)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(sections) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Saved {complete}/{len(ids)} complete call time spans to {output}")
    return 0 if complete == len(ids) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-ids", type=Path, default=ROOT / "call_ids.txt")
    parser.add_argument("--calls-dir", type=Path, default=ROOT / "calls")
    parser.add_argument("--output", type=Path, default=ROOT / "call_time_spans.txt")
    parser.add_argument("--padding-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.padding_seconds < 0:
        parser.error("Padding must be nonnegative")
    return export(args.call_ids, args.calls_dir, args.output, args.padding_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(f"Time-span export stopped: {error or type(error).__name__}")
