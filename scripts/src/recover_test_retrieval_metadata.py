#!/usr/bin/env python3
"""Recover exact-query metadata from the local timestamped raw export chunks.

Does not infer historical KB assignments from organizations or current agents.
The original test set is preserved; output is an enriched JSONL copy.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re

from prepare_test_data import convert


CHUNK = re.compile(r"chunk_(.+)_to_(.+)_\d+\.jsonl$")
KB_KEYS = {"knowledgebaseid", "knowledgebaseids", "kbid", "kbids"}


def kb_fields(value, prefix=""):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if re.sub(r"[^a-z]", "", key.lower()) in KB_KEYS:
                found[path] = item
            found.update(kb_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(kb_fields(item, f"{prefix}[{index}]"))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.test.resolve():
        parser.error("Output must differ from the original test set")
    rows = [json.loads(line) for line in args.test.open() if line.strip()]
    targets = defaultdict(list)
    buckets = defaultdict(set)
    for index, row in enumerate(rows):
        source = row["source"]
        targets[(source["timestamp"], source["call_id"])].append(index)
        stamp = datetime.fromisoformat(source["timestamp"])
        buckets[stamp.replace(minute=stamp.minute // 10 * 10, second=0, microsecond=0)].add(
            source["call_id"].encode()
        )
    matches = defaultdict(list)
    metadata_keys = Counter()
    scanned = 0
    for path in sorted(args.raw_dir.glob("*.jsonl")):
        match = CHUNK.fullmatch(path.name)
        if not match:
            continue
        start, end = (datetime.fromisoformat(s.replace("_", ":")) for s in match.groups())
        calls = set().union(*(ids for stamp, ids in buckets.items() if start <= stamp < end))
        if not calls:
            continue
        scanned += 1
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not any(call in line for call in calls):
                    continue
                raw = json.loads(line)
                metadata = raw.get("metadata") or {}
                call_id = metadata.get("call_id") or raw.get("thread_identifier")
                indices = targets.get((raw.get("timestamp"), call_id), [])
                if not indices:
                    continue
                source = convert(raw)
                for index in indices:
                    if source != rows[index]["source"]:
                        continue
                    metadata_keys.update(metadata.keys())
                    matches[index].append({
                        "raw_export_file": str(path),
                        "raw_export_line": line_number,
                        "agent_id": metadata.get("agent_id"),
                        "language": metadata.get("language"),
                        "agentLanguages": metadata.get("agentLanguages"),
                        "knowledge_base_fields": kb_fields(raw),
                    })
        if scanned % 50 == 0:
            print(f"Scanned {scanned} chunks; matched {len(matches)}/{len(rows)} queries", flush=True)
    if len(matches) != len(rows):
        raise ValueError(f"Only matched {len(matches)}/{len(rows)}; no output written")
    for index, row in enumerate(rows):
        recovered = matches[index]
        profiles = {
            json.dumps({k: v for k, v in item.items() if not k.startswith("raw_export_")}, sort_keys=True)
            for item in recovered
        }
        if len(profiles) != 1:
            raise ValueError(f"Conflicting raw metadata for {row['record_id']}")
        info = json.loads(next(iter(profiles)))
        info["knowledge_base_ids"] = None
        info["knowledge_base_status"] = (
            "inspect_raw_fields" if info["knowledge_base_fields"] else "not_logged_in_raw_export"
        )
        info["provenance"] = [
            {key: value for key, value in item.items() if key.startswith("raw_export_")}
            for item in recovered
        ]
        row["retrieval_metadata"] = info
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "test_file": str(args.test), "output_file": str(args.output),
        "total_queries": len(rows), "exact_source_matches": len(matches),
        "raw_chunks_scanned": scanned,
        "queries_with_agent_id": sum(bool(r["retrieval_metadata"]["agent_id"]) for r in rows),
        "unique_agent_ids": len({r["retrieval_metadata"]["agent_id"] for r in rows} - {None}),
        "queries_with_language": sum(r["retrieval_metadata"]["language"] is not None for r in rows),
        "queries_with_agentLanguages": sum(r["retrieval_metadata"]["agentLanguages"] is not None for r in rows),
        "queries_with_kb_fields": sum(bool(r["retrieval_metadata"]["knowledge_base_fields"]) for r in rows),
        "raw_metadata_keys": sorted(metadata_keys),
        "next_step": "Join call_id and query timestamp to historical call/agent configuration or retrieval logs to recover actual KB scope. Query-rewriter exports do not establish the backend retrieval language parameter.",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
