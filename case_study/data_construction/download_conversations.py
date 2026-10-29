#!/usr/bin/env python3
"""Download full retained call transcripts and retrieval evidence by call ID.

Only keyed DynamoDB reads and S3 downloads are performed. No rewriting,
turn selection, KB chunk export, or production writes are performed.
"""

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
CONSTRUCTION = ROOT.parent.parent / "retrieval_eval/data_construction"
sys.path.insert(0, str(CONSTRUCTION))

from export_call_retrieval_config import (
    ENGINE_FIELDS, SESSION_FIELDS, Reader, engine_ref, summarize_scope,
)
from resolve_node_kb_scope import CDN_BUCKETS, atomic_json, download_json, normalize_log


def save_conversation(transcript, path):
    if not isinstance(transcript, list):
        raise ValueError("Expected the full transcript as a JSON array")
    lines = []
    for turn in transcript:
        if not isinstance(turn, dict):
            raise ValueError("Unexpected transcript entry")
        role = turn.get("role", "event")
        content = turn.get("content")
        if not isinstance(content, str):
            content = json.dumps(turn, ensure_ascii=False)
        lines.append(f"{role}: {content}\n")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def collect(args):
    call_ids = list(dict.fromkeys(args.call_ids.read_text().split()))
    if not call_ids or any(not re.fullmatch(r"call_[A-Za-z0-9_-]+", sid) for sid in call_ids):
        raise ValueError("Expected one call_... ID per line")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cdn_buckets = dict(CDN_BUCKETS)
    args.offline = False
    with (args.output / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another conversation downloader is running") from None
        reader = Reader(args)
        sessions = reader.batch(args.call_table, [{"callId": sid} for sid in call_ids],
                                ["callId"] + SESSION_FIELDS)
        sessions = {session["callId"]: session for session in sessions.values()}
        engine_keys = defaultdict(list)
        for session in sessions.values():
            ref = engine_ref(session, args)
            if ref:
                table, key, _ = ref
                engine_keys[table].append(key)
        engines, engine_error = {}, None
        try:
            for table, keys in engine_keys.items():
                engines[table] = reader.batch(table, keys, list(keys[0]) + ENGINE_FIELDS)
        except Exception as error:
            # Configuration lookup must not prevent downloading the conversation.
            engine_error = str(error)
            print(f"Engine configuration unavailable: {error}", flush=True)
        results = []
        for number, sid in enumerate(call_ids, 1):
            print(f"[{number}/{len(call_ids)}] {sid}", flush=True)
            result = {"call_id": sid}
            session = sessions.get(sid)
            if session is None:
                result["status"] = "call_not_found"
            else:
                folder = args.output / "calls" / sid
                folder.mkdir(parents=True, exist_ok=True)
                atomic_json(folder / "call.json", session)
                result.update(org_id=session.get("orgId"), files={})
                scope = {"org_id": session.get("orgId"), "logged_kb_ids": [],
                         "configured_candidate_kb_ids": [],
                         "note": "Session-wide KB references; not a per-query scope assignment."}
                for field, filename in (("transcriptUrl", "transcript.json"),
                                        ("knowledgeBaseRetrievedContentsUrl", "retrieval.json"),
                                        ("nodeTransitionUrl", "node_transition.json")):
                    destination = folder / filename
                    if not session.get(field) and not destination.exists():
                        result["files"][filename] = {"status": "no_retained_url"}
                        print(f"  {filename}: no retained URL", flush=True)
                        continue
                    cached = destination.exists()
                    try:
                        value = download_json(session.get(field), destination, args)
                        if filename == "transcript.json":
                            save_conversation(value, folder / "conversation.txt")
                            result["transcript_entries"] = len(value)
                        elif filename == "retrieval.json":
                            if not isinstance(value, list):
                                raise ValueError("Expected a retrieval-log JSON array")
                            scope["logged_kb_ids"] = sorted({kb for entry in value
                                for kb in normalize_log(entry)["knowledgeBaseIds"]})
                        status = "cached" if cached else "downloaded"
                        result["files"][filename] = {"status": status}
                        print(f"  {filename}: {status}, {destination.stat().st_size:,} bytes", flush=True)
                    except Exception as error:
                        result["files"][filename] = {"status": "failed", "error": str(error)}
                        print(f"  {filename}: failed: {error}", flush=True)
                ref = engine_ref(session, args)
                if ref:
                    table, key, kind = ref
                    scope["engine_reference"] = {"table": table, "key": key, "type": kind}
                    engine = engines.get(table, {}).get(json.dumps(key, sort_keys=True))
                    if engine and engine.get("orgId") == session.get("orgId"):
                        scope.update(summarize_scope(session, engine, kind))
                        scope["configured_candidate_kb_ids"] = sorted(
                            set(scope["configured_global_kb_ids"]) | {
                                kb for node in scope["node_scope_candidates"]
                                for kb in node["knowledgeBaseIds"]})
                    else:
                        scope["configuration_error"] = engine_error or "Engine version missing or organization differs"
                atomic_json(folder / "retrieval_config.json", scope)
                result["logged_kb_ids"] = scope["logged_kb_ids"]
                result["configured_candidate_kb_ids"] = scope["configured_candidate_kb_ids"]
                transcript_ok = result["files"].get("transcript.json", {}).get("status") in ("cached", "downloaded")
                result["status"] = "conversation_saved" if transcript_ok else "conversation_unavailable"
            results.append(result)
            atomic_json(args.output / "index.json", {"requested_calls": len(call_ids), "calls": results})
        saved = sum(r["status"] == "conversation_saved" for r in results)
        print(f"Saved {saved}/{len(call_ids)} full conversations under {args.output / 'calls'}", flush=True)
        return 1 if saved != len(call_ids) or any(
            item.get("status") == "failed" for r in results for item in r.get("files", {}).values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-ids", type=Path, default=ROOT / "call_ids.txt")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--profile", help="Optional AWS CLI profile when running Python directly")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached DynamoDB records; existing S3 downloads are reused")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("Delay must be nonnegative")
    args.call_table, args.chat_table = "call", "chat"
    args.llm_table, args.flow_table = "retell-stateful-llm-version", "conversation-flow-version"
    # Shared reader only requires this name for request progress messages.
    args.chunk_table = "knowledge-base-chunk"
    return collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        raise SystemExit(f"Conversation download stopped: {error or type(error).__name__}")
