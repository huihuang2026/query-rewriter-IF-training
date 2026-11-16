#!/usr/bin/env python3
"""Combine JSONL datasets, optionally retaining each source group from its first file."""

import argparse
import json
from pathlib import Path
import tempfile


def merge(inputs, output, source_group_key=None):
    for path in inputs:
        if path.resolve() == output.resolve():
            raise ValueError("Output must differ from every input")
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_records, seen_groups = set(), set()
    kept = skipped = 0
    with tempfile.NamedTemporaryFile("w", dir=output.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for path in inputs:
                file_records, file_groups = set(), set()
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        rid = row["record_id"]
                        if rid in file_records:
                            raise ValueError(f"Duplicate record ID in {path}: {rid}")
                        file_records.add(rid)
                        if source_group_key:
                            group = row[source_group_key]
                            if group in seen_groups:
                                skipped += 1
                                continue
                            file_groups.add(group)
                        if rid in seen_records:
                            raise ValueError(f"Duplicate record ID across inputs: {rid}")
                        seen_records.add(rid)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        kept += 1
                seen_groups.update(file_groups)
            if not kept:
                raise ValueError("No records to merge; output unchanged")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    print(f"Saved {kept:,} records; skipped {skipped:,} overlapping-source records: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-group-key", help="Keep all rows of a group from its first input file")
    args = parser.parse_args()
    merge(args.input, args.output, args.source_group_key)


if __name__ == "__main__":
    main()
