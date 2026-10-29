#!/usr/bin/env python3
"""Align a case-study spreadsheet with test records and append paired model answers."""

import argparse
import csv
import json
from pathlib import Path
import tempfile

import test_query_rewriter as rewriter

MODES = ("with_user_instruction", "without_user_instruction")


def read_sheet(path):
    # Do not use DictReader: this spreadsheet has duplicate empty column names.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or rows[0][:3] != ["record_id", "user_defined_instruction", "conversation"]:
        raise ValueError("Expected record_id/instruction/conversation columns and two header rows")
    if any(rows[1][:3]) or rows[1][3:5] != ["With Inst", "Without Inst"]:
        raise ValueError("Expected a second header row labeling With Inst / Without Inst")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Inconsistent CSV column counts")
    ids = [row[0] for row in rows[2:]]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("CSV sample record IDs must be nonempty and unique")
    return rows


def canonical_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def aligned_records(sheet, test_file):
    wanted = {row[0] for row in sheet[2:]}
    matched = {}
    for row in rewriter.records(test_file, None):
        key = row["record_id"]
        if key in wanted:
            if key in matched:
                raise ValueError(f"Duplicate test record ID: {key}")
            matched[key] = row
    missing = wanted - matched.keys()
    if missing:
        raise ValueError(f"CSV IDs missing from test set: {sorted(missing)}")
    result = []
    for cells in sheet[2:]:
        row = matched[cells[0]]
        source = row["source"]
        instruction = rewriter.additional_instruction(source["system_prompt"])
        if not instruction or canonical_newlines(instruction) != canonical_newlines(cells[1].strip()):
            raise ValueError(f"User instruction differs from test set: {cells[0]}")
        if canonical_newlines(source["user_content"]) != canonical_newlines(cells[2]):
            raise ValueError(f"Conversation differs from test set: {cells[0]}")
        result.append(row)
    return result


def prepare(args):
    if args.output.resolve() in {args.input_csv.resolve(), args.test_file.resolve()}:
        raise ValueError("Selected test output must differ from input files")
    rows = aligned_records(read_sheet(args.input_csv), args.test_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            rewriter.write_record(handle, row)
    print(f"Selected {len(rows)} test records in CSV order: {args.output}")


def checked_answers(path, expected, model_name):
    saved = {}
    for row in rewriter.records(path, None):
        key = row["record_id"]
        if key in saved or key not in expected:
            raise ValueError(f"Duplicate/unexpected generated ID in {path}: {key}")
        if row.get("generation_model") != model_name:
            raise ValueError(f"Unexpected model in {path}: {key}")
        source = expected[key]["source"]
        if any(row["source"][field] != source[field] for field in ("system_prompt", "user_content")):
            raise ValueError(f"Generation input does not match test record: {key}")
        branches = rewriter.paired_generations(row)
        answers = [branches[mode].get("generated_query") for mode in MODES]
        if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError(f"Missing generated response: {key}")
        saved[key] = answers
    if saved.keys() != expected.keys():
        raise ValueError(f"Incomplete responses in {path}: expected {len(expected)}, found {len(saved)}")
    return saved


def export(args):
    sheet = read_sheet(args.input_csv)
    expected = {row["record_id"]: row for row in aligned_records(sheet, args.test_file)}
    protected = [args.input_csv, args.test_file, *(Path(path) for _, path in args.model_output)]
    if args.output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("CSV output must differ from all input files")
    names = set()
    for name, path in args.model_output:
        if name in names or name in sheet[0]:
            raise ValueError(f"Duplicate model column: {name}")
        names.add(name)
        answers = checked_answers(Path(path), expected, name)
        sheet[0].extend([name, ""])
        sheet[1].extend(["With Inst", "Without Inst"])
        for cells in sheet[2:]:
            cells.extend(answers[cells[0]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="",
                                         dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            csv.writer(handle).writerows(sheet)
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"CSV: {args.output} | {len(expected)} samples | {2 * len(names)} new columns")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("export", export)):
        command = commands.add_parser(name)
        command.add_argument("--input-csv", type=Path, required=True)
        command.add_argument("--test-file", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.set_defaults(function=function)
        if name == "export":
            command.add_argument("--model-output", nargs=2, action="append", required=True,
                                 metavar=("MODEL_NAME", "GENERATED_JSON"))
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
