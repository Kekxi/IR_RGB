"""Export a locally reviewed B3 Excel sheet to the server score CSV format.

Run with local Python that has openpyxl. The server-side scorer itself needs no
openpyxl; only this local convenience tool does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

from b3_discovery_score import _read_correspondence


CORE_FIELDS = ("id", "ir_gt_index", "rgb_gt_index", "status", "notes",
               "suggested_rgb_gt_index", "suggested_same_pixel_iou",
               "suggested_center_distance")
CHECKED = {"yes", "y", "true", "1", "checked", "已核对"}


def load_rows(workbook_path: Path) -> list[dict[str, str]]:
    book = load_workbook(workbook_path, read_only=False, data_only=True)
    sheet = book["BoxDraft"]
    values = sheet.values
    headers = [str(v).strip() if v is not None else "" for v in next(values)]
    required = set(CORE_FIELDS) | {"review_priority", "human_checked"}
    if not required.issubset(headers):
        raise ValueError(f"Missing columns: {sorted(required - set(headers))}")
    rows = []
    for values in values:
        if all(v is None for v in values):
            continue
        rows.append({field: "" if value is None else str(value).strip()
                     for field, value in zip(headers, values)})
    book.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all-checked", action="store_true")
    args = parser.parse_args()
    rows = load_rows(args.workbook)
    pairs = {pair["id"]: pair for pair in (json.loads(line) for line in
             (args.prepared / "pairs.jsonl").read_text(encoding="utf-8").splitlines())}
    expected = {(pid, i) for pid, pair in pairs.items()
                for i in range(len(pair["gt_ir"]))}
    submitted = [(row["id"], int(row["ir_gt_index"])) for row in rows]
    if len(rows) != len(expected) or set(submitted) != expected or len(set(submitted)) != len(rows):
        raise ValueError("Workbook row keys differ from prepared IR GT rows")
    checked = [row for row in rows if row["human_checked"].lower() in CHECKED]
    priority_unchecked = [row for row in rows if row["review_priority"] != "spot_check"
                          and row["human_checked"].lower() not in CHECKED]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="",
                                     suffix=".csv", delete=False,
                                     dir=output.parent) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=CORE_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CORE_FIELDS} for row in rows)
    try:
        _, score_audit = _read_correspondence(temp, pairs)
        report = {"rows": len(rows), "human_checked": len(checked),
                  "unchecked_priority": len(priority_unchecked),
                  "status_counts": dict(Counter(row["status"] for row in rows)),
                  "score_audit": score_audit,
                  "formal_human_review": len(checked) == len(rows),
                  "csv_sha256": hashlib.sha256(temp.read_bytes()).hexdigest()}
        audit_path = output.with_suffix(".audit.json")
        audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        if not score_audit["complete"]:
            raise ValueError(f"Correspondence incomplete: {score_audit['incomplete_rows']} unresolved; "
                             f"{len(score_audit['errors'])} errors. See {audit_path}")
        if priority_unchecked:
            raise ValueError(f"{len(priority_unchecked)} priority rows are not human checked. "
                             f"See {audit_path}")
        if args.require_all_checked and len(checked) != len(rows):
            raise ValueError(f"{len(rows) - len(checked)} rows are not human checked. "
                             f"See {audit_path}")
        output.write_bytes(temp.read_bytes())
        print(json.dumps({"server_csv": str(output), "human_checked": len(checked),
                          "rows": len(rows), "formal_human_review": report["formal_human_review"]},
                         ensure_ascii=False, indent=2))
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
