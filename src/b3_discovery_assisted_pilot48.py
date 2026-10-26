"""Resolve the 48-pair B3 pilot provisionally and run exploratory scoring.

This is an assistant/geometry-assisted correspondence table. It must never be
described as fully human-reviewed ground truth. The source draft is untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from b3_discovery_score import _read_correspondence, score


# IR-index decisions after inspecting the paired RGB/IR review composites.
# A None RGB index means there is no corresponding annotated RGB box.
DECISIONS = {
    ("mtdsp_01_20250117101737", 1): ("ir_only_outside_rgb_fov", None, "IR ship at extreme left; RGB zoom/FOV does not establish common coverage"),
    ("mtdsp_01_20250117102105", 0): ("ir_only_common_fov", None, "Small IR ship near left; faint RGB counterpart has no RGB box"),
    ("mtdsp_01_20250117102436", 1): ("ir_only_common_fov", None, "Distant IR ship has no matching annotated RGB box"),
    ("mtdsp_03_20250312101514", 1): ("ir_only_common_fov", None, "Narrow distant IR ship has no matching annotated RGB box"),
    ("mtdsp_05_20250117104103", 2): ("ir_only_common_fov", None, "Bright IR vessel has no matching annotated RGB box"),
    ("mtdsp_05_20250117104915", 1): ("ir_only_common_fov", None, "Left IR horizon ship has no matching annotated RGB box"),
    ("mtdsp_05_20250117105119", 0): ("ir_only_common_fov", None, "Small horizon ship at far right, distinct from large R0/I3 vessel"),
    ("mtdsp_05_20250117105119", 5): ("ir_only_common_fov", None, "Small horizon ship is distinct from the matched large vessels"),
    ("mtdsp_07_20250117111141", 5): ("ir_only_common_fov", None, "Separate left horizon vessel next to I3/R4"),
    ("mtdsp_07_20250117111537", 3): ("ir_only_common_fov", None, "Distant IR vessel has no annotated RGB counterpart"),
    ("mtdsp_07_20250117111937", 0): ("shared_rgb_uncertain", 2, "Small nearby vessel corresponds to RGB R2"),
    ("mtdsp_07_20250117112131", 0): ("shared_rgb_uncertain", 1, "Small lower vessel corresponds to RGB R1"),
    ("mtdsp_07_20250117112328", 0): ("ir_only_common_fov", None, "Distant horizon IR ship has no matching RGB annotation"),
    ("mtdsp_07_20250117112328", 1): ("shared_rgb_uncertain", 0, "Small lower vessel corresponds to RGB R0"),
    ("mtdsp_09_20250117103141", 0): ("shared_rgb_uncertain", 3, "Right edge large vessel corresponds to RGB R3"),
    ("mtdsp_09_20250117103247", 1): ("ir_only_common_fov", None, "Small left IR ship has no annotated RGB counterpart"),
    ("mtdsp_09_20250117103247", 5): ("ir_only_common_fov", None, "Left partial vessel is visible but lacks an RGB annotation"),
    ("mtdsp_09_20250117103441", 4): ("ir_only_common_fov", None, "Right edge horizon vessel is distinct from I0/R4"),
    ("mtdsp_13_20250314154739", 0): ("ir_only_common_fov", None, "Left horizon IR ship has no RGB box"),
    ("mtdsp_13_20250314154739", 1): ("shared_rgb_uncertain", 0, "Right horizon vessel corresponds to RGB R0"),
    ("mtdsp_13_20250314154744", 0): ("ir_only_common_fov", None, "Left horizon IR ship has no RGB box"),
    ("mtdsp_13_20250314154744", 1): ("shared_rgb_uncertain", 0, "Right horizon vessel corresponds to RGB R0"),
    ("mtdsp_13_20250314154749", 0): ("ir_only_common_fov", None, "Left horizon IR ship has no RGB box"),
    ("mtdsp_13_20250314154749", 1): ("shared_rgb_uncertain", 0, "Right horizon vessel corresponds to RGB R0"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.pilot_root
    prepared = root / "prepared"
    source = prepared / "correspondence_review.csv"
    target = prepared / "correspondence_assisted.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    used = set()
    for row in rows:
        key = (row["id"], int(row["ir_gt_index"]))
        if key in DECISIONS:
            category, rgb_index, reason = DECISIONS[key]
            row["status"] = category
            row["rgb_gt_index"] = "" if rgb_index is None else str(rgb_index)
            row["notes"] = "ASSISTANT_VISUAL_PROVISIONAL; " + reason
            used.add(key)
        elif row["status"] == "uncertain":
            raise ValueError(f"No decision for uncertain row: {key}")
        # Keep the source's blank human_checked field. Do not forge a human audit.
        row["human_checked"] = ""
    if used != set(DECISIONS):
        raise ValueError(f"Missing decision rows: {set(DECISIONS) - used}")
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pairs = {entry["id"]: entry for entry in (
        json.loads(line) for line in (prepared / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )}
    _, audit = _read_correspondence(target, pairs)
    if not audit["complete"]:
        raise ValueError(audit)
    decision_log = {
        "provenance": "assistant_assisted_preliminary",
        "human_checked": False,
        "source_csv": str(source),
        "assisted_csv": str(target),
        "csv_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "pairs": len(pairs),
        "rows": len(rows),
        "visual_decisions": len(DECISIONS),
        "remaining_geometry_draft_rows": len(rows) - len(DECISIONS),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "audit": audit,
        "decision_details": [
            {"id": pair_id, "ir_gt_index": index, "status": value[0],
             "rgb_gt_index": value[1], "reason": value[2]}
            for (pair_id, index), value in DECISIONS.items()
        ],
    }
    (prepared / "assistant_decisions.json").write_text(
        json.dumps(decision_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = score(prepared, root / "exports", target, root / "scores_assisted",
                   budgets=[0.1, 0.25, 0.5],
                   correspondence_provenance="assistant_assisted_preliminary")
    print(json.dumps({"status": result["status"], "pairs": len(pairs),
                      "rows": len(rows), "unique_gt": result.get("unique_gt"),
                      "status_counts": decision_log["status_counts"],
                      "score_dir": str(root / "scores_assisted")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
