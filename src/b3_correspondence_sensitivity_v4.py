"""Create two transparent, provisional GT-link scenarios and score B3 discovery.

The scenarios deliberately bracket unresolved RGB/IR correspondence. They are
diagnostic only: neither is human-verified, and the scorer's IDs are per frame.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from b3_discovery_score import score


WORKSPACE = Path(r"E:\HEU_4151\RGB_IR")
ROOT = WORKSPACE / "fog_ship_experiments_20261019" / "eval_b3_discovery_v4"
DRAFT = WORKSPACE / "fog_ship_experiments_20261019" / "b3_discovery_review_assisted" / "correspondence_review_draft.csv"
OUT = ROOT / "targeted_review_v3" / "sensitivity"

# User-confirmed: I4 and I6 at 11:11:56 are two physical ships. Their exact
# RGB link does not affect the key finding because both IR heads detect both.
OVERRIDES = {
    ("mtdsp_01_20250117101938", 0): ("ir_only_common_fov", None),
    ("mtdsp_05_20250117104221", 0): ("shared_rgb_uncertain", 3),
    ("mtdsp_07_20250117110915", 4): ("shared_rgb_uncertain", 3),
    ("mtdsp_07_20250117111156", 4): ("ir_only_common_fov", None),
    ("mtdsp_07_20250117111156", 6): ("shared_rgb_uncertain", 4),
    ("mtdsp_07_20250117111838", 1): ("shared_rgb_uncertain", 5),
    ("mtdsp_07_20250117111853", 2): ("shared_rgb_uncertain", 2),
    ("mtdsp_07_20250117111904", 1): ("shared_rgb_uncertain", 2),
    ("mtdsp_13_20250314154739", 0): ("ir_only_common_fov", None),
    ("mtdsp_13_20250314154739", 1): ("shared_rgb_uncertain", 0),
    ("mtdsp_13_20250314154744", 0): ("ir_only_common_fov", None),
    ("mtdsp_13_20250314154744", 1): ("shared_rgb_uncertain", 0),
    ("mtdsp_13_20250314154749", 0): ("ir_only_common_fov", None),
    ("mtdsp_13_20250314154749", 1): ("shared_rgb_uncertain", 0),
}


def make_scenario(rows: list[dict], name: str) -> Path:
    output = OUT / f"correspondence_{name}.csv"
    result = [dict(row) for row in rows]
    for row in result:
        key = (row["id"], int(row["ir_gt_index"]))
        if key in OVERRIDES:
            status, rgb_index = OVERRIDES[key]
            row["status"] = status
            row["rgb_gt_index"] = "" if rgb_index is None else str(rgb_index)
            row["notes"] = "ASSISTANT_PROVISIONAL_OVERRIDE; " + row["notes"]
    used = defaultdict(set)
    for row in result:
        if row["status"].startswith("shared_rgb_"):
            rgb_index = int(row["rgb_gt_index"])
            if rgb_index in used[row["id"]]:
                raise ValueError(f"Duplicate RGB link at {row['id']} R{rgb_index}")
            used[row["id"]].add(rgb_index)
    unresolved = [row for row in result if row["status"] == "uncertain"]
    unresolved.sort(key=lambda row: float(row["suggested_center_distance"] or 2.0))
    for row in unresolved:
        candidate = row["suggested_rgb_gt_index"]
        distance = float(row["suggested_center_distance"] or 2.0)
        if name == "linked_where_possible" and candidate and distance <= 0.15 and int(candidate) not in used[row["id"]]:
            row["status"] = "shared_rgb_uncertain"
            row["rgb_gt_index"] = candidate
            used[row["id"]].add(int(candidate))
            decision = "AUTO_LINK_DISTANCE_LE_0.15"
        else:
            row["status"] = "ir_only_common_fov"
            row["rgb_gt_index"] = ""
            decision = "AUTO_KEEP_IR_DISTINCT"
        row["notes"] = decision + "; " + row["notes"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(result)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with DRAFT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reports = {}
    for name in ("all_unresolved_distinct", "linked_where_possible"):
        correspondence = make_scenario(rows, name)
        report = score(ROOT / "prepared", ROOT / "exports", correspondence,
                       OUT / name, budgets=[0.25, 0.5, 0.75],
                       correspondence_provenance="assistant_assisted_preliminary")
        reports[name] = report
        print(name, report["status"], report.get("unique_gt"))
    summary = {"provenance": "assistant-assisted, unresolved GT links bracketed; NOT a human-reviewed physical-ship result",
               "unit": "unique GT target within each paired frame, not unique ship across video",
               "scenarios": {name: {"unique_gt": report.get("unique_gt"),
                                    "fixed_conf025": {system: {key: row[key] for key in ("unique_tp", "fppi")}
                                                      for system, row in report.get("fixed_conf025", {}).items()},
                                    "same_fppi_budget": {budget: {system: {key: row[key] for key in ("unique_tp", "fppi", "thresholds")}
                                                                        for system, row in systems.items()}
                                                         for budget, systems in report.get("same_fppi_budget", {}).items()}}
                             for name, report in reports.items()}}
    (OUT / "scenario_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
