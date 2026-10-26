"""Read-only audit of human-reviewed validation visibility labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED = {"clear", "haze_fog", "rain_snow", "sea_spray", "low_visibility_unknown", "uncertain"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--strict", action="store_true", help="Fail if any validation pair remains blank")
    args = p.parse_args()
    with (args.dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        eligible = {row["id"]: row for row in csv.DictReader(handle)
                    if row["split"] == "val" and row["same_image_size"] == "1"}
    with args.conditions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["id"] for row in rows]
    if len(eligible) != 877 or len(rows) != 877 or len(set(ids)) != 877 or set(ids) != set(eligible):
        raise ValueError("Expected exactly one visibility row for each of the 877 eligible val pairs")
    by_event: dict[str, Counter] = defaultdict(Counter)
    invalid = []
    for row in rows:
        value = (row["visibility"] or "").strip().lower()
        if value and value not in ALLOWED:
            invalid.append((row["id"], value))
        if row["event"] != eligible[row["id"]]["event"]:
            raise ValueError(f"Event mismatch for {row['id']}")
        by_event[row["event"]][value or "unreviewed"] += 1
    if invalid:
        raise ValueError(f"Invalid visibility values (first 10): {invalid[:10]}")
    total = Counter()
    for event, counts in sorted(by_event.items()):
        total.update(counts)
        print(event, dict(sorted(counts.items())))
    print("TOTAL", dict(sorted(total.items())))
    if args.strict and total["unreviewed"]:
        raise SystemExit(f"{total['unreviewed']} pairs still need review")


if __name__ == "__main__":
    main()
