"""Build a small B3-vs-independent review queue from full V4 fog exports.

GT correspondence and video tracks remain provisional. This script never edits
the original annotation CSV or claims event-level ground truth identities.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from b3_discovery_score import _jsonl, _native_matches, _predictions
from b3_discovery_review_tracks import track_modality


VARIANTS = ("rgb_only", "ir_only", "b3_rgb", "b3_ir")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    root = args.full_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not 0 <= args.conf <= 1:
        raise ValueError("Invalid confidence threshold")
    pairs = {entry["id"]: entry for entry in _jsonl(root / "prepared" / "pairs.jsonl")}
    if len(pairs) != 633:
        raise ValueError(f"Expected 633 fog pairs, found {len(pairs)}")
    rgb_tracks = track_modality(list(pairs.values()), "rgb")
    exports = _predictions(pairs, root / "exports")
    matched = _native_matches(pairs, exports, args.iou)
    conf = {}
    fp = Counter()
    for variant in VARIANTS:
        for pair_id, value, index in matched[variant]:
            if index is None:
                fp[variant] += value >= args.conf
            else:
                conf[(variant, pair_id, index)] = value

    with args.draft.open("r", encoding="utf-8-sig", newline="") as handle:
        draft = list(csv.DictReader(handle))
    expected = sum(len(pair["gt_ir"]) for pair in pairs.values())
    if len(draft) != expected:
        raise ValueError(f"Draft row count {len(draft)} != {expected}")
    seen_ir = set()
    linked_rgb = set()
    possible_rgb = set()
    for row in draft:
        key = (row["id"], int(row["ir_gt_index"]))
        if key in seen_ir or key[0] not in pairs or key[1] >= len(pairs[key[0]]["gt_ir"]):
            raise ValueError(f"Invalid or duplicate IR row: {key}")
        seen_ir.add(key)
        if row["status"].startswith("shared_"):
            rgb_key = (row["id"], int(row["rgb_gt_index"]))
            if rgb_key in linked_rgb:
                raise ValueError(f"Two IR GTs mapped to RGB GT {rgb_key}")
            linked_rgb.add(rgb_key)
        elif row["status"] == "uncertain":
            candidate_text = row.get("auto_candidate_rgb_index") or row.get("suggested_rgb_gt_index")
            if candidate_text:
                possible_rgb.add((row["id"], int(candidate_text)))

    def value(variant: str, pair_id: str, index: int | None) -> float:
        return 0.0 if index is None else conf.get((variant, pair_id, index), 0.0)

    def hit(variant: str, pair_id: str, index: int | None) -> bool:
        return index is not None and value(variant, pair_id, index) >= args.conf

    queue = []
    scenario_counter = Counter()
    for row in draft:
        pair_id = row["id"]
        index = int(row["ir_gt_index"])
        status = row["status"]
        rgb_index = int(row["rgb_gt_index"]) if row["rgb_gt_index"] else None
        candidate = (int(row["auto_candidate_rgb_index"])
                     if row.get("auto_candidate_rgb_index") else None)
        if candidate is None and row.get("suggested_rgb_gt_index"):
            candidate = int(row["suggested_rgb_gt_index"])
        if candidate is not None and not 0 <= candidate < len(pairs[pair_id]["gt_rgb"]):
            candidate = None
        if status.startswith("shared_"):
            possibilities = [("shared", rgb_index)]
        elif status.startswith("ir_only_"):
            possibilities = [("ir_only", None)]
        elif status == "uncertain":
            possibilities = [("ir_only_if_unpaired", None)]
            if candidate is not None and (pair_id, candidate) not in linked_rgb:
                possibilities.append(("shared_if_same_ship", candidate))
        else:
            raise ValueError(f"Unexpected draft status: {pair_id} I{index} {status}")
        scenario = []
        for label, proposed_rgb in possibilities:
            independent = hit("ir_only", pair_id, index) or hit("rgb_only", pair_id, proposed_rgb)
            b3 = hit("b3_ir", pair_id, index) or hit("b3_rgb", pair_id, proposed_rgb)
            direction = "b3_only" if b3 and not independent else (
                "independent_only" if independent and not b3 else "tie")
            scenario.append((label, direction))
            scenario_counter[(status, label, direction)] += 1
        directions = {direction for _, direction in scenario}
        if directions == {"tie"}:
            continue
        display_rgb = rgb_index if rgb_index is not None else candidate
        review_image = root / "prepared" / "review_images" / f"{pair_id}.jpg"
        queue.append({
            "pair_id": pair_id,
            "event": pairs[pair_id]["event"],
            "target": f"I{index}",
            "ir_gt_index": index,
            "rgb_candidate": "" if display_rgb is None else f"R{display_rgb}",
            "draft_status": status,
            "scenario_outcomes": "; ".join(f"{label}:{direction}" for label, direction in scenario),
            "correspondence_changes_outcome": "yes" if len(directions) > 1 else "no",
            "ir_track_id": row.get("ir_track_id", ""),
            "rgb_track_id": row.get("rgb_track_id", ""),
            "rgb_only_conf": round(value("rgb_only", pair_id, display_rgb), 4),
            "b3_rgb_conf": round(value("b3_rgb", pair_id, display_rgb), 4),
            "ir_only_conf": round(value("ir_only", pair_id, index), 4),
            "b3_ir_conf": round(value("b3_ir", pair_id, index), 4),
            "review_image": str(review_image),
            "reviewed_status": "",
            "reviewed_rgb_gt_index": "",
            "physical_ship_id": "",
            "review_notes": "",
        })

    # RGB GTs not already paired with an IR GT in the draft may also change.
    for pair_id, pair in pairs.items():
        for index in range(len(pair["gt_rgb"])):
            if (pair_id, index) in linked_rgb:
                continue
            a = hit("rgb_only", pair_id, index)
            b = hit("b3_rgb", pair_id, index)
            if a == b:
                continue
            row = {
                "pair_id": pair_id, "event": pair["event"], "target": f"R{index}",
                "ir_gt_index": "", "rgb_candidate": f"R{index}",
                "draft_status": "rgb_unlinked_in_draft",
                "scenario_outcomes": "rgb_unlinked:" + ("b3_only" if b else "independent_only"),
                "correspondence_changes_outcome": (
                    "possible" if (pair_id, index) in possible_rgb else "no"),
                "ir_track_id": "", "rgb_track_id": rgb_tracks[(pair_id, index)],
                "rgb_only_conf": round(value("rgb_only", pair_id, index), 4),
                "b3_rgb_conf": round(value("b3_rgb", pair_id, index), 4),
                "ir_only_conf": "", "b3_ir_conf": "",
                "review_image": str(root / "prepared" / "review_images" / f"{pair_id}.jpg"),
                "reviewed_status": "", "reviewed_rgb_gt_index": "",
                "physical_ship_id": "", "review_notes": "",
            }
            queue.append(row)
    queue.sort(key=lambda row: (row["event"], row["pair_id"], row["target"]))

    # A track ID here is a geometry-based review group, not a verified ship ID.
    grouped = defaultdict(list)
    for row in queue:
        provisional_track = row["ir_track_id"] or row["rgb_track_id"] or row["target"]
        grouped[(row["event"], provisional_track)].append(row)
    groups = []
    for (event, track), rows in grouped.items():
        rows.sort(key=lambda row: row["pair_id"])
        representative = rows[len(rows) // 2]
        groups.append({
            "event": event, "candidate_track": track, "affected_frames": len(rows),
            "first_frame": rows[0]["pair_id"],
            "representative_frame": representative["pair_id"],
            "last_frame": rows[-1]["pair_id"],
            "b3_only_frames": sum("b3_only" in row["scenario_outcomes"] for row in rows),
            "independent_only_frames": sum("independent_only" in row["scenario_outcomes"] for row in rows),
            "correspondence_sensitive_frames": sum(row["correspondence_changes_outcome"] != "no" for row in rows),
            "representative_image": representative["review_image"],
            "physical_ship_id": "", "reviewed_result": "", "review_notes": "",
        })
    groups.sort(key=lambda group: (-group["affected_frames"], group["event"], group["candidate_track"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "disagreement_frames.csv", queue, list(queue[0]) if queue else [])
    write_csv(output / "candidate_ship_groups.csv", groups, list(groups[0]) if groups else [])
    summary = {
        "provenance": "geometry_draft_and_native_gt_matches; no human-reviewed correspondence",
        "pairs": len(pairs), "rgb_gt": sum(len(pair["gt_rgb"]) for pair in pairs.values()),
        "ir_gt": expected, "conf": args.conf, "iou": args.iou,
        "draft_status": dict(Counter(row["status"] for row in draft)),
        "disagreement_rows": len(queue), "affected_frames": len({row["pair_id"] for row in queue}),
        "candidate_ship_groups": len(groups),
        "correspondence_sensitive_rows": sum(row["correspondence_changes_outcome"] != "no" for row in queue),
        "native_tp": {variant: sum(value(variant, pair_id, index) >= args.conf
                                   for pair_id, pair in pairs.items()
                                   for index in range(len(pair["gt_ir"] if variant in ("ir_only", "b3_ir") else pair["gt_rgb"])))
                      for variant in VARIANTS},
        "native_fp": {variant: fp[variant] for variant in VARIANTS},
        "scenario_outcomes": {"|".join(key): value for key, value in scenario_counter.items()},
        "warning": "Do not count candidate tracks as confirmed physical ships or score all 633 unique ships from unresolved GT links.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("pairs", "disagreement_rows", "affected_frames",
                                                 "candidate_ship_groups", "correspondence_sensitive_rows",
                                                 "native_tp", "native_fp")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
