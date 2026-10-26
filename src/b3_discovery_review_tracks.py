"""Group per-frame B3 correspondence proposals into candidate ship tracks.

Tracks are approximate review aids, never ground-truth identities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def track_modality(pairs: list[dict], modality: str) -> dict[tuple[str, int], str]:
    by_camera = defaultdict(list)
    for pair in pairs:
        camera = pair["id"].split("_")[1]
        by_camera[camera].append(pair)
    mapping = {}
    for camera, sequence in by_camera.items():
        sequence.sort(key=lambda p: p["id"])
        tracks = {}
        next_number = 1
        for pair in sequence:
            image_time = datetime.strptime(pair["id"].split("_")[2], "%Y%m%d%H%M%S")
            boxes = pair[f"gt_{modality}"]
            eligible = [(tid, box) for tid, (last_time, box) in tracks.items()
                        if (image_time - last_time).total_seconds() <= 18]
            cost = np.full((len(boxes), len(eligible) + len(boxes)), 100.0)
            for i, box in enumerate(boxes):
                for j, (_, old) in enumerate(eligible):
                    distance = math.dist(center(box), center(old))
                    w = max(box[2] - box[0], old[2] - old[0])
                    h = max(box[3] - box[1], old[3] - old[1])
                    if distance <= max(0.035, 0.45 * w) and abs(center(box)[1] - center(old)[1]) < max(0.035, 0.6 * h):
                        cost[i, j] = distance / 0.055
                cost[i, len(eligible) + i] = 1.0
            assigned = dict(zip(*linear_sum_assignment(cost))) if boxes else {}
            for i, box in enumerate(boxes):
                j = int(assigned[i])
                if j < len(eligible) and cost[i, j] < 1.0:
                    tid = eligible[j][0]
                else:
                    tid = f"{camera}-{modality[0].upper()}{next_number:03d}"
                    next_number += 1
                mapping[(pair["id"], i)] = tid
                tracks[tid] = (image_time, box)
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    pairs = [json.loads(line) for line in (args.prepared / "pairs.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    rgb_tracks = track_modality(pairs, "rgb")
    ir_tracks = track_modality(pairs, "ir")
    with args.draft.open(encoding="utf-8-sig", newline="") as stream:
        draft = list(csv.DictReader(stream))
    for row in draft:
        key = (row["id"], int(row["ir_gt_index"]))
        row["ir_track_id"] = ir_tracks[key]
        row["rgb_track_id"] = (rgb_tracks[(row["id"], int(row["rgb_gt_index"]))]
                               if row["rgb_gt_index"] else "")
    groups = defaultdict(list)
    for row in draft:
        groups[row["ir_track_id"]].append(row)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    by_track = []
    for track_id, rows in groups.items():
        rows.sort(key=lambda r: r["id"])
        rgb_vote = Counter(r["rgb_track_id"] for r in rows if r["rgb_track_id"])
        statuses = Counter(r["status"] for r in rows)
        priorities = Counter(r["review_priority"] for r in rows)
        representative = rows[len(rows) // 2]
        by_track.append({
            "ir_track_id": track_id,
            "frames": len(rows),
            "first_id": rows[0]["id"],
            "representative_id": representative["id"],
            "last_id": rows[-1]["id"],
            "candidate_rgb_track": rgb_vote.most_common(1)[0][0] if rgb_vote else "",
            "candidate_vote_frames": rgb_vote.most_common(1)[0][1] if rgb_vote else 0,
            "other_rgb_tracks": ";".join(f"{tid}:{num}" for tid, num in rgb_vote.most_common()[1:]),
            "shared_frames": statuses["shared_rgb_uncertain"],
            "ir_only_frames": statuses["ir_only_common_fov"],
            "uncertain_frames": statuses["uncertain"],
            "review_match_frames": priorities["review_match"],
            "review_ir_only_frames": priorities["review_ir_only"],
            "representative_image": str((args.prepared / "review_images" /
                                          f"{representative['id']}.jpg").resolve()),
            "human_checked": "",
            "notes": "Candidate track; review representative and first/last or conflicting frames",
        })
    by_track.sort(key=lambda r: (-(bool(r["uncertain_frames"] or
                                       r["ir_only_frames"] or r["other_rgb_tracks"])),
                                 -(r["uncertain_frames"] + r["ir_only_frames"]),
                                 -bool(r["other_rgb_tracks"]),
                                 -r["review_match_frames"], -r["frames"],
                                 r["ir_track_id"]))
    with (output / "track_review_queue.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_track[0]))
        writer.writeheader()
        writer.writerows(by_track)
    with (output / "correspondence_review_draft.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(draft[0]))
        writer.writeheader()
        writer.writerows(draft)
    priority_tracks = sum(bool(r["ir_only_frames"] or r["uncertain_frames"] or
                               r["other_rgb_tracks"]) for r in by_track)
    summary = {"ir_candidate_tracks": len(by_track),
               "rgb_candidate_tracks": len(set(rgb_tracks.values())),
               "priority_tracks": priority_tracks,
               "tracks_with_ir_only_or_uncertain": sum(bool(r["ir_only_frames"] or
                                                      r["uncertain_frames"])
                                                for r in by_track),
               "tracks_with_rgb_track_conflict": sum(bool(r["other_rgb_tracks"])
                                                     for r in by_track),
               "warning": "Tracks are geometry-based proposals, not reviewed ground truth"}
    (output / "track_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
