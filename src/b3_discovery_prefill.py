"""Create a conservative correspondence *draft* for human review.

The draft uses GT box geometry only. It does not establish visual identity or
RGB visibility, and it never edits the original correspondence_review.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def overlap(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def metric(ir: list[float], rgb: list[float]) -> tuple[float, float, float]:
    ic, rc = center(ir), center(rgb)
    dx, dy = abs(ic[0] - rc[0]), abs(ic[1] - rc[1])
    distance = math.hypot(dx, dy)
    ir_w, ir_h = ir[2] - ir[0], ir[3] - ir[1]
    rgb_w, rgb_h = rgb[2] - rgb[0], rgb[3] - rgb[1]
    size_penalty = 0.10 * (abs(math.log(ir_w / rgb_w)) +
                           abs(math.log(ir_h / rgb_h)))
    iou = overlap(ir, rgb)
    cost = math.hypot(dx / 0.075, dy / 0.065) + size_penalty - 0.30 * iou
    return cost, distance, iou


def propose(pair: dict) -> dict[int, dict]:
    ir_boxes, rgb_boxes = pair["gt_ir"], pair["gt_rgb"]
    ni, nr = len(ir_boxes), len(rgb_boxes)
    if not ni:
        return {}
    distances = np.zeros((ni, nr), dtype=float)
    costs = np.full((ni, nr), 100.0, dtype=float)
    ious = np.zeros((ni, nr), dtype=float)
    for i, ir in enumerate(ir_boxes):
        for j, rgb in enumerate(rgb_boxes):
            cost, distance, iou = metric(ir, rgb)
            distances[i, j], ious[i, j] = distance, iou
            if distance <= 0.115 or iou >= 0.03:
                costs[i, j] = cost
    # Private dummy columns allow each IR GT to remain unmatched.
    matrix = np.full((ni, nr + ni), 100.0, dtype=float)
    matrix[:, :nr] = costs
    for i in range(ni):
        matrix[i, nr + i] = 1.35
    assigned = {int(i): int(j) for i, j in zip(*linear_sum_assignment(matrix))}
    out = {}
    for i, ir in enumerate(ir_boxes):
        j = assigned[i]
        nearest_j = int(np.argmin(costs[i])) if nr else None
        nearest_distance = float(np.min(distances[i])) if nr else float("inf")
        near_candidate = nearest_j is not None and nearest_distance < 0.115
        if j < nr and costs[i, j] < 1.35:
            cost, distance, iou = costs[i, j], distances[i, j], ious[i, j]
            other_ir = min((costs[k, j] for k in range(ni) if k != i), default=100.0)
            other_rgb = min((costs[i, k] for k in range(nr) if k != j), default=100.0)
            nearest_competitor = min(other_ir, other_rgb)
            margin = (None if nearest_competitor >= 100.0 else
                      nearest_competitor - cost)
            if distance <= 0.055 and cost <= 0.88 and (margin is None or margin >= 0.25):
                out[i] = {"rgb": j, "status": "shared_rgb_uncertain",
                          "priority": "spot_check", "method": "unique_close_match",
                          "distance": distance, "cost": cost, "margin": margin,
                          "iou": iou}
            elif distance <= 0.08 and cost <= 1.12 and (margin is None or margin >= 0.14):
                out[i] = {"rgb": j, "status": "shared_rgb_uncertain",
                          "priority": "review_match", "method": "moderate_match",
                          "distance": distance, "cost": cost, "margin": margin,
                          "iou": iou}
            else:
                out[i] = {"rgb": None, "status": "uncertain",
                          "priority": "review_match", "method": "ambiguous_match",
                          "candidate": j, "distance": distance, "cost": cost,
                          "margin": margin, "iou": iou}
        elif not near_candidate and 0.08 <= center(ir)[0] <= 0.92:
            out[i] = {"rgb": None, "status": "ir_only_common_fov",
                      "priority": "review_ir_only", "method": "unpaired_interior",
                      "distance": nearest_distance, "cost": None,
                      "margin": None, "iou": None}
        else:
            out[i] = {"rgb": None, "status": "uncertain",
                      "priority": "review_ir_only", "method": "unpaired_edge_or_close",
                      "candidate": nearest_j, "distance": nearest_distance,
                      "cost": None, "margin": None, "iou": None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared, output = args.prepared.resolve(), args.output.resolve()
    if prepared == output:
        raise ValueError("Output must be separate from the original prepared folder")
    output.mkdir(parents=True, exist_ok=True)
    pairs = {p["id"]: p for p in (json.loads(line) for line in
             (prepared / "pairs.jsonl").read_text(encoding="utf-8").splitlines())}
    with (prepared / "correspondence_review.csv").open(
            "r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    proposals = {pair_id: propose(pair) for pair_id, pair in pairs.items()}
    draft = []
    for raw in raw_rows:
        row = dict(raw)
        pair_id, ir_index = row["id"], int(row["ir_gt_index"])
        p = proposals[pair_id][ir_index]
        # Never clobber a human decision if the source file is updated later.
        if row["status"].strip() or row["rgb_gt_index"].strip():
            raise ValueError(f"Source row already reviewed: {(pair_id, ir_index)}")
        row["rgb_gt_index"] = "" if p["rgb"] is None else str(p["rgb"])
        row["status"] = p["status"]
        row["notes"] = ("AUTO_DRAFT_UNREVIEWED; geometry_only; "
                        f"method={p['method']}; RGB visibility needs visual review")
        row["review_priority"] = p["priority"]
        row["auto_distance"] = ("" if not math.isfinite(p["distance"])
                                else f"{p['distance']:.4f}")
        row["auto_cost"] = ("" if p["cost"] is None else f"{p['cost']:.3f}")
        row["auto_margin"] = ("" if p["margin"] is None else f"{p['margin']:.3f}")
        row["auto_iou"] = ("" if p["iou"] is None else f"{p['iou']:.3f}")
        row["auto_candidate_rgb_index"] = ("" if p.get("candidate") is None
                                           else str(p["candidate"]))
        row["human_checked"] = ""
        row["review_image"] = str((prepared / "review_images" /
                                   f"{pair_id}.jpg").resolve())
        draft.append(row)
    fields = list(draft[0])
    with (output / "correspondence_review_draft.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(draft)
    counts = Counter(row["review_priority"] for row in draft)
    status_counts = Counter(row["status"] for row in draft)
    summary = {"rows": len(draft), "priority": dict(counts),
               "status": dict(status_counts),
               "human_reviewed": 0,
               "warning": "AUTO_DRAFT_UNREVIEWED is not a validated GT correspondence"}
    (output / "prefill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
