"""Check reviewed weak-RGB targets after masking IR in the trained B1 model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from analyze_fog_misses import assign, iou, read_boxes
from eval_b1_modality_mask_v4 import MaskValidator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True,
                   help="Output of compare_v4_rgb_dual_weak_ir.py")
    p.add_argument("--device", default="0")
    args = p.parse_args()
    dataset, comparison = args.dataset.resolve(), args.comparison.resolve()
    with args.review.open(encoding="gb18030", newline="") as handle:
        review = [r for r in csv.DictReader(handle)
                  if r["human_visibility"] == "rgb_weak_ir_clear"]
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        pairs = {r["id"]: r for r in csv.DictReader(handle)}
    MaskValidator.masked_modality = "ir"
    if not (comparison / "dual_mask_ir" / "labels").is_dir():
        YOLO(str(args.weights.resolve())).val(
            validator=MaskValidator, data=str(comparison / "dual.yaml"),
            imgsz=960, batch=2, device=args.device, project=str(comparison),
            name="dual_mask_ir", exist_ok=False, conf=0.05, iou=0.7,
            save_txt=True, save_conf=True, plots=False)
    with (comparison / "per_target.csv").open(encoding="utf-8-sig", newline="") as handle:
        earlier = list(csv.DictReader(handle))
    if len(earlier) != len(review):
        raise ValueError("Earlier target count differs from review")
    for r, old in zip(review, earlier, strict=True):
        if r["id"] != old["id"]:
            raise ValueError("Earlier target order differs from review")
    hits = {}
    gt_by_id = {}
    for key in sorted({r["id"] for r in review}):
        gt = read_boxes(dataset / pairs[key]["rgb_label"], predictions=False)
        gt_by_id[key] = gt
        pred = read_boxes(comparison / "dual_mask_ir" / "labels" / f"{key}.txt",
                          predictions=True)
        hits[key] = {score: assign(gt, pred, score, 0.5) for score in (0.05, 0.25)}
    rows = []
    for r, old in zip(review, earlier, strict=True):
        gt = gt_by_id[r["id"]]
        x, y, w, h = map(float, r["gt_xywh_norm"].split())
        box = (x-w/2, y-h/2, x+w/2, y+h/2)
        overlaps = [iou(box, g[1]) for g in gt]
        index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1
        row = {"id": r["id"], "distance": r["human_distance"],
               "old_v4_iou": overlaps[index] if index >= 0 else 0}
        if index >= 0 and overlaps[index] >= 0.5:
            for score in (0.05, 0.25):
                row[f"dual_hit_conf{score:.2f}"] = int(old[f"dual_hit_conf{score:.2f}"])
                row[f"dual_mask_ir_hit_conf{score:.2f}"] = int(hits[r["id"]][score][index])
        rows.append(row)
    with (comparison / "ir_dependence_per_target.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["old_v4_iou"] >= 0.5]
    result = {}
    for distance in ("all", "far", "middle"):
        subset = valid if distance == "all" else [r for r in valid if r["distance"] == distance]
        for score in (0.05, 0.25):
            counts = Counter((r[f"dual_hit_conf{score:.2f}"],
                              r[f"dual_mask_ir_hit_conf{score:.2f}"]) for r in subset)
            result[f"{distance}_conf{score:.2f}"] = {
                "n": len(subset), "both_hit": counts[(1, 1)],
                "normal_only": counts[(1, 0)], "masked_only": counts[(0, 1)],
                "both_miss": counts[(0, 0)]}
    (comparison / "ir_dependence_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
