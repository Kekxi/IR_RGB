"""Compare RGB-only and B1 dual detection on reviewed RGB-weak/IR-clear targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from analyze_fog_misses import assign, iou, read_boxes
from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--rgb", type=Path, required=True)
    p.add_argument("--dual", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.review.open(encoding="gb18030", newline="") as handle:
        review = [r for r in csv.DictReader(handle)
                  if r["human_visibility"] == "rgb_weak_ir_clear"]
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        pairs = {r["id"]: r for r in csv.DictReader(handle)}
    ids = sorted({r["id"] for r in review})
    manifest = output / "review_images.txt"
    manifest.write_text("\n".join((dataset / pairs[key]["rgb_image"]).as_posix()
                                  for key in ids) + "\n", encoding="utf-8")
    for name, weights in (("rgb", args.rgb), ("dual", args.dual)):
        yaml = output / f"{name}.yaml"
        yaml.write_text(f"path: {json.dumps(dataset.as_posix())}\n"
                        f"train: {json.dumps(manifest.as_posix())}\n"
                        f"val: {json.dumps(manifest.as_posix())}\n"
                        f"channels: {6 if name == 'dual' else 3}\n"
                        "names:\n  0: ship\n", encoding="utf-8")
        kwargs = {"validator": RGBIRDetectionValidator} if name == "dual" else {}
        YOLO(str(weights.resolve())).val(data=str(yaml), imgsz=960, batch=2,
                                         device=args.device, project=str(output),
                                         name=name, exist_ok=False, conf=0.05,
                                         iou=0.7, save_txt=True, save_conf=True,
                                         plots=False, **kwargs)
    hits = {name: {} for name in ("rgb", "dual")}
    gt_by_id = {}
    for key in ids:
        gt = read_boxes(dataset / pairs[key]["rgb_label"], predictions=False)
        gt_by_id[key] = gt
        for name in hits:
            pred = read_boxes(output / name / "labels" / f"{key}.txt", predictions=True)
            hits[name][key] = {score: assign(gt, pred, score, 0.5)
                               for score in (0.05, 0.25)}
    rows = []
    for r in review:
        gt = gt_by_id[r["id"]]
        x, y, w, h = map(float, r["gt_xywh_norm"].split())
        old = (x - w/2, y - h/2, x + w/2, y + h/2)
        overlaps = [iou(old, box[1]) for box in gt]
        index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1
        row = {"id": r["id"], "event": r["event"], "distance": r["human_distance"],
               "old_v4_iou": overlaps[index] if index >= 0 else 0}
        if index >= 0 and overlaps[index] >= 0.5:
            for score in (0.05, 0.25):
                for name in hits:
                    row[f"{name}_hit_conf{score:.2f}"] = int(hits[name][r["id"]][score][index])
        rows.append(row)
    with (output / "per_target.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["old_v4_iou"] >= 0.5]
    summary = {"reviewed": len(rows), "matched_v4": len(valid),
               "unmatched_v4": len(rows) - len(valid)}
    for distance in ("all", "far", "middle"):
        subset = valid if distance == "all" else [r for r in valid if r["distance"] == distance]
        for score in (0.05, 0.25):
            counts = Counter((r[f"rgb_hit_conf{score:.2f}"],
                              r[f"dual_hit_conf{score:.2f}"]) for r in subset)
            summary[f"{distance}_conf{score:.2f}"] = {
                "n": len(subset), "both_hit": counts[(1, 1)],
                "rgb_only": counts[(1, 0)], "dual_only": counts[(0, 1)],
                "both_miss": counts[(0, 0)]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
