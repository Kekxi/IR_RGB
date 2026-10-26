"""Compare B2 control/align on human-reviewed RGB-weak, IR-clear targets."""

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
import b2_variants  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--align", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.review.open(encoding="gb18030", newline="") as handle:
        review = [r for r in csv.DictReader(handle)
                  if r["human_visibility"] == "rgb_weak_ir_clear"]
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        pairs = {r["id"]: r for r in csv.DictReader(handle)}
    ids = sorted({r["id"] for r in review})
    manifest = output / "review_images.txt"
    manifest.write_text("\n".join((dataset / pairs[k]["rgb_image"]).as_posix()
                                  for k in ids) + "\n", encoding="utf-8")
    data = output / "review.yaml"
    data.write_text(f"path: {json.dumps(dataset.as_posix())}\n"
                    f"train: {json.dumps(manifest.as_posix())}\n"
                    f"val: {json.dumps(manifest.as_posix())}\n"
                    "channels: 6\nnames:\n  0: ship\n", encoding="utf-8")

    for name, weights in (("control", args.control), ("align", args.align)):
        YOLO(str(weights.resolve())).val(
            validator=RGBIRDetectionValidator, data=str(data), imgsz=960,
            batch=2, device=args.device, project=str(output), name=name,
            exist_ok=True, conf=0.05, iou=0.7, save_txt=True,
            save_conf=True, plots=False)

    hits = {name: {} for name in ("control", "align")}
    gt_by_id = {}
    for key in ids:
        gt = read_boxes(dataset / pairs[key]["rgb_label"], predictions=False)
        gt_by_id[key] = gt
        for name in hits:
            pred = read_boxes(output / name / "labels" / f"{key}.txt", predictions=True)
            hits[name][key] = {threshold: assign(gt, pred, threshold, 0.5)
                               for threshold in (0.05, 0.25)}

    result = []
    for r in review:
        gt = gt_by_id[r["id"]]
        old = tuple(map(float, r["gt_xywh_norm"].split()))
        old_xyxy = (old[0] - old[2] / 2, old[1] - old[3] / 2,
                    old[0] + old[2] / 2, old[1] + old[3] / 2)
        overlaps = [iou(old_xyxy, box[1]) for box in gt]
        index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1
        row = {"id": r["id"], "event": r["event"], "distance": r["human_distance"],
               "old_gt_index": r["gt_index"], "v4_gt_index": index + 1,
               "old_v4_iou": overlaps[index] if index >= 0 else 0}
        if index >= 0 and overlaps[index] >= 0.5:
            for threshold in (0.05, 0.25):
                tag = f"hit_conf{threshold:.2f}"
                for name in hits:
                    row[f"{name}_{tag}"] = int(hits[name][r["id"]][threshold][index])
        result.append(row)
    columns = list(result[0])
    with (output / "per_target.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result)
    valid = [r for r in result if r["old_v4_iou"] >= 0.5]
    summary = {"reviewed_targets": len(result), "reviewed_pairs": len(ids),
               "matched_to_v4_gt": len(valid), "unmatched_to_v4_gt": len(result) - len(valid)}
    for distance in ("all", "far", "middle"):
        subset = valid if distance == "all" else [r for r in valid if r["distance"] == distance]
        for threshold in (0.05, 0.25):
            tag = f"hit_conf{threshold:.2f}"
            counter = Counter((r[f"control_{tag}"], r[f"align_{tag}"]) for r in subset)
            summary[f"{distance}_conf{threshold:.2f}"] = {
                "n": len(subset), "both_hit": counter[(1, 1)],
                "control_only": counter[(1, 0)], "align_only": counter[(0, 1)],
                "both_miss": counter[(0, 0)]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
