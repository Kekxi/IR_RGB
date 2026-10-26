"""Export a reproducible per-GT-box B0/B1 fog miss review without changing labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401  register custom B1 model classes


def xywh_to_xyxy(x: float, y: float, w: float, h: float):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def read_boxes(path: Path, predictions: bool):
    if not path.exists():
        return []
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        expected = 6 if predictions else 5
        if len(parts) != expected:
            raise ValueError(f"{path}:{line_no}: expected {expected} columns")
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        confidence = float(parts[5]) if predictions else None
        boxes.append((cls, xywh_to_xyxy(x, y, w, h), confidence, (x, y, w, h)))
    return boxes


def assign(gt, pred, score_threshold: float, iou_threshold: float):
    """One-to-one greedy matching, predictions ordered by confidence."""
    candidates = sorted(
        (p for p in pred if p[2] is not None and p[2] >= score_threshold),
        key=lambda p: p[2], reverse=True,
    )
    unused = set(range(len(gt)))
    hit = [False] * len(gt)
    for p in candidates:
        possible = [(iou(p[1], gt[g][1]), g) for g in unused if p[0] == gt[g][0]]
        if not possible:
            continue
        overlap, g = max(possible)
        if overlap >= iou_threshold:
            hit[g] = True
            unused.remove(g)
    return hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Read B0/B1 TXT exports already present in output/b0 and output/b1")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        eligible = {r["id"]: r for r in csv.DictReader(f)
                    if r["split"] == "val" and r["same_image_size"] == "1"}
    with args.conditions.open(encoding="utf-8-sig", newline="") as f:
        conditions = list(csv.DictReader(f))
    if len(eligible) != 877 or len(conditions) != 877 or {r["id"] for r in conditions} != set(eligible):
        raise ValueError("Conditions must match all 877 eligible validation pairs")
    fog_ids = [r["id"] for r in conditions if r["visibility"].strip() == "haze_fog"]
    if not fog_ids:
        raise ValueError("No haze_fog samples")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "fog_images.txt"
    manifest.write_text("\n".join((dataset / eligible[id_]["rgb_image"]).as_posix()
                                  for id_ in fog_ids) + "\n", encoding="utf-8")
    data_yaml = output / "fog.yaml"
    data_yaml.write_text(
        f"path: {json.dumps(dataset.as_posix())}\n"
        f"train: {json.dumps(manifest.as_posix())}\n"
        f"val: {json.dumps(manifest.as_posix())}\n"
        "channels: 6\nnames:\n  0: ship\n", encoding="utf-8")

    for variant, weights in (("b0", args.b0), ("b1", args.b1)):
        pred_dir = output / variant / "labels"
        if args.reuse_predictions:
            if not pred_dir.is_dir():
                raise FileNotFoundError(pred_dir)
            continue
        if pred_dir.exists():
            raise FileExistsError(f"Prediction output already exists: {pred_dir}")
        model = YOLO(str(weights.resolve()))
        model.val(
            validator=RGBIRDetectionValidator, data=str(data_yaml),
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=str(output), name=variant, exist_ok=False,
            conf=0.05, iou=0.7, save_txt=True, save_conf=True, plots=False,
        )

    fieldnames = ["id", "event", "rgb_image", "ir_image", "gt_index", "gt_xywh_norm",
                  "width_px", "height_px", "long_side_at_960", "size_proxy", "center_y_norm",
                  "rgb_gt_count", "ir_gt_count", "b0_hit", "b1_hit", "comparison",
                  "human_distance", "human_visibility", "human_comment"]
    rows = []
    for id_ in fog_ids:
        pair = eligible[id_]
        rgb_path = dataset / pair["rgb_image"]
        ir_path = dataset / pair["ir_image"]
        im = cv2.imread(str(rgb_path))
        if im is None:
            raise FileNotFoundError(rgb_path)
        height, width = im.shape[:2]
        gt = read_boxes(dataset / pair["rgb_label"], predictions=False)
        ir_gt = read_boxes(dataset / pair["ir_label"], predictions=False)
        hits = {}
        for variant in ("b0", "b1"):
            pred = read_boxes(output / variant / "labels" / f"{id_}.txt", predictions=True)
            hits[variant] = assign(gt, pred, args.score, args.match_iou)
        for index, item in enumerate(gt, 1):
            x, y, w, h = item[3]
            long_side_at_960 = max(w * width, h * height) * 960 / max(width, height)
            size_proxy = ("small" if long_side_at_960 < 32 else
                          "medium" if long_side_at_960 < 96 else "large")
            a, b = hits["b0"][index - 1], hits["b1"][index - 1]
            comparison = ("both_hit" if a and b else "b0_only" if a else
                          "b1_only" if b else "both_miss")
            rows.append({
                "id": id_, "event": pair["event"], "rgb_image": str(rgb_path),
                "ir_image": str(ir_path), "gt_index": index,
                "gt_xywh_norm": " ".join(f"{v:.6f}" for v in (x, y, w, h)),
                "width_px": round(w * width, 1), "height_px": round(h * height, 1),
                "long_side_at_960": round(long_side_at_960, 1), "size_proxy": size_proxy,
                "center_y_norm": round(y, 4), "rgb_gt_count": len(gt),
                "ir_gt_count": len(ir_gt), "b0_hit": int(a), "b1_hit": int(b),
                "comparison": comparison, "human_distance": "",
                "human_visibility": "", "human_comment": "",
            })
    with (output / "per_ship_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    priority = {"small": 0, "medium": 1, "large": 2}
    queue = sorted((r for r in rows if r["comparison"] in ("b0_only", "both_miss")),
                   key=lambda r: (priority[r["size_proxy"]], r["comparison"] != "b0_only",
                                  r["event"], r["id"], r["gt_index"]))
    with (output / "manual_queue.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue)
    counts = Counter(r["comparison"] for r in rows)
    by_size = {size: dict(Counter(r["comparison"] for r in rows if r["size_proxy"] == size))
               for size in ("small", "medium", "large")}
    summary = {
        "fog_images": len(fog_ids), "rgb_gt_ships": len(rows), "score_threshold": args.score,
        "match_iou": args.match_iou, "size_proxy": "longer bbox side at 960-pixel long image side; <32, 32-95, >=96",
        "comparison": dict(counts), "by_size": by_size,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Review file:", output / "per_ship_review.csv")
    print("Manual queue:", output / "manual_queue.csv", f"({len(queue)} boxes)")


if __name__ == "__main__":
    main()
