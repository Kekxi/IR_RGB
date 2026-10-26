"""Export B2 fog-small misses and ten-pair, event-wise contact sheets.

This is a read-only diagnostic. It never changes source images or labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "fog_ship_code_20261019"
sys.path.insert(0, str(CODE))
from rgb_ir_midfusion import RGBIRDetectionValidator  # noqa: E402
import fog_reliability  # noqa: E402,F401
import b2_variants  # noqa: E402,F401
from analyze_fog_misses import assign, read_boxes  # noqa: E402
from make_fog_miss_contact_sheets import (  # noqa: E402
    FONT_PANEL, FONT_TITLE, HEADER_H, MARGIN, PAGE_W, ROW_H, draw_row,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--conditions", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--small", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.50)
    ap.add_argument("--reuse-predictions", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        pairs = {r["id"]: r for r in csv.DictReader(f)
                 if r["split"] == "val" and r["same_image_size"] == "1"}
    with args.conditions.open(encoding="utf-8-sig", newline="") as f:
        conditions = list(csv.DictReader(f))
    if len(pairs) != 877 or len(conditions) != 877 or {r["id"] for r in conditions} != set(pairs):
        raise ValueError("Expected exactly the 877 eligible v3 validation pairs and conditions")
    fog_ids = [r["id"] for r in conditions if r["visibility"].strip() == "haze_fog"]
    if len(fog_ids) != 633:
        raise ValueError(f"Expected 633 fog pairs; found {len(fog_ids)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "fog_images.txt"
    manifest.write_text("\n".join((dataset / pairs[id_]["rgb_image"]).as_posix()
                                  for id_ in fog_ids) + "\n", encoding="utf-8")
    yaml = output / "fog.yaml"
    yaml.write_text(f"path: {json.dumps(dataset.as_posix())}\n"
                    f"train: {json.dumps(manifest.as_posix())}\n"
                    f"val: {json.dumps(manifest.as_posix())}\n"
                    "channels: 6\nnames:\n  0: ship\n", encoding="utf-8")

    for name, weights in (("control", args.control), ("small", args.small)):
        label_dir = output / name / "labels"
        if args.reuse_predictions:
            if not label_dir.is_dir():
                raise FileNotFoundError(label_dir)
            continue
        if label_dir.exists():
            raise FileExistsError(f"Existing predictions: {label_dir}; use --reuse-predictions")
        model = YOLO(str(weights.resolve()))
        model.val(validator=RGBIRDetectionValidator, data=str(yaml),
                  imgsz=args.imgsz, batch=args.batch, device=args.device,
                  project=str(output), name=name, exist_ok=False,
                  conf=args.conf, iou=0.7, save_txt=True, save_conf=True, plots=False)

    fields = ["id", "event", "rgb_image", "ir_image", "gt_index", "gt_xywh_norm",
              "long_side_at_960", "control_hit", "small_hit", "comparison",
              "human_distance", "human_visibility", "human_cause", "human_comment"]
    all_rows = []
    for id_ in fog_ids:
        pair = pairs[id_]
        rgb_path, ir_path = dataset / pair["rgb_image"], dataset / pair["ir_image"]
        image = cv2.imread(str(rgb_path))
        if image is None:
            raise FileNotFoundError(rgb_path)
        height, width = image.shape[:2]
        gt = read_boxes(dataset / pair["rgb_label"], predictions=False)
        hits = {}
        for name in ("control", "small"):
            pred = read_boxes(output / name / "labels" / f"{id_}.txt", predictions=True)
            hits[name] = assign(gt, pred, args.conf, args.match_iou)
        for i, item in enumerate(gt):
            x, y, w, h = item[3]
            # Match the validator's float32 target coordinates at the 48 px boundary.
            scale = args.imgsz / max(width, height)
            long_side = max(float(np.float32(w) * np.float32(width * scale)),
                            float(np.float32(h) * np.float32(height * scale)))
            if long_side > 48:
                continue
            control_hit, small_hit = hits["control"][i], hits["small"][i]
            comparison = ("both_hit" if control_hit and small_hit else
                          "control_only" if control_hit else
                          "small_only" if small_hit else "both_miss")
            all_rows.append({
                "id": id_, "event": pair["event"],
                "rgb_image": str(rgb_path), "ir_image": str(ir_path),
                "gt_index": i + 1,
                "gt_xywh_norm": " ".join(f"{v:.6f}" for v in (x, y, w, h)),
                "long_side_at_960": round(long_side, 2),
                "control_hit": int(control_hit), "small_hit": int(small_hit),
                "comparison": comparison, "human_distance": "",
                "human_visibility": "", "human_cause": "", "human_comment": "",
            })
    for name in ("control", "small"):
        hit_count = sum(int(r[f"{name}_hit"]) for r in all_rows)
        print(f"{name}: {hit_count}/{len(all_rows)} fog-small GT boxes matched")
    if len(all_rows) != 330:
        raise ValueError(f"Expected 330 fog-small GT boxes; got {len(all_rows)}; check size convention")
    with (output / "all_fog_small_boxes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(all_rows)

    priority = {"small_only": 0, "control_only": 1, "both_miss": 2}
    queue = sorted((r for r in all_rows if r["comparison"] != "both_hit"),
                   key=lambda r: (r["event"], priority[r["comparison"]],
                                  r["id"], int(r["gt_index"])))
    with (output / "manual_queue_reviewed.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(queue)
    counts = dict(Counter(r["comparison"] for r in all_rows))
    (output / "summary.json").write_text(json.dumps({
        "fog_pairs": len(fog_ids), "fog_small_gt": len(all_rows),
        "imgsz": args.imgsz, "conf": args.conf, "match_iou": args.match_iou,
        "comparison": counts, "review_rows": len(queue),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    sheet_dir = output / "contact_sheets_10"
    sheet_dir.mkdir(exist_ok=True)
    by_event = defaultdict(list)
    for row_no, row in enumerate(queue, 1):
        by_event[row["event"]].append((row_no, row))
    links = ["# 雾天小船漏检逐船复核", "",
             "每页最多 10 条待审核船框，按采集事件分页。左侧红框是 RGB 真值框；"
             "IR 不投影 RGB 框，避免弱配准造成误导。行号对应 manual_queue_reviewed.csv。", ""]
    for event_no, (event, items) in enumerate(sorted(by_event.items()), 1):
        links += [f"## event_{event_no:02d} — {event}（{len(items)} 条）", ""]
        pages = math.ceil(len(items) / 10)
        for page in range(pages):
            subset = items[page * 10:(page + 1) * 10]
            canvas = Image.new("RGB", (PAGE_W, HEADER_H + len(subset) * ROW_H + MARGIN), "#eef2f7")
            draw = ImageDraw.Draw(canvas)
            draw.text((MARGIN, 19), f"event_{event_no:02d}  |  page {page + 1:02d}/{pages:02d}",
                      fill="#122033", font=FONT_TITLE)
            draw.text((MARGIN, 55), "Red = RGB GT; IR has no projected RGB box."
                      "  small_only = control missed; control_only = small missed.",
                      fill="#475569", font=FONT_PANEL)
            for offset, (row_no, row) in enumerate(subset):
                display = dict(row, size_proxy=f"{row['long_side_at_960']}px")
                draw_row(canvas, display, row_no, HEADER_H + offset * ROW_H)
            filename = f"event_{event_no:02d}_page_{page + 1:02d}.jpg"
            canvas.save(sheet_dir / filename, format="JPEG", quality=88, subsampling=0, optimize=True)
            links.append(f"- [第 {page + 1} 页：CSV 行 {subset[0][0]}–{subset[-1][0]}]({filename})")
        links.append("")
    (sheet_dir / "index.md").write_text("\n".join(links) + "\n", encoding="utf-8")
    print("comparison:", counts)
    print("review queue:", output / "manual_queue_reviewed.csv")
    print("contact sheet index:", sheet_dir / "index.md")


if __name__ == "__main__":
    main()
