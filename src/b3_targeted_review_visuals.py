"""Render compact prediction comparisons for the most consequential B3 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from b3_discovery_score import _jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    root, review = args.full_root, args.review_root
    with (review / "disagreement_frames.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (review / "candidate_ship_groups.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        groups = list(csv.DictReader(handle))
    selected = [row for row in rows if row["correspondence_changes_outcome"] != "no"]
    for group in groups:
        for row in rows:
            if row["pair_id"] == group["representative_frame"] and row not in selected:
                selected.append(row)
                break
    # Stable de-duplication preserves the two priority categories above.
    selected = list({(row["pair_id"], row["target"]): row for row in selected}.values())
    pairs = {row["id"]: row for row in _jsonl(root / "prepared" / "pairs.jsonl")}
    exports = {variant: {row["id"]: row for row in _jsonl(root / "exports" / variant / "predictions.jsonl")}
               for variant in ("rgb_only", "ir_only", "b3_rgb", "b3_ir")}
    out = review / "visuals"
    out.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for row in selected:
        pair_id = row["pair_id"]
        base = Image.open(row["review_image"]).convert("RGB")
        pane_w = base.width // 2
        pane_h = base.height
        canvas = Image.new("RGB", (base.width, 2 * (pane_h + 32)), "#161a22")
        for y in (32, pane_h + 64):
            canvas.paste(base, (0, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "Independent: RGB-only (left) | IR-only (right) / green predictions", fill="white", font=font)
        draw.text((8, pane_h + 38), "B3: fusion RGB (left) | native IR (right) / magenta predictions", fill="white", font=font)
        for variant, y_shift, color, x_shift in (
            ("rgb_only", 32, "#38ff70", 0), ("ir_only", 32, "#38ff70", pane_w),
            ("b3_rgb", pane_h + 64, "#ff50df", 0), ("b3_ir", pane_h + 64, "#ff50df", pane_w),
        ):
            for box in exports[variant][pair_id]["boxes"]:
                if box["conf"] < args.conf:
                    continue
                x1, y1, x2, y2 = box["xyxy"]
                coords = (x_shift + x1 * pane_w, y_shift + y1 * pane_h,
                          x_shift + x2 * pane_w, y_shift + y2 * pane_h)
                draw.rectangle(coords, outline=color, width=2)
                draw.text((coords[0], max(y_shift, coords[1] - 14)), f"{box['conf']:.2f}", fill=color, font=font)
        # Yellow is the GT box under review, already labeled I# or R# on base.
        modality = "ir" if row["target"].startswith("I") else "rgb"
        index = int(row["target"][1:])
        x1, y1, x2, y2 = pairs[pair_id][f"gt_{modality}"][index]
        x_shift = pane_w if modality == "ir" else 0
        for y_shift in (32, pane_h + 64):
            draw.rectangle((x_shift + x1 * pane_w, y_shift + y1 * pane_h,
                            x_shift + x2 * pane_w, y_shift + y2 * pane_h),
                           outline="#ffe930", width=4)
        canvas.save(out / f"{pair_id}_{row['target']}.png")
    (review / "visuals_summary.json").write_text(
        json.dumps({"selected_targets": len(selected), "images": len(list(out.glob("*.png"))),
                    "rule": "all correspondence-sensitive rows plus one representative of every candidate ship group",
                    "legend": "yellow target GT; green independent predictions; magenta B3 predictions; conf>=0.25"},
                   indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(selected)} target comparisons in {out}")


if __name__ == "__main__":
    main()
