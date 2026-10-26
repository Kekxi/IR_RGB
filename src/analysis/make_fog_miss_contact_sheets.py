"""Make ten-pair RGB/IR contact sheets from the manual fog-miss queue."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PANEL_H = 405
RGB_W = 720
IR_W = 720
ZOOM_W = 360
GAP = 18
MARGIN = 24
HEADER_H = 88
ROW_H = 473
PAGE_W = MARGIN * 2 + RGB_W + IR_W + ZOOM_W + GAP * 2


def font(size: int):
    for path in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(26)
FONT_META = font(19)
FONT_PANEL = font(17)


def box_xyxy(row: dict[str, str], width: int, height: int):
    x, y, w, h = map(float, row["gt_xywh_norm"].split())
    return ((x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height)


def cover(image: Image.Image, width: int, height: int):
    return ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)


def zoom_region(image: Image.Image, box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    crop_w = max(200, min(650, bw * 4.0))
    crop_h = max(150, min(490, bh * 4.0))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, min(image.width - crop_w, cx - crop_w / 2))
    top = max(0, min(image.height - crop_h, cy - crop_h / 2))
    right = min(image.width, left + crop_w)
    bottom = min(image.height, top + crop_h)
    patch = image.crop((int(left), int(top), int(right), int(bottom)))
    drawn = ImageDraw.Draw(patch)
    drawn.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline="#ff3131", width=4)
    return cover(patch, ZOOM_W, PANEL_H)


def draw_row(canvas: Image.Image, row: dict[str, str], row_no: int, y: int):
    rgb_path, ir_path = Path(row["rgb_image"]), Path(row["ir_image"])
    if not rgb_path.is_file() or not ir_path.is_file():
        raise FileNotFoundError(f"Missing pair for CSV row {row_no}: {rgb_path} / {ir_path}")
    with Image.open(rgb_path) as f:
        rgb = f.convert("RGB")
    with Image.open(ir_path) as f:
        ir = f.convert("RGB")
    box = box_xyxy(row, *rgb.size)
    rgb_marked = rgb.copy()
    ImageDraw.Draw(rgb_marked).rectangle(box, outline="#ff3131", width=max(4, rgb.width // 250))
    rgb_view = cover(rgb_marked, RGB_W, PANEL_H)
    ir_view = cover(ir, IR_W, PANEL_H)
    zoom_view = zoom_region(rgb, box)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((MARGIN - 8, y - 4, PAGE_W - MARGIN + 8, y + ROW_H - 8),
                           radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
    title = (f"#{row_no:03d}  CSV line {row_no + 1}   {row['id']}   "
             f"GT {row['gt_index']}   {row['comparison']}   {row['size_proxy']}")
    draw.text((MARGIN, y + 8), title, fill="#17212f", font=FONT_META)
    for panel, x, width, label in (
        (rgb_view, MARGIN, RGB_W, "RGB  |  red = selected GT box"),
        (ir_view, MARGIN + RGB_W + GAP, IR_W, "IR  |  no box projection"),
        (zoom_view, MARGIN + RGB_W + IR_W + GAP * 2, ZOOM_W, "RGB zoom of selected box"),
    ):
        px = x + (width - panel.width) // 2
        py = y + 39 + (PANEL_H - panel.height) // 2
        canvas.paste(panel, (px, py))
        draw.text((x + 2, y + 447), label, fill="#475569", font=FONT_PANEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-page", type=int, default=10)
    args = ap.parse_args()
    if args.per_page < 1:
        raise ValueError("--per-page must be positive")
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty review CSV")
    args.output.mkdir(parents=True, exist_ok=True)
    total_pages = math.ceil(len(rows) / args.per_page)
    links = ["# Fog miss contact sheets", "",
             "Each page contains 10 CSV rows (the final page may contain fewer). "
             "A red box marks the selected RGB ground-truth box. "
             "IR is shown without a projected box because the modalities are weakly aligned.", ""]
    for page in range(total_pages):
        start = page * args.per_page
        subset = rows[start:start + args.per_page]
        page_h = HEADER_H + len(subset) * ROW_H + MARGIN
        canvas = Image.new("RGB", (PAGE_W, page_h), "#eef2f7")
        draw = ImageDraw.Draw(canvas)
        title = f"Fog ship review  |  page {page + 1:02d}/{total_pages:02d}  |  CSV rows {start + 1}-{start + len(subset)}"
        draw.text((MARGIN, 19), title, fill="#122033", font=FONT_TITLE)
        draw.text((MARGIN, 55), "B0-only = B0 hit / B1 miss; both-miss = neither hit; GT box from RGB label.",
                  fill="#475569", font=FONT_PANEL)
        for offset, row in enumerate(subset):
            draw_row(canvas, row, start + offset + 1, HEADER_H + offset * ROW_H)
        name = f"fog_miss_page_{page + 1:02d}.jpg"
        canvas.save(args.output / name, format="JPEG", quality=88, subsampling=0, optimize=True)
        links.append(f"- [Page {page + 1:02d}: rows {start + 1}-{start + len(subset)}]({name})")
        print(name, canvas.size)
    (args.output / "index.md").write_text("\n".join(links) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
