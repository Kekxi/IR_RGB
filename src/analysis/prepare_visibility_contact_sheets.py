"""Make read-only RGB/IR validation contact sheets for human visibility review."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int):
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        result = Image.new("RGB", size, "#26313a")
        result.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--page-size", type=int, default=24)
    args = p.parse_args()
    if args.page_size < 2 or args.page_size % 2:
        raise ValueError("--page-size must be a positive even number")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing review folder: {args.output}")

    with (args.dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        pairs = {row["id"]: row for row in csv.DictReader(handle)
                 if row["split"] == "val" and row["same_image_size"] == "1"}
    with args.conditions.open(encoding="utf-8-sig", newline="") as handle:
        conditions = list(csv.DictReader(handle))
    ids = [row["id"] for row in conditions]
    if len(ids) != 877 or len(set(ids)) != 877 or set(ids) != set(pairs):
        raise ValueError("Conditions must contain exactly the 877 eligible validation IDs")

    by_event = defaultdict(list)
    for item in conditions:
        by_event[pairs[item["id"]]["event"]].append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    small = font(17)
    large = font(22)
    counts = Counter((row["visibility"] or "").strip().lower() for row in conditions)
    index = ["# 验证集 RGB/IR 能见度复核总览", "",
             "这些图只用于定位和初筛。看不清天气或船只时，打开两路原图确认。",
             "不在总览图上修改检测框；能见度分类写在另存的 val_conditions.csv 副本中。", "",
             f"共 {len(conditions)} 对，当前未分类 {counts['']} 对。", ""]
    for event_no, (event, rows) in enumerate(sorted(by_event.items()), 1):
        rows.sort(key=lambda row: row["id"])
        index.extend([f"## 事件 {event_no:02d}：{event}（{len(rows)} 对）", ""])
        for start in range(0, len(rows), args.page_size):
            page = rows[start:start + args.page_size]
            page_no = start // args.page_size + 1
            filename = f"event_{event_no:02d}_page_{page_no:02d}.jpg"
            tile_w, tile_h, top = 540, 183, 70
            canvas = Image.new("RGB", (tile_w * 2, top + tile_h * ((len(page) + 1) // 2)), "#f5f7f9")
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 18), f"Event {event_no:02d} / page {page_no:02d} / {event}", fill="#14283c", font=large)
            for i, item in enumerate(page):
                pair = pairs[item["id"]]
                col, row = i % 2, i // 2
                x, y = col * tile_w + 12, top + row * tile_h
                state = (item["visibility"] or "UNREVIEWED").strip() or "UNREVIEWED"
                draw.text((x, y), f"{item['id']}   {state}", fill="#1b3145", font=small)
                for j, key in enumerate(("rgb_image", "ir_image")):
                    source = args.dataset / pair[key]
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    thumb = thumbnail(source, (250, 140))
                    canvas.paste(thumb, (x + j * 258, y + 25))
                draw.text((x + 5, y + 166), "RGB", fill="#47647d", font=small)
                draw.text((x + 263, y + 166), "IR", fill="#47647d", font=small)
            canvas.save(args.output / filename, quality=86, subsampling=1)
            index.append(f"- [{filename}](<{(args.output / filename).resolve().as_posix()}>)："
                         f"{page[0]['id']} 至 {page[-1]['id']}（{len(page)} 对）")
        index.append("")
    (args.output / "index.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Saved {len(list(args.output.glob('event_*_page_*.jpg')))} pages to {args.output}")


if __name__ == "__main__":
    main()
