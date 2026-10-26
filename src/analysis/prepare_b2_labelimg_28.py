"""Prepare isolated, small LabelImg batches for B2 fog-small label review."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent / "fog_small_miss_b2_s42_v3"
DATASET = Path(r"E:\HEU_4151\RGB_IR\MTDSP_v3_labelreview_20261020\dataset")
SOURCE = BASE / "label_quality_followup.csv"
OUT = BASE / "LabelImg_28_review"
LABEL_DIR = DATASET / "labels" / "rgb" / "val"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Review directory exists; refusing to replace it: {OUT}")
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 28 or any(r["label_review_decision"].strip() for r in rows):
        raise ValueError("Expected 28 unreviewed rows in label_quality_followup.csv")
    by_event = defaultdict(list)
    for row in rows:
        by_event[row["event"]].append(row)
    # At most nine images per LabelImg window, to avoid the earlier next-image crash.
    batches = []
    for event, group in sorted(by_event.items()):
        batches.extend((event, group[i:i + 9]) for i in range(0, len(group), 9))
    if [len(x[1]) for x in batches] != [7, 9, 9, 3]:
        raise ValueError("Unexpected label-review event distribution")

    manifest = []
    launch = []
    for batch_no, (event, group) in enumerate(batches, 1):
        name = f"batch_{batch_no:02d}"
        folder = OUT / name
        image_dir = folder / "rgb_images"
        label_dir = folder / "rgb_labels"
        ir_dir = folder / "ir_reference"
        for path in (image_dir, label_dir, ir_dir):
            path.mkdir(parents=True, exist_ok=False)
        (label_dir / "classes.txt").write_text("ship\n", encoding="utf-8")
        for row in group:
            id_ = row["id"]
            rgb = Path(row["rgb_image"])
            ir = Path(row["ir_image"])
            source_label = LABEL_DIR / f"{id_}.txt"
            if not all(p.is_file() for p in (rgb, ir, source_label)):
                raise FileNotFoundError(f"Missing pair or label for {id_}")
            shutil.copy2(rgb, image_dir / rgb.name)
            shutil.copy2(ir, ir_dir / ir.name)
            target_label = label_dir / source_label.name
            shutil.copyfile(source_label, target_label)
            target_label.chmod(0o666)
            if sha256(target_label) != sha256(source_label):
                raise AssertionError(f"Copied label hash mismatch: {id_}")
            manifest.append({
                "batch": name, "source_csv_line": row["source_csv_line"],
                "id": id_, "event": event, "gt_index": row["gt_index"],
                "gt_xywh_norm": row["gt_xywh_norm"],
                "human_cause": row["human_cause"],
                "human_comment": row["human_comment"],
                "source_label": str(source_label), "review_label": str(target_label),
                "source_sha256": sha256(source_label),
            })
        exe = Path(r"D:\anaconda3\envs\labelimg\Scripts\labelImg.exe")
        if not exe.is_file():
            raise FileNotFoundError(exe)
        script = folder / "open_labelimg.ps1"
        script.write_text(
            f"Start-Process -FilePath '{exe}' "
            f"-ArgumentList @('{image_dir}','{label_dir / 'classes.txt'}','{label_dir}') "
            f"-WorkingDirectory '{folder}' -WindowStyle Normal\n",
            encoding="utf-8-sig",
        )
        launch.append((name, event, len(group), script))

    with (OUT / "review_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with (OUT / "review_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    lines = ["# 28 张 RGB 图的 LabelImg 独立复核副本", "",
             "每个批次只包含 3–9 张图，避免一次翻阅大量图像造成闪退。"
             "LabelImg 保存到 `batch_XX/rgb_labels`，不会修改 MTDSP v3 数据集。"
             "`ir_reference` 仅供查看，不把 RGB 框投影到 IR。", "",
             "先确认 LabelImg 右侧已有船框且格式为 YOLO，再按 D 翻下一张。"
             "不要在空白框状态下保存。只在确认类别/坐标确有问题时修改并保存。", "",
             "每张图对应的原 CSV 行号、GT 序号和坐标见 `review_manifest.csv`。"
             "LabelImg 框列表显示顺序可能不同，应结合坐标定位目标。", "",
             "| 批次 | 事件 | 图数 | 启动脚本 |", "| --- | --- | ---: | --- |"]
    for name, event, count, script in launch:
        lines.append(f"| {name} | {event} | {count} | [{name}/open_labelimg.ps1]({name}/open_labelimg.ps1) |")
    lines += ["", "核对后在上级 `label_quality_followup.csv` 填"
              " `ship_correct` / `box_needs_adjustment` / `non_ship` / `uncertain`，"
              "并写 `label_review_note`。若已在副本中改框，注明对应图像。"
              "之后比较副本与 v3 原标签，并仅应用人工确认的改动。", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("Created", OUT)
    print("Batch sizes", [count for _, _, count, _ in launch])
    print("First script", launch[0][3])


if __name__ == "__main__":
    main()
