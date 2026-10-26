"""Create an isolated complete dataset with the 28 confirmed LabelImg edits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


BASE = Path(__file__).resolve().parent
SOURCE = Path(r"E:\HEU_4151\RGB_IR\MTDSP_v3_labelreview_20261020\dataset")
REVIEW = BASE / "fog_small_miss_b2_s42_v3" / "LabelImg_28_review"
ROOT = Path(r"E:\HEU_4151\RGB_IR\MTDSP_v4_labelreview_20261021")
DATA = ROOT / "dataset"


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_labels(src: Path, dest: Path):
    for file in src.rglob("*.txt"):
        target = dest / file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)


def main():
    if ROOT.exists():
        raise FileExistsError(f"v4 directory already exists: {ROOT}")
    with (REVIEW / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        review = list(csv.DictReader(f))
    if len(review) != 28:
        raise ValueError("Expected 28 reviewed labels")
    for item in review:
        src = Path(item["source_label"])
        if sha(src) != item["source_sha256"] or sha(src) == sha(Path(item["review_label"])):
            raise ValueError(f"Source changed or review missing: {item['id']}")

    DATA.mkdir(parents=True)
    # Hard links save disk space; images are never edited in this workflow.
    shutil.copytree(SOURCE / "images", DATA / "images", copy_function=os.link)
    copy_labels(SOURCE / "labels", DATA / "labels")
    if (SOURCE / "unpaired").exists():
        shutil.copytree(SOURCE / "unpaired" / "images", DATA / "unpaired" / "images",
                        copy_function=os.link)
        copy_labels(SOURCE / "unpaired" / "labels", DATA / "unpaired" / "labels")
    for name in ("pairs.csv", "candidates.csv", "unpaired.csv", "fusion.yaml", "rgb.yaml", "ir.yaml"):
        if (SOURCE / name).is_file():
            shutil.copyfile(SOURCE / name, DATA / name)
    for item in review:
        dest = DATA / "labels" / "rgb" / "val" / f"{item['id']}.txt"
        shutil.copyfile(item["review_label"], dest)
        if sha(dest) != sha(Path(item["review_label"])):
            raise AssertionError(f"v4 label mismatch: {item['id']}")

    with (DATA / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        pairs = list(csv.DictReader(f))
    eligible = {split: [r for r in pairs if r["split"] == split and r["same_image_size"] == "1"]
                for split in ("train", "val")}
    if len(eligible["train"]) != 1266 or len(eligible["val"]) != 877:
        raise ValueError("Unexpected eligible pair counts")
    for split, rows in eligible.items():
        for row in rows:
            for key in ("rgb_image", "ir_image", "rgb_label", "ir_label"):
                if not (DATA / row[key]).is_file():
                    raise FileNotFoundError(DATA / row[key])
    manifests = ROOT / "manifests"
    manifests.mkdir()
    for split, rows in eligible.items():
        (manifests / f"{split}.txt").write_text(
            "\n".join((DATA / r["rgb_image"]).as_posix() for r in rows) + "\n", encoding="utf-8")
    (manifests / "fusion_eligible.yaml").write_text(
        f"path: {json.dumps(DATA.as_posix())}\n"
        f"train: {json.dumps((manifests / 'train.txt').as_posix())}\n"
        f"val: {json.dumps((manifests / 'val.txt').as_posix())}\n"
        "channels: 6\nnames:\n  0: ship\n", encoding="utf-8")
    provenance = {
        "source_dataset": str(SOURCE), "review_manifest": str(REVIEW / "review_manifest.csv"),
        "changed_rgb_val_labels": len(review), "coordinate_only": 25, "one_box_removed": 3,
        "eligible_train_pairs": len(eligible["train"]), "eligible_val_pairs": len(eligible["val"]),
        "image_storage": "NTFS hardlinks to v3 (read-only workflow)",
        "label_storage": "independent copied TXT files",
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# MTDSP v4: 28-frame LabelImg review\n\n"
        "Independent labels; images are hard links to v3. Training labels and train/val split are unchanged. "
        "Only 28 RGB validation labels differ: 25 coordinate adjustments and three one-box removals. "
        "Use `manifests/fusion_eligible.yaml` for local evaluation. "
        "This version is local and must not be confused with the v3 dataset on the server.\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
