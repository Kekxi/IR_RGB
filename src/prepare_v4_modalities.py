"""Build paired-ID RGB, IR, and dual-stream manifests for the V4 dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def prepare(dataset: Path, output: Path) -> dict:
    dataset, output = dataset.resolve(), output.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Manifest output must be outside the dataset")
    pairs = dataset / "pairs.csv"
    if not pairs.is_file():
        raise FileNotFoundError(pairs)
    files = {variant: {"train": [], "val": []}
             for variant in ("rgb", "ir")}
    ids = {"train": set(), "val": set()}
    excluded = []
    with pairs.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2145 or len({r["id"] for r in rows}) != 2145:
        raise ValueError("Expected 2145 unique V4 pair IDs")
    for row in rows:
        split, key = row["split"], row["id"]
        if split not in ids:
            raise ValueError(f"Unexpected split: {split}")
        for field in ("rgb_image", "ir_image", "rgb_label", "ir_label"):
            if not (dataset / row[field]).is_file():
                raise FileNotFoundError(f"{key}: missing {field}")
        if row["same_image_size"] != "1":
            excluded.append(key)
            continue
        ids[split].add(key)
        for variant in files:
            image = (dataset / row[f"{variant}_image"]).resolve()
            label = (dataset / row[f"{variant}_label"]).resolve()
            if (image.parts[-4:-1] != ("images", variant, split)
                    or label.parts[-4:-1] != ("labels", variant, split)
                    or image.stem != label.stem):
                raise ValueError(f"Unexpected {variant} image/label layout for {key}")
            files[variant][split].append(image.as_posix())
    if len(ids["train"]) != 1266 or len(ids["val"]) != 877 or len(excluded) != 2:
        raise ValueError("V4 eligible split must have 1266 train, 877 val, 2 excluded")
    output.mkdir(parents=True, exist_ok=True)
    for variant in files:
        for split in ids:
            manifest = output / f"{variant}_{split}.txt"
            manifest.write_text("\n".join(files[variant][split]) + "\n", encoding="utf-8")
        yaml = output / f"{variant}_eligible.yaml"
        yaml.write_text(
            f"path: {json.dumps(dataset.as_posix())}\n"
            f"train: {json.dumps((output / f'{variant}_train.txt').as_posix())}\n"
            f"val: {json.dumps((output / f'{variant}_val.txt').as_posix())}\n"
            "channels: 3\nnames:\n  0: ship\n", encoding="utf-8")
    summary = {"dataset": str(dataset), "train": len(ids["train"]),
               "val": len(ids["val"]), "excluded": excluded,
               "rgb_labels": "labels/rgb", "ir_labels": "labels/ir"}
    (output / "modality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output), ensure_ascii=False, indent=2))
