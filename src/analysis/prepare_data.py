"""Create read-only-to-source train/val manifests for equal-size MTDSP RGB/IR pairs.

Only writes to --output. It never edits images or LabelImg labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def prepare(dataset: Path, output: Path) -> dict:
    dataset = dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("--output must be outside the frozen dataset")
    pairs_path = dataset / "pairs.csv"
    if not pairs_path.is_file():
        raise FileNotFoundError(pairs_path)

    manifests: dict[str, list[str]] = {"train": [], "val": []}
    excluded: list[str] = []
    ids: set[str] = set()
    with pairs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = row["id"]
            if pair_id in ids:
                raise ValueError(f"Duplicate pair ID: {pair_id}")
            ids.add(pair_id)
            split = row["split"]
            if split not in manifests:
                raise ValueError(f"Unexpected split for {pair_id}: {split}")
            for column in ("rgb_image", "ir_image", "rgb_label", "ir_label"):
                if not (dataset / row[column]).is_file():
                    raise FileNotFoundError(f"{pair_id}: missing {column}={row[column]}")
            if row["same_image_size"] != "1":
                excluded.append(pair_id)
                continue
            rgb = (dataset / row["rgb_image"]).resolve()
            ir = (dataset / row["ir_image"]).resolve()
            if rgb.name != ir.name or rgb.parent.name != split or ir.parent.name != split:
                raise ValueError(f"Unexpected pair layout for {pair_id}: {rgb}, {ir}")
            manifests[split].append(rgb.as_posix())

    if len(ids) != 2145 or len(manifests["train"]) != 1266 or len(manifests["val"]) != 877:
        raise ValueError(f"Dataset count differs from frozen version: {len(ids)}, "
                         f"train={len(manifests['train'])}, val={len(manifests['val'])}")
    output.mkdir(parents=True, exist_ok=True)
    for split, paths in manifests.items():
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    yaml_text = (f"path: {json.dumps(dataset.as_posix())}\n"
                 f"train: {json.dumps((output / 'train.txt').as_posix())}\n"
                 f"val: {json.dumps((output / 'val.txt').as_posix())}\n"
                 "channels: 6\nnames:\n  0: ship\n")
    (output / "fusion_eligible.yaml").write_text(yaml_text, encoding="utf-8")
    summary = {"dataset": str(dataset), "pairs": len(ids), "train": len(manifests["train"]),
               "val": len(manifests["val"]), "excluded_size_mismatch": excluded}
    (output / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing pairs.csv and images/")
    parser.add_argument("--output", type=Path, required=True, help="Writable directory for manifest files")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output), ensure_ascii=False, indent=2))
