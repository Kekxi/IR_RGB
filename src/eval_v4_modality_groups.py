"""Evaluate RGB, IR, or dual-stream best.pt on the same V4 fog/clear pair IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401; checkpoint class registration


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("rgb", "ir", "dual"), required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        pairs = {r["id"]: r for r in csv.DictReader(handle)
                 if r["split"] == "val" and r["same_image_size"] == "1"}
    with args.conditions.open(encoding="utf-8-sig", newline="") as handle:
        conditions = list(csv.DictReader(handle))
    if len(pairs) != 877 or len(conditions) != 877 or {r["id"] for r in conditions} != set(pairs):
        raise ValueError("Expected the same 877 eligible validation IDs")
    groups = {"haze_fog": [], "clear": []}
    modality = "rgb" if args.variant == "dual" else args.variant
    for row in conditions:
        group = row["visibility"].strip().lower()
        if group not in groups:
            raise ValueError(f"Missing or unexpected condition for {row['id']}: {group}")
        groups[group].append((dataset / pairs[row["id"]][f"{modality}_image"]).as_posix())
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    results = []
    for group, images in groups.items():
        manifest = output / f"{group}.txt"
        manifest.write_text("\n".join(images) + "\n", encoding="utf-8")
        yaml = output / f"{group}.yaml"
        yaml.write_text(
            f"path: {json.dumps(dataset.as_posix())}\n"
            f"train: {json.dumps(manifest.as_posix())}\n"
            f"val: {json.dumps(manifest.as_posix())}\n"
            f"channels: {6 if args.variant == 'dual' else 3}\n"
            "names:\n  0: ship\n", encoding="utf-8")
        kwargs = {"validator": RGBIRDetectionValidator} if args.variant == "dual" else {}
        metrics = model.val(
            data=str(yaml), imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=str(output), name=f"eval_{group}", exist_ok=False, plots=True,
            **kwargs)
        results.append({"variant": args.variant, "condition": group, "images": len(images),
                        "AP50": metrics.box.map50, "AP50_95": metrics.box.map,
                        "recall": float(metrics.box.mr),
                        "label_coordinate": modality})
    with (output / "group_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(results)


if __name__ == "__main__":
    main()
