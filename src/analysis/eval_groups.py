"""Evaluate the same checkpoint on reviewed fog/clear validation subsets."""

from __future__ import annotations

import argparse
import csv
import json
import importlib.util
from collections import Counter
from pathlib import Path

from ultralytics import YOLO
from rgb_ir_midfusion import RGBIRDetectionValidator  # also registers B0 pickle classes
import fog_reliability  # noqa: F401; registers B1 pickle classes
if importlib.util.find_spec("b2_variants") is not None:
    import b2_variants  # noqa: F401; registers B2 pickle classes when available


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--allow-partial", action="store_true", help="Exploratory only; excludes unreviewed val IDs")
    args = p.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        eligible = {r["id"]: r for r in csv.DictReader(f)
                    if r["split"] == "val" and r["same_image_size"] == "1"}
    with args.conditions.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(eligible) != 877 or len(rows) != 877 or {r["id"] for r in rows} != set(eligible):
        raise ValueError("Conditions must contain every one of the 877 eligible val IDs exactly once")
    groups: dict[str, list[str]] = {"haze_fog": [], "clear": []}
    unreviewed = []
    other = Counter()
    for row in rows:
        condition = row["visibility"].strip().lower()
        if condition in groups:
            groups[condition].append((dataset / eligible[row["id"]]["rgb_image"]).as_posix())
        elif condition == "":
            unreviewed.append(row["id"])
        elif condition in ("rain_snow", "sea_spray", "low_visibility_unknown", "uncertain"):
            other[condition] += 1
        else:
            raise ValueError(f"Invalid visibility {condition!r} for {row['id']}")
    if unreviewed and not args.allow_partial:
        raise ValueError(f"{len(unreviewed)} val IDs lack visibility review; use --allow-partial only for exploratory checks")
    if not all(groups.values()):
        raise ValueError("Both haze_fog and clear groups need at least one image")
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    result_rows = []
    for condition, images in groups.items():
        manifest = output / f"{condition}.txt"
        manifest.write_text("\n".join(images) + "\n", encoding="utf-8")
        data_yaml = output / f"{condition}.yaml"
        data_yaml.write_text(f"path: {json.dumps(dataset.as_posix())}\n"
                             f"train: {json.dumps(manifest.as_posix())}\n"
                             f"val: {json.dumps(manifest.as_posix())}\n"
                             "channels: 6\nnames:\n  0: ship\n", encoding="utf-8")
        metrics = model.val(validator=RGBIRDetectionValidator, data=str(data_yaml),
                            imgsz=args.imgsz, batch=args.batch, device=args.device,
                            project=str(output), name=f"eval_{condition}", plots=True)
        result_rows.append({"condition": condition, "images": len(images),
                            "AP50": metrics.box.map50, "AP50_95": metrics.box.map,
            "recall": float(metrics.box.mr), "unreviewed_excluded": len(unreviewed),
            "other_conditions_excluded": sum(other.values())})
    print("Other reviewed conditions excluded from fog/clear comparison:", dict(other))
    with (output / "group_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(result_rows)


if __name__ == "__main__":
    main()
