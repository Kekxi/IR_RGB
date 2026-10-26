"""Measure RGB/IR backbone feature magnitudes before the B1 reliability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    yolo = YOLO(str(args.weights.resolve()))
    sums = {scale: {modality: {"sum_abs": 0.0, "n": 0}
                    for modality in ("rgb", "ir")}
            for scale in ("P3", "P4", "P5")}
    hooks = []
    for index, modality in ((0, "rgb"), (1, "ir")):
        def record(_module, _inputs, outputs, branch=modality):
            for scale, feature in zip(("P3", "P4", "P5"), outputs, strict=True):
                item = sums[scale][branch]
                item["sum_abs"] += float(feature.detach().float().abs().sum())
                item["n"] += feature.numel()
        hooks.append(yolo.model.model[index].register_forward_hook(record))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        yolo.val(validator=RGBIRDetectionValidator, data=str(args.data.resolve()),
                 imgsz=args.imgsz, batch=args.batch, device=args.device,
                 project=str(args.output.resolve()), name="validation", exist_ok=False,
                 plots=False)
    finally:
        for hook in hooks:
            hook.remove()
    result = {}
    for scale, values in sums.items():
        result[scale] = {modality: values[modality]["sum_abs"] / values[modality]["n"]
                         for modality in ("rgb", "ir")}
        result[scale]["ir_over_rgb"] = result[scale]["ir"] / result[scale]["rgb"]
    (args.output / "feature_energy.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
