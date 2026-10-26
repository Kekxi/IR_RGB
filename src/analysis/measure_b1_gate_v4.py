"""Measure B1 RGB gate weights during a paired validation pass."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
from fog_reliability import DualStreamReliabilityDetectionModel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    model = YOLO(str(args.weights.resolve()))
    if not isinstance(model.model, DualStreamReliabilityDetectionModel):
        raise TypeError("Expected B1 dual-stream checkpoint")
    stats = {name: {"n": 0, "sum": 0.0, "sum_sq": 0.0,
                    "min": 1.0, "max": 0.0,
                    "rgb_gt_0_6": 0, "ir_gt_0_6": 0}
             for name in ("P3", "P4", "P5")}
    hooks = []
    for index, name in ((2, "P3"), (3, "P4"), (4, "P5")):
        def record(_module, _input, output, scale=name):
            weight = output.detach().float().sigmoid()
            item = stats[scale]
            item["n"] += weight.numel()
            item["sum"] += float(weight.sum())
            item["sum_sq"] += float((weight * weight).sum())
            item["min"] = min(item["min"], float(weight.min()))
            item["max"] = max(item["max"], float(weight.max()))
            item["rgb_gt_0_6"] += int((weight > 0.6).sum())
            item["ir_gt_0_6"] += int((weight < 0.4).sum())
        hooks.append(model.model.model[index].gate.register_forward_hook(record))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        model.val(validator=RGBIRDetectionValidator, data=str(args.data.resolve()),
                  imgsz=args.imgsz, batch=args.batch, device=args.device,
                  project=str(args.output.resolve()), name="validation", exist_ok=False,
                  plots=False)
    finally:
        for hook in hooks:
            hook.remove()
    result = {}
    for scale, item in stats.items():
        if not item["n"]:
            raise RuntimeError(f"No gate outputs captured at {scale}")
        n = item["n"]
        mean = item["sum"] / n
        result[scale] = {"mean_rgb_weight": mean,
                         "std_rgb_weight": math.sqrt(max(item["sum_sq"] / n - mean * mean, 0)),
                         "min_rgb_weight": item["min"], "max_rgb_weight": item["max"],
                         "fraction_rgb_weight_gt_0_6": item["rgb_gt_0_6"] / n,
                         "fraction_ir_weight_gt_0_6": item["ir_gt_0_6"] / n,
                         "positions": n}
    (args.output / "gate_weights.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
