"""Controlled IR translation test; RGB images and RGB-coordinate labels stay fixed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401, checkpoint registration
import b2_variants  # noqa: F401, checkpoint registration


class ShiftedIRValidator(RGBIRDetectionValidator):
    shift_px = 0

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        shift = self.shift_px
        if shift:
            image = batch["img"].clone()
            ir = image[:, 3:6]
            shifted = torch.zeros_like(ir)
            shifted[:, :, :, shift:] = ir[:, :, :, :-shift]
            image[:, 3:6] = shifted
            batch["img"] = image
        return batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights.resolve()))
    rows = []
    for shift in (0, 4, 8, 16):
        ShiftedIRValidator.shift_px = shift
        metrics = model.val(
            validator=ShiftedIRValidator, data=str(args.data.resolve()),
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=str(args.output.resolve()), name=f"ir_right_{shift}px",
            exist_ok=False, plots=False,
        )
        rows.append({"ir_shift_input_px": shift, "AP50": metrics.box.map50,
                     "AP50_95": metrics.box.map, "recall": float(metrics.box.mr)})
    with (args.output / "shift_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
