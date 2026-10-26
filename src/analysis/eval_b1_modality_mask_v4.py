"""Diagnostic zero-input ablation of a trained B1 model on V4 RGB targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401


class MaskValidator(RGBIRDetectionValidator):
    masked_modality = ""

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if self.masked_modality == "ir":
            batch["img"][:, 3:6].zero_()
        elif self.masked_modality == "rgb":
            batch["img"][:, :3].zero_()
        else:
            raise ValueError(self.masked_modality)
        return batch


def main() -> None:
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
    for masked in ("ir", "rgb"):
        MaskValidator.masked_modality = masked
        metrics = model.val(validator=MaskValidator, data=str(args.data.resolve()),
                            imgsz=args.imgsz, batch=args.batch, device=args.device,
                            project=str(args.output.resolve()), name=f"mask_{masked}",
                            exist_ok=False, plots=False)
        rows.append({"masked": masked, "AP50": float(metrics.box.map50),
                     "AP50_95": float(metrics.box.map), "recall": float(metrics.box.mr)})
    with (args.output / "mask_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
