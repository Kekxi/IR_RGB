"""Train and validate either B2 hypothesis from one trained B1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator  # register paired detector classes
from fog_reliability import DualStreamReliabilityDetectionModel, RGBIRReliabilityTrainer  # register B1 classes
from b2_variants import (
    DualStreamAlignmentDetectionModel,
    DualStreamSmallTargetDetectionModel,
    RGBIRAlignmentTrainer,
    RGBIRSmallTargetTrainer,
)

MODELS = {"control": DualStreamReliabilityDetectionModel,
          "align": DualStreamAlignmentDetectionModel, "small": DualStreamSmallTargetDetectionModel}
TRAINERS = {"control": RGBIRReliabilityTrainer,
            "align": RGBIRAlignmentTrainer, "small": RGBIRSmallTargetTrainer}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("smoke", "debug", "train", "val"))
    p.add_argument("--variant", choices=tuple(MODELS), required=True)
    p.add_argument("--weights", type=Path, required=True,
                   help="B1 best.pt for smoke/debug/train; matching B2 best.pt for val")
    p.add_argument("--data", type=Path, help="fusion_eligible.yaml for debug/train/val")
    p.add_argument("--project", type=Path)
    p.add_argument("--name")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    model = YOLO(str(args.weights.resolve()))
    if args.command == "smoke":
        if not isinstance(model.model, DualStreamReliabilityDetectionModel):
            raise TypeError("Smoke test requires a B1 checkpoint")
        candidate = (model.model if args.variant == "control" else MODELS[args.variant].from_b1(model.model)).eval()
        with torch.no_grad():
            prediction = candidate(torch.zeros(1, 6, 128, 128))
        print(f"B2 {args.variant} CPU forward OK: {type(prediction).__name__}")
        return
    if not args.data or not args.data.is_file() or not args.project or not args.name:
        raise ValueError("--data, --project and --name are required")
    if args.command == "val":
        if not isinstance(model.model, MODELS[args.variant]) or (args.variant == "control" and isinstance(
            model.model, (DualStreamAlignmentDetectionModel, DualStreamSmallTargetDetectionModel)
        )):
            raise TypeError(f"Validation requires a B2 {args.variant} checkpoint")
        metrics = model.val(
            validator=RGBIRDetectionValidator, data=str(args.data.resolve()),
            project=str(args.project.resolve()), name=args.name, exist_ok=False,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            workers=args.workers, save_json=False, plots=True,
        )
        print("AP50:", metrics.box.map50, "AP50-95:", metrics.box.map, "recall:", metrics.box.mr)
        return
    if not isinstance(model.model, DualStreamReliabilityDetectionModel) or isinstance(
        model.model, (DualStreamAlignmentDetectionModel, DualStreamSmallTargetDetectionModel)
    ):
        raise TypeError("Training requires the original B1 best.pt")
    debug = args.command == "debug"
    model.train(
        trainer=TRAINERS[args.variant], data=str(args.data.resolve()),
        project=str(args.project.resolve()), name=args.name, exist_ok=False,
        epochs=1 if debug else args.epochs,
        patience=1 if debug else args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        nbs=1 if debug else 64,
        device=args.device, workers=args.workers,
        fraction=0.02 if debug else 1.0,
        optimizer="SGD", lr0=0.001, lrf=0.01, momentum=0.937,
        weight_decay=5e-4, cos_lr=True, close_mosaic=10,
        amp=True, cache=False, seed=args.seed, deterministic=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        save=True, plots=True,
    )


if __name__ == "__main__":
    main()
