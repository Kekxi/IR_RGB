"""Train matched V4 RGB-only, IR-only, and B1 dual-stream YOLO11m baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from rgb_ir_midfusion import RGBIRDetectionValidator
from fog_reliability import DualStreamReliabilityDetectionModel, RGBIRReliabilityTrainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("smoke", "debug", "train", "val"))
    p.add_argument("--variant", choices=("rgb", "ir", "dual"), required=True)
    p.add_argument("--weights", type=Path, required=True,
                   help="YOLO11m pretrained checkpoint for smoke/debug/train; matching best.pt for val")
    p.add_argument("--data", type=Path)
    p.add_argument("--project", type=Path)
    p.add_argument("--name")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.command != "smoke" and (not args.data or not args.data.is_file()
                                    or not args.project or not args.name):
        raise ValueError("--data, --project and --name are required")

    yolo = YOLO(str(args.weights.resolve()))
    if args.command == "smoke":
        model = (DualStreamReliabilityDetectionModel(yolo.model)
                 if args.variant == "dual" else yolo.model).eval()
        channels = 6 if args.variant == "dual" else 3
        with torch.no_grad():
            output = model(torch.zeros(1, channels, 64, 64))
        print(f"{args.variant} forward OK: {type(output).__name__}")
        return

    if args.command == "val":
        if args.variant == "dual" and not isinstance(yolo.model, DualStreamReliabilityDetectionModel):
            raise TypeError("Dual validation requires a B1 dual-stream checkpoint")
        kwargs = {"validator": RGBIRDetectionValidator} if args.variant == "dual" else {}
        metrics = yolo.val(
            data=str(args.data.resolve()), imgsz=args.imgsz, batch=args.batch,
            workers=args.workers, device=args.device, project=str(args.project.resolve()),
            name=args.name, exist_ok=False, plots=True, **kwargs)
        print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map,
              "recall:", metrics.box.mr)
        return

    debug = args.command == "debug"
    trainer = {"trainer": RGBIRReliabilityTrainer} if args.variant == "dual" else {}
    yolo.train(
        data=str(args.data.resolve()), project=str(args.project.resolve()),
        name=args.name, exist_ok=False, epochs=1 if debug else args.epochs,
        patience=1 if debug else args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, seed=args.seed,
        fraction=0.02 if debug else 1.0, nbs=1 if debug else 64,
        optimizer="SGD", lr0=0.001, lrf=0.01, momentum=0.937,
        weight_decay=5e-4, cos_lr=True, close_mosaic=10,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        pretrained=True, amp=True, cache=False, deterministic=True,
        save=True, plots=True, **trainer)


if __name__ == "__main__":
    main()
