"""Train and validate paired YOLO11 B0 (concat) and B1 (reliability gate)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# Import both modules before YOLO.load: saved checkpoints contain custom classes.
from rgb_ir_midfusion import DualStreamDetectionModel, RGBIRDetectionTrainer, RGBIRDetectionValidator
from fog_reliability import DualStreamReliabilityDetectionModel, RGBIRReliabilityTrainer


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train", "val"):
        s = sub.add_parser(command)
        s.add_argument("--variant", choices=("b0", "b1"), required=True)
        s.add_argument("--weights", type=Path, required=True,
                       help="train/smoke: official yolo11m.pt; val: this variant's best.pt")
        if command != "smoke":
            s.add_argument("--data", type=Path, required=True, help="prepare_data.py output YAML")
            s.add_argument("--project", type=Path, required=True)
            s.add_argument("--name", required=True)
            s.add_argument("--imgsz", type=int, default=960)
            s.add_argument("--batch", type=int, default=2)
            s.add_argument("--device", default="0")
            s.add_argument("--workers", type=int, default=2)
        if command == "train":
            s.add_argument("--epochs", type=int, default=220)
            s.add_argument("--patience", type=int, default=40)
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--fraction", type=float, default=1.0,
                           help="Debug only when <1; formal comparison uses 1")
    return p


def main() -> None:
    args = parser().parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.command != "smoke" and not args.data.is_file():
        raise FileNotFoundError(args.data)

    if args.command == "smoke":
        single = YOLO(str(args.weights)).model
        cls = DualStreamDetectionModel if args.variant == "b0" else DualStreamReliabilityDetectionModel
        model = cls(single).eval()
        with torch.no_grad():
            result = model(torch.zeros(1, 6, 64, 64))
        print(f"{args.variant} smoke forward OK; output type={type(result).__name__}")
        return

    model = YOLO(str(args.weights))
    if args.command == "val":
        metrics = model.val(
            validator=RGBIRDetectionValidator,
            data=str(args.data.resolve()),
            imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
            project=str(args.project.resolve()), name=args.name,
            save_json=False, plots=True,
        )
        print("AP50:", metrics.box.map50, "AP50-95:", metrics.box.map)
        return

    trainer = RGBIRDetectionTrainer if args.variant == "b0" else RGBIRReliabilityTrainer
    model.train(
        trainer=trainer,
        data=str(args.data.resolve()),
        project=str(args.project.resolve()), name=args.name, exist_ok=False,
        epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, fraction=args.fraction,
        optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=5e-4, cos_lr=True, close_mosaic=10,
        pretrained=True, amp=True, cache=False, seed=args.seed, deterministic=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        save=True, plots=True,
    )


if __name__ == "__main__":
    main()
