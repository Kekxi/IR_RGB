"""Continue B3 training with matched control/direct/local IR-P3 variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from b3_dual_head import B3DualHeadDetectionModel, B3DualHeadTrainer, B3IRValidator, B3RGBValidator
from b3_phase2 import B3P3DirectTrainer, B3P3LocalTrainer, B3Phase2DetectionModel


TRAINERS = {"control": B3DualHeadTrainer, "direct": B3P3DirectTrainer,
            "local": B3P3LocalTrainer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "val"))
    parser.add_argument("--variant", choices=TRAINERS, help="Required for train")
    parser.add_argument("--head", choices=("rgb", "ir"), help="Required for val")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if ultralytics.__version__ != "8.3.163":
        raise RuntimeError(f"Expected ultralytics 8.3.163, found {ultralytics.__version__}")
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.command == "train":
        if args.variant is None or args.head is not None:
            parser.error("train requires --variant and does not use --head")
        if args.epochs < 1:
            parser.error("--epochs must be positive")
    elif args.head is None or args.variant is not None:
        parser.error("val requires --head and does not use --variant")
    yolo = YOLO(str(args.weights.resolve()))
    if args.command == "train":
        if type(yolo.model) is not B3DualHeadDetectionModel:
            raise TypeError("All three variants must start from the same B3 phase-1 best.pt")
        yolo.train(
            trainer=TRAINERS[args.variant], data=str(args.data.resolve()),
            project=str(args.project.resolve()), name=args.name, exist_ok=False,
            epochs=args.epochs, patience=args.epochs, imgsz=args.imgsz,
            batch=args.batch, workers=args.workers, device=args.device, seed=args.seed,
            optimizer="SGD", lr0=0.001, lrf=0.01, momentum=0.937,
            weight_decay=5e-4, cos_lr=True, close_mosaic=10,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, pretrained=False,
            amp=True, cache=False, deterministic=True, save=True, plots=True,
        )
        return
    if not isinstance(yolo.model, B3DualHeadDetectionModel):
        raise TypeError("Validation requires a B3 native-dual-head checkpoint")
    head = args.head
    metrics = yolo.val(
        validator=B3RGBValidator if head == "rgb" else B3IRValidator,
        data=str(args.data.resolve()), project=str(args.project.resolve()),
        name=args.name, exist_ok=False, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        conf=0.001, iou=0.7, max_det=300, save_json=False, plots=False,
    )
    summary = {"head": head, "coordinate_system": head,
               "architecture": type(yolo.model).__name__,
               "transfer_radius": (yolo.model.ir_p3_transfer.radius
                                   if isinstance(yolo.model, B3Phase2DetectionModel) else None),
               "weights": str(args.weights.resolve()), "data": str(args.data.resolve()),
               "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
               "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}
    output = args.project / args.name / "native_metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
