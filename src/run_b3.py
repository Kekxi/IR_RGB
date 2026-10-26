"""Train B3 phase-1 dual-native-head model or evaluate either native head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from fog_reliability import DualStreamReliabilityDetectionModel
from b3_dual_head import (
    B3DualHeadDetectionModel,
    B3DualHeadTrainer,
    B3IRValidator,
    B3RGBValidator,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "val"))
    parser.add_argument("--weights", type=Path, required=True,
                        help="V4 B1 best.pt for train; B3 best.pt for val")
    parser.add_argument("--data", type=Path, required=True,
                        help="B3 fusion_eligible.yaml; paths must point to RGB images")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--head", choices=("rgb", "ir"), help="Required for val")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if ultralytics.__version__ != "8.3.163":
        raise RuntimeError(f"B3 is pinned to ultralytics 8.3.163, found {ultralytics.__version__}")
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.command == "train" and args.head is not None:
        parser.error("--head is only used with val")
    if args.command == "val" and args.head is None:
        parser.error("val requires --head rgb or --head ir")

    yolo = YOLO(str(args.weights.resolve()))
    if args.command == "train":
        if type(yolo.model) is not DualStreamReliabilityDetectionModel:
            raise TypeError("B3 training must start from the original V4 B1 best.pt")
        yolo.train(
            trainer=B3DualHeadTrainer,
            data=str(args.data.resolve()),
            project=str(args.project.resolve()),
            name=args.name,
            exist_ok=False,
            epochs=args.epochs,
            patience=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=5e-4,
            cos_lr=True,
            close_mosaic=10,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            pretrained=False,
            amp=True,
            cache=False,
            deterministic=True,
            save=True,
            plots=True,
        )
        return

    if not isinstance(yolo.model, B3DualHeadDetectionModel):
        raise TypeError("Native-head validation requires a trained B3 best.pt")
    metrics = yolo.val(
        validator=B3RGBValidator if args.head == "rgb" else B3IRValidator,
        data=str(args.data.resolve()),
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        save_json=False,
        plots=True,
    )
    summary = {
        "head": args.head,
        "coordinate_system": args.head,
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    output = args.project / args.name / "native_metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
