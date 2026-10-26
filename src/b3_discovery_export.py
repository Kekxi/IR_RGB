"""Prepare V4 fog-pair review material and export native predictions from five detectors.

Ultralytics is pinned to 8.3.163 for checkpoint and validator compatibility.
The export uses the normal validator postprocessing and writes original-image
coordinate boxes normalized to [0, 1]. No image or label in the dataset is edited.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VARIANTS = ("rgb_only", "ir_only", "b1", "b3_rgb", "b3_ir")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _gt(path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or int(float(fields[0])) != 0:
            raise ValueError(f"Expected one-class YOLO box at {path}:{line_number}")
        _, xc, yc, w, h = map(float, fields)
        if not all(math.isfinite(v) for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid box at {path}:{line_number}")
        box = [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
        if min(box) < -1e-5 or max(box) > 1 + 1e-5:
            raise ValueError(f"Box outside image at {path}:{line_number}")
        boxes.append([max(0.0, min(1.0, v)) for v in box])
    return boxes


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def _review_image(pair: dict, output: Path, dataset: Path) -> None:
    import cv2

    panels = []
    for modality, color in (("rgb", (255, 80, 30)), ("ir", (30, 80, 255))):
        image = cv2.imread(str(dataset / pair[f"{modality}_image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(pair[f"{modality}_image"])
        height, width = image.shape[:2]
        for index, box in enumerate(pair[f"gt_{modality}"]):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (round(x1 * width), round(y1 * height)),
                          (round(x2 * width), round(y2 * height)), color, 2)
            cv2.putText(image, f"{modality[0].upper()}{index}",
                        (max(0, round(x1 * width)), max(28, round(y1 * height) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        scale = min(1.0, 900 / width)
        if scale < 1:
            image = cv2.resize(image, (round(width * scale), round(height * scale)))
        panels.append(image)
    if panels[0].shape[0] != panels[1].shape[0]:
        raise ValueError(f"Different RGB/IR panel height for {pair['id']}")
    combined = cv2.hconcat(panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), combined, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"Failed to write {output}")


def prepare(dataset: Path, conditions: Path, output: Path, limit: int = 0,
            review_images: bool = False) -> dict:
    dataset, output = dataset.resolve(), output.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Output must be outside the dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared output is not empty: {output}")
    pairs = {row["id"]: row for row in _rows(dataset / "pairs.csv")
             if row["split"] == "val" and row["same_image_size"] == "1"}
    weather = {row["id"]: row["visibility"].strip().lower() for row in _rows(conditions)}
    if len(pairs) != 877 or set(weather) != set(pairs):
        raise ValueError("Expected the 877 eligible V4 validation pairs and conditions")
    selected = [pairs[pair_id] for pair_id in sorted(pairs) if weather[pair_id] == "haze_fog"]
    if len(selected) != 633:
        raise ValueError(f"Expected 633 haze/fog pairs, got {len(selected)}")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit:
        selected = selected[:limit]
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    rgb_images, ir_images = [], []
    review_rows = []
    for row in selected:
        pair = {"id": row["id"], "event": row["event"], "visibility": "haze_fog",
                "rgb_image": row["rgb_image"], "ir_image": row["ir_image"],
                "gt_rgb": _gt(dataset / row["rgb_label"]),
                "gt_ir": _gt(dataset / row["ir_label"])}
        prepared.append(pair)
        rgb_images.append((dataset / row["rgb_image"]).as_posix())
        ir_images.append((dataset / row["ir_image"]).as_posix())
        for ir_index, ir_box in enumerate(pair["gt_ir"]):
            candidates = []
            for rgb_index, rgb_box in enumerate(pair["gt_rgb"]):
                ir_center = ((ir_box[0] + ir_box[2]) / 2, (ir_box[1] + ir_box[3]) / 2)
                rgb_center = ((rgb_box[0] + rgb_box[2]) / 2, (rgb_box[1] + rgb_box[3]) / 2)
                distance = math.dist(ir_center, rgb_center)
                candidates.append((distance, rgb_index, _iou(ir_box, rgb_box)))
            candidates.sort()
            near = candidates[0] if candidates else None
            review_rows.append({"id": pair["id"], "ir_gt_index": ir_index,
                                "rgb_gt_index": "", "status": "", "notes": "",
                                "suggested_rgb_gt_index": near[1] if near else "",
                                "suggested_same_pixel_iou": round(near[2], 4) if near else "",
                                "suggested_center_distance": round(near[0], 4) if near else ""})
        if review_images:
            _review_image(pair, output / "review_images" / f"{pair['id']}.jpg", dataset)
    (output / "pairs.jsonl").write_text(
        "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in prepared), encoding="utf-8")
    fields = ("id", "ir_gt_index", "rgb_gt_index", "status", "notes",
              "suggested_rgb_gt_index", "suggested_same_pixel_iou",
              "suggested_center_distance")
    with (output / "correspondence_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    for modality, images in (("rgb", rgb_images), ("ir", ir_images), ("fusion", rgb_images)):
        manifest = output / f"{modality}_val.txt"
        manifest.write_text("\n".join(images) + "\n", encoding="utf-8")
        yaml = (f"path: {json.dumps(dataset.as_posix())}\n"
                f"train: {json.dumps(manifest.as_posix())}\n"
                f"val: {json.dumps(manifest.as_posix())}\n"
                f"channels: {6 if modality == 'fusion' else 3}\n"
                "names:\n  0: ship\n")
        (output / f"{modality}.yaml").write_text(yaml, encoding="utf-8")
    summary = {"pairs": len(prepared), "rgb_gt": sum(len(p["gt_rgb"]) for p in prepared),
               "ir_gt": sum(len(p["gt_ir"]) for p in prepared),
               "review_rows": len(review_rows), "review_images": review_images}
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def export(prepared: Path, variant: str, weights: Path, output: Path,
           imgsz: int, batch: int, workers: int, device: str) -> dict:
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    from b3_dual_head import B3IRValidator, B3RGBValidator, B3DualHeadDetectionModel
    from b3_phase2 import B3Phase2DetectionModel  # Register phase-2 checkpoint class before YOLO loads it.
    from fog_reliability import DualStreamReliabilityDetectionModel
    from rgb_ir_midfusion import RGBIRDetectionValidator

    if ultralytics.__version__ != "8.3.163":
        raise RuntimeError(f"Expected ultralytics 8.3.163; found {ultralytics.__version__}")
    prepared, weights, output = prepared.resolve(), weights.resolve(), output.resolve()
    if not (prepared / "pairs.jsonl").is_file() or not weights.is_file():
        raise FileNotFoundError("Prepared pairs or checkpoint missing")
    save_dir = output / variant
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Export output is not empty: {save_dir}")

    class ExportMixin:
        def init_metrics(self, model):
            self.export_rows = []
            return super().init_metrics(model)

        def update_metrics(self, preds, batch):
            for index, pred in enumerate(preds):
                native = self._prepare_pred(pred, self._prepare_batch(index, batch))
                height, width = map(int, batch["ori_shape"][index])
                boxes = []
                for box, conf, cls in zip(native["bboxes"].cpu().tolist(),
                                          native["conf"].cpu().tolist(),
                                          native["cls"].cpu().tolist()):
                    if int(cls) != 0:
                        raise ValueError("Expected one ship class")
                    normalized = [box[0] / width, box[1] / height,
                                  box[2] / width, box[3] / height]
                    boxes.append({"xyxy": normalized, "conf": float(conf)})
                self.export_rows.append({"id": Path(batch["im_file"][index]).stem,
                                         "width": width, "height": height, "boxes": boxes})
            return super().update_metrics(preds, batch)

        def finalize_metrics(self):
            super().finalize_metrics()
            path = self.save_dir / "predictions.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in self.export_rows),
                            encoding="utf-8")

    class StandardExportValidator(ExportMixin, DetectionValidator):
        pass

    class B1ExportValidator(ExportMixin, RGBIRDetectionValidator):
        pass

    class B3RGBExportValidator(ExportMixin, B3RGBValidator):
        pass

    class B3IRExportValidator(ExportMixin, B3IRValidator):
        pass

    validator = {"rgb_only": StandardExportValidator, "ir_only": StandardExportValidator,
                 "b1": B1ExportValidator, "b3_rgb": B3RGBExportValidator,
                 "b3_ir": B3IRExportValidator}[variant]
    data = prepared / ("ir.yaml" if variant == "ir_only" else
                       "rgb.yaml" if variant == "rgb_only" else "fusion.yaml")
    model = YOLO(str(weights))
    if variant == "b1" and type(model.model) is not DualStreamReliabilityDetectionModel:
        raise TypeError("B1 variant requires B1 best.pt")
    if variant.startswith("b3_") and not isinstance(model.model, B3DualHeadDetectionModel):
        raise TypeError("B3 variant requires a B3 native-dual-head checkpoint")
    if variant in ("rgb_only", "ir_only") and isinstance(model.model, DualStreamReliabilityDetectionModel):
        raise TypeError("Single-stream variant requires its own single-stream checkpoint")
    metrics = model.val(validator=validator, data=str(data), project=str(output), name=variant,
                        exist_ok=True, imgsz=imgsz, batch=batch, workers=workers, device=device,
                        conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False)
    rows = [json.loads(line) for line in (save_dir / "predictions.jsonl").read_text().splitlines()]
    expected = {json.loads(line)["id"] for line in (prepared / "pairs.jsonl").read_text().splitlines()}
    if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError(f"Export IDs differ from prepared IDs: {variant}")
    summary = {"variant": variant, "images": len(rows), "detections": sum(len(row["boxes"]) for row in rows),
               "coordinate": "ir" if variant in ("ir_only", "b3_ir") else "rgb",
               "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
               "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
               "weights": str(weights), "imgsz": imgsz, "nms_conf": 0.001,
               "nms_iou": 0.7, "max_det": 300}
    (save_dir / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--conditions", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--limit", type=int, default=0, help="Only for local smoke tests")
    prep.add_argument("--review-images", action="store_true")
    exp = sub.add_parser("export")
    exp.add_argument("--prepared", type=Path, required=True)
    exp.add_argument("--variant", choices=VARIANTS, required=True)
    exp.add_argument("--weights", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--imgsz", type=int, default=960)
    exp.add_argument("--batch", type=int, default=2)
    exp.add_argument("--workers", type=int, default=2)
    exp.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.dataset, args.conditions, args.output, args.limit, args.review_images)
    else:
        result = export(args.prepared, args.variant, args.weights, args.output,
                        args.imgsz, args.batch, args.workers, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
