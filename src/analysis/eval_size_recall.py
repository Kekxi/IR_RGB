"""Matched ship recall by input-pixel box size at conf>=0.25, IoU>=0.50."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

from rgb_ir_midfusion import RGBIRDetectionValidator
import fog_reliability  # noqa: F401, checkpoint registration
import b2_variants  # noqa: F401, checkpoint registration


class SizeRecallValidator(RGBIRDetectionValidator):
    last_instance = None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.counts = {"small_le48": [0, 0], "medium_48_128": [0, 0], "large_gt128": [0, 0]}
        type(self).last_instance = self

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        input_h, input_w = batch["img"].shape[-2:]
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"].reshape(-1) == si
            boxes_norm = batch["bboxes"][idx]
            if not len(boxes_norm):
                continue
            widths = boxes_norm[:, 2] * input_w
            heights = boxes_norm[:, 3] * input_h
            max_side = torch.maximum(widths, heights)
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)
            gt = pbatch["bboxes"]
            pred_keep = predn["conf"] >= 0.25
            pred_boxes = predn["bboxes"][pred_keep]
            pred_scores = predn["conf"][pred_keep]
            matched = torch.zeros(len(gt), device=gt.device, dtype=torch.bool)
            if len(pred_boxes):
                ious = box_iou(pred_boxes, gt)
                for pi in pred_scores.argsort(descending=True):
                    values = ious[pi].masked_fill(matched, -1)
                    gi = values.argmax()
                    if values[gi] >= 0.5:
                        matched[gi] = True
            for group, mask in (
                ("small_le48", max_side <= 48),
                ("medium_48_128", (max_side > 48) & (max_side <= 128)),
                ("large_gt128", max_side > 128),
            ):
                self.counts[group][0] += int(matched[mask].sum())
                self.counts[group][1] += int(mask.sum())


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
    model.val(validator=SizeRecallValidator, data=str(args.data.resolve()),
              project=str(args.output.resolve()), name="validation", exist_ok=False,
              imgsz=args.imgsz, batch=args.batch, device=args.device,
              conf=0.25, plots=False)
    counts = SizeRecallValidator.last_instance.counts
    rows = [{"size_input_px": group, "matched": matched, "gt": total,
             "recall_iou50_conf25": matched / total if total else ""}
            for group, (matched, total) in counts.items()]
    with (args.output / "size_recall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
