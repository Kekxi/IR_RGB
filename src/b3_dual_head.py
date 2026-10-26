"""B3 phase 1: native RGB and IR heads with independent detection losses.

Pinned to ultralytics 8.3.163, matching the existing V4 experiments.
"""

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel

from fog_reliability import DualStreamReliabilityDetectionModel, RGBIRReliabilityTrainer
from rgb_ir_midfusion import PairedRGBIRDataset, _ir_batch, _rgb_batch


class B3DualLabelDataset(PairedRGBIRDataset):
    """Apply exactly the same spatial transforms to both sets of native boxes.

    IR classes are temporarily offset by ``nc`` while Ultralytics transforms
    the paired six-channel image and boxes. They are split back before collation,
    so neither detection loss ever sees the other modality's target boxes.
    """

    @staticmethod
    def _ir_label_path(rgb_image: str) -> Path:
        image = Path(rgb_image)
        if image.parent.parent.name != "rgb" or image.parent.parent.parent.name != "images":
            raise ValueError(f"Expected dataset/images/rgb/SPLIT/NAME: {image}")
        return image.parents[3] / "labels" / "ir" / image.parent.name / f"{image.stem}.txt"

    def get_labels(self):
        labels = super().get_labels()
        nc = len(self.data["names"])
        for label in labels:
            path = self._ir_label_path(label["im_file"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired IR label: {path}")
            rows = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{path}:{line_number}: expected class and four xywh values")
                rows.append([float(value) for value in fields])
            values = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if len(values):
                cls, xywh = values[:, 0], values[:, 1:]
                if (not np.isfinite(values).all() or (cls != np.floor(cls)).any()
                        or (cls < 0).any() or (cls >= nc).any()
                        or (xywh < 0).any() or (xywh > 1).any()
                        or (xywh[:, 2:] <= 0).any()):
                    raise ValueError(f"Invalid IR YOLO labels: {path}")
            label["ir_cls"] = values[:, :1].copy()
            label["ir_bboxes"] = values[:, 1:].copy()
        return labels

    def update_labels_info(self, label):
        ir_cls = label.pop("ir_cls")
        ir_bboxes = label.pop("ir_bboxes")
        nc = len(self.data["names"])
        label["cls"] = np.concatenate((label["cls"], ir_cls + nc), axis=0)
        label["bboxes"] = np.concatenate((label["bboxes"], ir_bboxes), axis=0)
        return super().update_labels_info(label)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        cls = sample["cls"].reshape(-1, 1)
        bboxes = sample["bboxes"].reshape(-1, 4)
        batch_idx = sample["batch_idx"].reshape(-1)
        ir_mask = cls[:, 0] >= len(self.data["names"])
        sample["ir_cls"] = cls[ir_mask] - len(self.data["names"])
        sample["ir_bboxes"] = bboxes[ir_mask]
        sample["ir_batch_idx"] = batch_idx[ir_mask]
        sample["cls"] = cls[~ir_mask]
        sample["bboxes"] = bboxes[~ir_mask]
        sample["batch_idx"] = batch_idx[~ir_mask]
        return sample

    @staticmethod
    def collate_fn(samples):
        ir_cls = [sample.pop("ir_cls") for sample in samples]
        ir_bboxes = [sample.pop("ir_bboxes") for sample in samples]
        ir_batch_idx = [sample.pop("ir_batch_idx") for sample in samples]
        batch = PairedRGBIRDataset.collate_fn(samples)
        batch["ir_cls"] = torch.cat(ir_cls, 0)
        batch["ir_bboxes"] = torch.cat(ir_bboxes, 0)
        batch["ir_batch_idx"] = torch.cat([value + i for i, value in enumerate(ir_batch_idx)], 0)
        return batch


def _neck_forward(layers, p3, p4, p5):
    """Run the 13 standard YOLO11 neck/head modules on native P3/P4/P5."""
    x = layers[0](p5)
    neck_p4 = layers[2](layers[1]([x, p4]))
    x = layers[3](neck_p4)
    head_p3 = layers[5](layers[4]([x, p3]))
    x = layers[6](head_p3)
    head_p4 = layers[8](layers[7]([x, neck_p4]))
    x = layers[9](head_p4)
    head_p5 = layers[11](layers[10]([x, p5]))
    return layers[12]([head_p3, head_p4, head_p5])


class B3DualHeadDetectionModel(DualStreamReliabilityDetectionModel):
    """Keep B1's RGB fusion head; add an IR-native neck/head without box merging."""

    def __init__(self, b1: DualStreamReliabilityDetectionModel):
        if type(b1) is not DualStreamReliabilityDetectionModel:
            raise TypeError("B3 phase 1 requires an original B1 checkpoint")
        nn.Module.__init__(self)
        self.model = deepcopy(b1.model)
        if len(self.model) != 18:
            raise ValueError("Expected the existing YOLO11m B1 model with 18 modules")
        self.ir_neck = nn.ModuleList(deepcopy(list(b1.model[5:])))
        self.yaml = deepcopy(b1.yaml)
        self.yaml["architecture"] = "B3 phase 1: B1 RGB fusion head + native IR head"
        self.names = deepcopy(b1.names)
        self.nc = b1.nc
        self.inplace = b1.inplace
        self.end2end = b1.end2end
        self.stride = b1.stride.clone()
        self.save = []
        self.task = "detect"
        self.args = deepcopy(b1.args)
        self.inference_head = "ir"  # IR-native AP selects best.pt during training.

    def _forward_heads(self, x):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected BCHW RGB+IR input with 6 channels, got {tuple(x.shape)}")
        rgb_p3, rgb_p4, rgb_p5 = self.model[0](x[:, :3])
        ir_p3, ir_p4, ir_p5 = self.model[1](x[:, 3:])
        p3 = self.model[2](torch.cat((rgb_p3, ir_p3), dim=1))
        p4 = self.model[3](torch.cat((rgb_p4, ir_p4), dim=1))
        p5 = self.model[4](torch.cat((rgb_p5, ir_p5), dim=1))
        rgb_out = _neck_forward(self.model[5:], p3, p4, p5)
        ir_out = _neck_forward(self.ir_neck, ir_p3, ir_p4, ir_p5)
        return rgb_out, ir_out

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        rgb_out, ir_out = self._forward_heads(x)
        if self.inference_head == "rgb":
            return rgb_out
        if self.inference_head == "ir":
            return ir_out
        raise ValueError(f"Unknown inference head: {self.inference_head}")

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        # Validation supplies one head's inference output. Both raw head outputs
        # are needed for separate native losses, so recompute in that case.
        rgb_out, ir_out = self._forward_heads(batch["img"])
        rgb_batch = {
            "batch_idx": batch.get("rgb_batch_idx", batch["batch_idx"]),
            "cls": batch.get("rgb_cls", batch["cls"]),
            "bboxes": batch.get("rgb_bboxes", batch["bboxes"]),
        }
        ir_batch = {
            "batch_idx": batch["ir_batch_idx"],
            "cls": batch["ir_cls"],
            "bboxes": batch["ir_bboxes"],
        }
        rgb_loss, rgb_items = self.criterion(rgb_out, rgb_batch)
        ir_loss, ir_items = self.criterion(ir_out, ir_batch)
        return rgb_loss + ir_loss, torch.cat((rgb_items, ir_items), 0)


def _build_b3_dataset(owner, img_path, mode, batch):
    stride = getattr(owner, "stride", None)
    if stride is None:
        model = getattr(owner, "model", None)
        stride = max(int(de_parallel(model).stride.max() if model else 0), 32)
    return B3DualLabelDataset(
        img_path=img_path,
        imgsz=owner.args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=owner.args,
        rect=owner.args.rect or mode == "val",
        cache=None,
        single_cls=owner.args.single_cls,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task="detect",
        classes=owner.args.classes,
        data=owner.data,
        fraction=owner.args.fraction if mode == "train" else 1.0,
    )


class B3NativeValidator(DetectionValidator):
    HEAD = "ir"

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            de_parallel(trainer.model).inference_head = self.HEAD
            if trainer.ema and trainer.ema.ema is not None:
                trainer.ema.ema.inference_head = self.HEAD
        elif isinstance(model, B3DualHeadDetectionModel):
            model.inference_head = self.HEAD
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path, mode="val", batch=None):
        return _build_b3_dataset(self, img_path, mode, batch or self.args.batch)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if self.HEAD == "ir":
            batch["rgb_batch_idx"] = batch["batch_idx"]
            batch["rgb_cls"] = batch["cls"]
            batch["rgb_bboxes"] = batch["bboxes"]
            batch["batch_idx"] = batch["ir_batch_idx"].to(self.device)
            batch["cls"] = batch["ir_cls"].to(self.device)
            batch["bboxes"] = batch["ir_bboxes"].to(self.device)
        return batch

    def plot_val_samples(self, batch, ni):
        view = _rgb_batch(batch) if self.HEAD == "rgb" else _ir_batch(batch)
        DetectionValidator.plot_val_samples(self, view, ni)

    def plot_predictions(self, batch, preds, ni, max_det=None):
        view = _rgb_batch(batch) if self.HEAD == "rgb" else _ir_batch(batch)
        DetectionValidator.plot_predictions(self, view, preds, ni, max_det=max_det)


class B3RGBValidator(B3NativeValidator):
    HEAD = "rgb"


class B3IRValidator(B3NativeValidator):
    HEAD = "ir"


class B3DualHeadTrainer(RGBIRReliabilityTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return _build_b3_dataset(self, img_path, mode, batch or self.args.batch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, B3DualHeadDetectionModel):
            return weights.float()
        if type(weights) is not DualStreamReliabilityDetectionModel:
            raise TypeError("Training B3 requires the V4 B1 best.pt checkpoint")
        LOGGER.info("Building B3: B1 RGB fusion head plus separately supervised IR-native head.")
        return B3DualHeadDetectionModel(weights)

    def get_validator(self):
        self.loss_names = (
            "rgb_box_loss", "rgb_cls_loss", "rgb_dfl_loss",
            "ir_box_loss", "ir_cls_loss", "ir_dfl_loss",
        )
        return B3IRValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch, ni):
        super().plot_training_samples(batch, ni)
        ir_batch = dict(batch)
        for key in ("cls", "bboxes", "batch_idx"):
            ir_batch[key] = batch[f"ir_{key}"]
        plot_images(
            labels=_ir_batch(ir_batch),
            paths=batch["im_file"],
            fname=self.save_dir / f"train_ir_batch{ni}.jpg",
            on_plot=self.on_plot,
        )


# Checkpoints pickle custom model classes under ultralytics.nn.tasks.
B3DualHeadDetectionModel.__module__ = ultralytics_tasks.__name__
ultralytics_tasks.B3DualHeadDetectionModel = B3DualHeadDetectionModel
