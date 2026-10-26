"""RGB/IR paired loader and YOLO11 P3-P5 mid-fusion components."""

from __future__ import annotations

import math
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.patches import imread
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel


BACKBONE_END = 11
FEATURE_LAYERS = (4, 6, 10)


class PairedRGBIRDataset(YOLODataset):
    """Read same-named RGB and IR images and apply synchronized transforms."""

    def load_image(self, i: int, rect_mode: bool = True):
        image = self.ims[i]
        if image is not None:
            return image, self.im_hw0[i], self.im_hw[i]

        rgb_path = Path(self.im_files[i])
        # Expected layout: dataset/images/{rgb,ir}/{train,val}/same_name.jpg.
        if rgb_path.parent.parent.name != "rgb":
            raise ValueError(f"Unexpected RGB path layout: {rgb_path}")
        ir_path = rgb_path.parent.parent.parent / "ir" / rgb_path.parent.name / rgb_path.name
        rgb = imread(str(rgb_path), flags=cv2.IMREAD_COLOR)
        ir = imread(str(ir_path), flags=cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None:
            raise FileNotFoundError(f"Unable to read RGB/IR pair: {rgb_path.name}")
        if ir.ndim == 3:
            ir = ir[..., 0]

        h0, w0 = rgb.shape[:2]
        if ir.shape != (h0, w0):
            raise ValueError(
                f"RGB/IR size mismatch for {rgb_path.name}: "
                f"RGB={w0}x{h0}, IR={ir.shape[1]}x{ir.shape[0]}"
            )

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        image = np.concatenate((rgb, np.repeat(ir[..., None], 3, axis=2)), axis=2)

        if rect_mode:
            ratio = self.imgsz / max(h0, w0)
            if ratio != 1:
                size = (
                    min(math.ceil(w0 * ratio), self.imgsz),
                    min(math.ceil(h0 * ratio), self.imgsz),
                )
                image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = image, (h0, w0), image.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                old = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[old], self.im_hw0[old], self.im_hw[old] = None, None, None

        return image, (h0, w0), image.shape[:2]


class YOLO11Backbone(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        features = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in FEATURE_LAYERS:
                features.append(x)
        return tuple(features)


def _feature_channels(backbone: YOLO11Backbone) -> tuple[int, int, int]:
    training = backbone.training
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.zeros(1, 3, 64, 64))
    backbone.train(training)
    if len(features) != 3:
        raise RuntimeError("The model is not a standard YOLO11 P3/P4/P5 detector.")
    return tuple(int(feature.shape[1]) for feature in features)


def _fusion_layer(channels: int) -> Conv:
    """Concat+1x1 Conv initialized to average corresponding RGB/IR channels."""
    layer = Conv(channels * 2, channels, k=1, s=1, act=False)
    with torch.no_grad():
        layer.conv.weight.zero_()
        index = torch.arange(channels)
        layer.conv.weight[index, index, 0, 0] = 0.5
        layer.conv.weight[index, index + channels, 0, 0] = 0.5
        layer.bn.weight.fill_(1.0)
        layer.bn.bias.zero_()
        layer.bn.running_mean.zero_()
        layer.bn.running_var.fill_(1.0)
    return layer


class DualStreamDetectionModel(DetectionModel):
    """Two YOLO11 backbones, P3/P4/P5 fusion, and one shared neck/head."""

    def __init__(self, single_stream: DetectionModel):
        nn.Module.__init__(self)

        rgb_backbone = YOLO11Backbone(deepcopy(list(single_stream.model[:BACKBONE_END])))
        ir_backbone = YOLO11Backbone(deepcopy(list(single_stream.model[:BACKBONE_END])))
        channels = _feature_channels(rgb_backbone)
        neck_and_head = deepcopy(list(single_stream.model[BACKBONE_END:]))
        if len(neck_and_head) != 13:
            raise RuntimeError("Expected the standard YOLO11 detection neck and head.")

        self.model = nn.ModuleList(
            [
                rgb_backbone,
                ir_backbone,
                *[_fusion_layer(channel) for channel in channels],
                *neck_and_head,
            ]
        )
        self.yaml = deepcopy(single_stream.yaml)
        self.yaml.update(
            channels=6,
            architecture="RGB-IR dual backbone, P3/P4/P5 concat+1x1",
        )
        self.names = deepcopy(single_stream.names)
        self.nc = len(self.names)
        self.inplace = single_stream.inplace
        self.end2end = getattr(single_stream, "end2end", False)
        self.stride = self.model[-1].stride.clone()
        self.save = []
        self.task = "detect"
        self.args: Any = {}

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected BCHW RGB+IR input with 6 channels, got {tuple(x.shape)}")

        rgb_p3, rgb_p4, rgb_p5 = self.model[0](x[:, :3])
        ir_p3, ir_p4, ir_p5 = self.model[1](x[:, 3:])
        p3 = self.model[2](torch.cat((rgb_p3, ir_p3), dim=1))
        p4 = self.model[3](torch.cat((rgb_p4, ir_p4), dim=1))
        p5 = self.model[4](torch.cat((rgb_p5, ir_p5), dim=1))

        x = self.model[5](p5)
        neck_p4 = self.model[7](self.model[6]([x, p4]))
        x = self.model[8](neck_p4)
        head_p3 = self.model[10](self.model[9]([x, p3]))
        x = self.model[11](head_p3)
        head_p4 = self.model[13](self.model[12]([x, neck_p4]))
        x = self.model[14](head_p4)
        head_p5 = self.model[16](self.model[15]([x, p5]))
        return self.model[17]([head_p3, head_p4, head_p5])


class ChannelAttention(nn.Module):
    """CBAM channel attention used by the FCEM-style fusion baseline."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        maximum = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    """CBAM spatial attention used by the FCEM-style fusion baseline."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((avg, maximum), dim=1)))


class FeatureCrossEnhancement(nn.Module):
    """Cross-enhance RGB/IR features, then use B4's 1x1 projection."""

    def __init__(self, channels: int):
        super().__init__()
        self.rgb_channel = ChannelAttention(channels)
        self.ir_channel = ChannelAttention(channels)
        self.rgb_spatial = SpatialAttention()
        self.ir_spatial = SpatialAttention()
        self.project = _fusion_layer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = torch.chunk(x, 2, dim=1)

        cross_channel = self.rgb_channel(rgb) + self.ir_channel(ir)
        rgb = rgb * cross_channel
        ir = ir * cross_channel

        cross_spatial = self.rgb_spatial(rgb) + self.ir_spatial(ir)
        rgb = rgb * cross_spatial
        ir = ir * cross_spatial
        return self.project(torch.cat((rgb, ir), dim=1))


class DualStreamFCEMDetectionModel(DualStreamDetectionModel):
    """B5: B4 dual-stream detector with FCEM-style P3/P4/P5 fusion."""

    def __init__(self, single_stream: DetectionModel):
        super().__init__(single_stream)
        channels = _feature_channels(self.model[0])
        for index, channel in enumerate(channels, start=2):
            self.model[index] = FeatureCrossEnhancement(channel)
        self.yaml["architecture"] = "RGB-IR dual backbone, P3/P4/P5 FCEM-style cross enhancement"


YOLO11Backbone.__module__ = ultralytics_tasks.__name__
DualStreamDetectionModel.__module__ = ultralytics_tasks.__name__
ChannelAttention.__module__ = ultralytics_tasks.__name__
SpatialAttention.__module__ = ultralytics_tasks.__name__
FeatureCrossEnhancement.__module__ = ultralytics_tasks.__name__
DualStreamFCEMDetectionModel.__module__ = ultralytics_tasks.__name__
ultralytics_tasks.YOLO11Backbone = YOLO11Backbone
ultralytics_tasks.DualStreamDetectionModel = DualStreamDetectionModel
ultralytics_tasks.ChannelAttention = ChannelAttention
ultralytics_tasks.SpatialAttention = SpatialAttention
ultralytics_tasks.FeatureCrossEnhancement = FeatureCrossEnhancement
ultralytics_tasks.DualStreamFCEMDetectionModel = DualStreamFCEMDetectionModel


def _build_dataset(owner, img_path: str, mode: str, batch: int):
    stride = getattr(owner, "stride", None)
    if stride is None:
        model = getattr(owner, "model", None)
        stride = max(int(de_parallel(model).stride.max() if model else 0), 32)
    return PairedRGBIRDataset(
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


def _rgb_batch(batch):
    output = dict(batch)
    output["img"] = batch["img"][:, :3]
    return output


def _ir_batch(batch):
    output = dict(batch)
    output["img"] = batch["img"][:, 3:6]
    return output


class RGBIRDetectionValidator(DetectionValidator):
    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return _build_dataset(self, img_path, mode, batch or self.args.batch)

    def plot_val_samples(self, batch, ni: int):
        super().plot_val_samples(_rgb_batch(batch), ni)

        rgb_save_dir = self.save_dir
        try:
            self.save_dir = rgb_save_dir / "ir"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            super().plot_val_samples(_ir_batch(batch), ni)
        finally:
            self.save_dir = rgb_save_dir

    def plot_predictions(self, batch, preds, ni: int, max_det: int | None = None):
        super().plot_predictions(_rgb_batch(batch), preds, ni, max_det=max_det)

        rgb_save_dir = self.save_dir
        try:
            self.save_dir = rgb_save_dir / "ir"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            super().plot_predictions(_ir_batch(batch), preds, ni, max_det=max_det)
        finally:
            self.save_dir = rgb_save_dir


class RGBIRDetectionTrainer(DetectionTrainer):
    def get_dataset(self):
        data = super().get_dataset()
        data["channels"] = 6
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        return _build_dataset(self, img_path, mode, batch or self.args.batch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, DualStreamDetectionModel):
            return weights.float()
        if weights is None:
            raise ValueError("YOLO11 pretrained weights are required.")

        base = DetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=3,
            verbose=verbose and RANK == -1,
        )
        base.load(weights)
        LOGGER.info("Using RGB/IR dual backbones with P3/P4/P5 Concat+1x1 fusion.")
        return DualStreamDetectionModel(base)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return RGBIRDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch, ni: int):
        batch = _rgb_batch(batch)
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )


class RGBIRFCEMDetectionTrainer(RGBIRDetectionTrainer):
    """Trainer for the B5 FCEM-style cross-enhancement fusion model."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, DualStreamFCEMDetectionModel):
            return weights.float()
        if weights is None:
            raise ValueError("YOLO11 pretrained weights are required.")

        base = DetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=3,
            verbose=verbose and RANK == -1,
        )
        base.load(weights)
        LOGGER.info("Using RGB/IR dual backbones with P3/P4/P5 FCEM-style fusion.")
        return DualStreamFCEMDetectionModel(base)
