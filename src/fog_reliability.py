"""B1: spatial RGB/IR reliability gates for the existing YOLO11 mid-fusion model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK

from rgb_ir_midfusion import (
    DualStreamDetectionModel,
    RGBIRDetectionTrainer,
    _feature_channels,
    _fusion_layer,
)


class SpatialReliabilityFusion(nn.Module):
    """Predict an RGB weight at every spatial location, then retain B0's 1x1 fusion.

    The five input maps describe branch energy, local contrast, and cross-modal
    disagreement. Last-layer zero initialization makes the initial gate 0.5,
    so this module initially computes exactly the B0 fusion projection.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(5, 8, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.project = _fusion_layer(channels)

    @staticmethod
    def _relative_map(x: torch.Tensor) -> torch.Tensor:
        return x / x.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = x.chunk(2, dim=1)
        rgb_energy = rgb.abs().mean(dim=1, keepdim=True)
        ir_energy = ir.abs().mean(dim=1, keepdim=True)
        rgb_contrast = (rgb - F.avg_pool2d(rgb, 3, stride=1, padding=1)).abs().mean(1, keepdim=True)
        ir_contrast = (ir - F.avg_pool2d(ir, 3, stride=1, padding=1)).abs().mean(1, keepdim=True)
        disagreement = (rgb - ir).abs().mean(dim=1, keepdim=True)
        descriptors = torch.cat(
            [self._relative_map(z) for z in (rgb_energy, ir_energy, rgb_contrast, ir_contrast, disagreement)],
            dim=1,
        )
        rgb_weight = torch.sigmoid(self.gate(descriptors))
        return self.project(torch.cat((2 * rgb_weight * rgb, 2 * (1 - rgb_weight) * ir), dim=1))


class DualStreamReliabilityDetectionModel(DualStreamDetectionModel):
    """The B0 detector with only P3/P4/P5 fusion blocks replaced by B1 gates."""

    def __init__(self, single_stream: DetectionModel):
        super().__init__(single_stream)
        for index, channels in enumerate(_feature_channels(self.model[0]), start=2):
            self.model[index] = SpatialReliabilityFusion(channels)
        self.yaml["architecture"] = "RGB-IR dual YOLO11, P3/P4/P5 spatial reliability gate"


class RGBIRReliabilityTrainer(RGBIRDetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, DualStreamReliabilityDetectionModel):
            return weights.float()
        if weights is None:
            raise ValueError("YOLO11 pretrained weights are required.")
        base = DetectionModel(cfg=cfg, nc=self.data["nc"], ch=3, verbose=verbose and RANK == -1)
        base.load(weights)
        LOGGER.info("Using P3/P4/P5 spatial RGB/IR reliability gates.")
        return DualStreamReliabilityDetectionModel(base)


# Ultralytics checkpoints pickle custom classes by module name. Register before loading.
for custom_class in (SpatialReliabilityFusion, DualStreamReliabilityDetectionModel):
    custom_class.__module__ = ultralytics_tasks.__name__
    setattr(ultralytics_tasks, custom_class.__name__, custom_class)
