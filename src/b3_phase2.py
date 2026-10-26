"""B3 phase 2: IR-first P3 cross-modal transfer with optional local matching.

The original B3 RGB head is unchanged. The IR head retains its native IR P3
skip path; RGB contributes a bounded, confidence-controlled residual. The
residual gain starts at zero, so both variants initially reproduce B3 exactly.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.utils import LOGGER

from b3_dual_head import B3DualHeadDetectionModel, B3DualHeadTrainer, _neck_forward


class IRFirstP3Transfer(nn.Module):
    """At each IR P3 site, sample nearby RGB features and gate their residual.

    ``radius=4`` searches a 9x9 RGB neighborhood. ``radius=0`` uses the
    same-coordinate RGB feature as a direct-transfer ablation. No IR feature
    is multiplied by match confidence or discarded.
    """

    def __init__(self, channels: int, radius: int = 4, key_dim: int = 16):
        super().__init__()
        if radius < 0 or key_dim < 1:
            raise ValueError("radius and key_dim must be nonnegative/positive")
        self.radius = radius
        self.ir_key = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.rgb_key = nn.Conv2d(channels, key_dim, 1, bias=False)
        with torch.no_grad():
            self.rgb_key.weight.copy_(self.ir_key.weight)
        self.rgb_value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Conv2d(3, 1, 3, padding=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)
        self.gain = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        if radius:
            shift = torch.arange(-radius, radius + 1)
            yy, xx = torch.meshgrid(shift, shift, indexing="ij")
            self.register_buffer("dx", xx.reshape(1, -1, 1, 1).float(), persistent=False)
            self.register_buffer("dy", yy.reshape(1, -1, 1, 1).float(), persistent=False)

    def _align_rgb(self, rgb: torch.Tensor, ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.radius == 0:
            confidence = torch.ones_like(ir[:, :1], dtype=torch.float32)
            return rgb, confidence
        batch, _, height, width = rgb.shape
        side = 2 * self.radius + 1
        count = side * side
        ir_key = F.normalize(self.ir_key(ir).float(), dim=1)
        rgb_key = F.normalize(self.rgb_key(rgb).float(), dim=1)
        patches = F.unfold(rgb_key, kernel_size=side, padding=self.radius)
        patches = patches.reshape(batch, rgb_key.shape[1], count, height, width)
        similarity = (ir_key.unsqueeze(2) * patches).sum(dim=1)
        # Padded candidates must never win at image edges.
        valid = F.unfold(torch.ones((1, 1, height, width), device=rgb.device),
                         kernel_size=side, padding=self.radius).reshape(1, count, height, width) > 0
        similarity = similarity.masked_fill(~valid, -1e4)
        logits = similarity * self.logit_scale.exp().clamp(max=30)
        probability = torch.softmax(logits, dim=1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        max_entropy = valid.sum(dim=1, keepdim=True).float().log().clamp_min(1e-6)
        concentration = (1 - entropy / max_entropy).clamp(0, 1)
        peak_similarity = similarity.amax(dim=1, keepdim=True)
        confidence = concentration * torch.sigmoid(4 * (peak_similarity - 0.1))
        offset_x = (probability * self.dx).sum(dim=1)
        offset_y = (probability * self.dy).sum(dim=1)
        gy, gx = torch.meshgrid(torch.arange(height, device=rgb.device),
                                torch.arange(width, device=rgb.device), indexing="ij")
        grid = torch.stack((2 * (gx.float().unsqueeze(0) + offset_x) / max(width - 1, 1) - 1,
                            2 * (gy.float().unsqueeze(0) + offset_y) / max(height - 1, 1) - 1), dim=-1)
        aligned = F.grid_sample(rgb.float(), grid, mode="bilinear",
                                padding_mode="border", align_corners=True).to(rgb.dtype)
        return aligned, confidence

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        if rgb.shape != ir.shape:
            raise ValueError(f"P3 feature shapes differ: {tuple(rgb.shape)} vs {tuple(ir.shape)}")
        aligned_rgb, confidence = self._align_rgb(rgb, ir)
        rgb_contrast = (aligned_rgb - F.avg_pool2d(aligned_rgb, 3, 1, 1)).abs().mean(1, keepdim=True)
        ir_contrast = (ir - F.avg_pool2d(ir, 3, 1, 1)).abs().mean(1, keepdim=True)
        descriptors = torch.cat((confidence.to(ir.dtype), rgb_contrast, ir_contrast), dim=1)
        strength = confidence.to(ir.dtype) * torch.sigmoid(self.gate(descriptors))
        # This identity skip is essential for IR-only ships with no RGB match.
        return ir + self.gain.tanh() * strength * self.rgb_value(aligned_rgb)


class B3Phase2DetectionModel(B3DualHeadDetectionModel):
    """Native B3 heads plus IR-first RGB-to-IR transfer at P3 only."""

    @classmethod
    def from_b3(cls, source: B3DualHeadDetectionModel, radius: int):
        if type(source) is not B3DualHeadDetectionModel:
            raise TypeError("Phase 2 must start from the original B3 phase-1 checkpoint")
        result = deepcopy(source).float()
        result.__class__ = cls
        channels = result.model[2].project.conv.out_channels
        result.ir_p3_transfer = IRFirstP3Transfer(channels, radius=radius)
        result.yaml["architecture"] = f"B3 phase 2: IR-first P3 RGB transfer, radius={radius}"
        result.yaml["ir_p3_transfer_radius"] = radius
        return result

    def _forward_heads(self, x):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected BCHW RGB+IR input with 6 channels, got {tuple(x.shape)}")
        rgb_p3, rgb_p4, rgb_p5 = self.model[0](x[:, :3])
        ir_p3, ir_p4, ir_p5 = self.model[1](x[:, 3:])
        p3 = self.model[2](torch.cat((rgb_p3, ir_p3), dim=1))
        p4 = self.model[3](torch.cat((rgb_p4, ir_p4), dim=1))
        p5 = self.model[4](torch.cat((rgb_p5, ir_p5), dim=1))
        rgb_out = _neck_forward(self.model[5:], p3, p4, p5)
        ir_p3_used = self.ir_p3_transfer(rgb_p3, ir_p3)
        ir_out = _neck_forward(self.ir_neck, ir_p3_used, ir_p4, ir_p5)
        return rgb_out, ir_out


class _B3Phase2Trainer(B3DualHeadTrainer):
    radius = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, B3Phase2DetectionModel):
            if weights.ir_p3_transfer.radius != self.radius:
                raise ValueError("Phase-2 checkpoint has the wrong transfer radius")
            return weights.float()
        if type(weights) is not B3DualHeadDetectionModel:
            raise TypeError("Phase 2 requires the original B3 phase-1 best.pt")
        LOGGER.info("Building IR-first P3 transfer from B3 phase 1, radius=%s", self.radius)
        return B3Phase2DetectionModel.from_b3(weights, radius=self.radius)


class B3P3DirectTrainer(_B3Phase2Trainer):
    radius = 0


class B3P3LocalTrainer(_B3Phase2Trainer):
    radius = 4


# Ultralytics checkpoints pickle these classes under ultralytics.nn.tasks.
for custom_class in (IRFirstP3Transfer, B3Phase2DetectionModel):
    custom_class.__module__ = ultralytics_tasks.__name__
    setattr(ultralytics_tasks, custom_class.__name__, custom_class)
