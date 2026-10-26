"""Two alternative B2 hypotheses initialized from the same trained B1 detector.

align: local P4 IR-to-RGB matching with match-confidence-aware B1 gating.
small: P3 detail residual and an auxiliary small-ship center objective.
The RGB-coordinate detection head and evaluation labels are unchanged.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.utils import LOGGER

from fog_reliability import DualStreamReliabilityDetectionModel, RGBIRReliabilityTrainer, SpatialReliabilityFusion


class ConfidenceAlignmentFusion(SpatialReliabilityFusion):
    """P4-only local matching; confidence controls spatial transfer, not IR existence."""

    def __init__(self, old: SpatialReliabilityFusion, radius: int = 2, key_dim: int = 32):
        channels = old.project.conv.out_channels
        super().__init__(channels)
        self.radius = radius
        self.rgb_key = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.ir_key = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.match_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.alignment_logit = nn.Parameter(torch.tensor(-2.0))
        self.gate[0] = nn.Conv2d(6, 8, 3, padding=1, bias=True)
        with torch.no_grad():
            self.gate[0].weight.zero_()
            self.gate[0].weight[:, :5].copy_(old.gate[0].weight)
            self.gate[0].bias.copy_(old.gate[0].bias)
            self.gate[-1].weight.copy_(old.gate[-1].weight)
            self.gate[-1].bias.copy_(old.gate[-1].bias)
        self.project.load_state_dict(old.project.state_dict())

        shift = torch.arange(-radius, radius + 1)
        yy, xx = torch.meshgrid(shift, shift, indexing="ij")
        self.register_buffer("dx", xx.reshape(1, -1, 1, 1).float(), persistent=False)
        self.register_buffer("dy", yy.reshape(1, -1, 1, 1).float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, ir = x.chunk(2, dim=1)
        b, _, h, w = rgb.shape
        rgb_key = F.normalize(self.rgb_key(rgb).float(), dim=1)
        ir_key = F.normalize(self.ir_key(ir).float(), dim=1)
        k = (2 * self.radius + 1) ** 2
        patches = F.unfold(ir_key, kernel_size=2 * self.radius + 1, padding=self.radius)
        patches = patches.reshape(b, ir_key.shape[1], k, h, w)
        similarity = (rgb_key.unsqueeze(2) * patches).sum(dim=1)
        probability = torch.softmax(similarity * self.match_logit_scale.exp().clamp(max=30), dim=1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        confidence = (1 - entropy / math.log(k)).clamp(0, 1)
        offset_x = (probability * self.dx).sum(dim=1)
        offset_y = (probability * self.dy).sum(dim=1)
        gy, gx = torch.meshgrid(
            torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
        )
        grid = torch.stack((
            2 * (gx.float().unsqueeze(0) + offset_x) / max(w - 1, 1) - 1,
            2 * (gy.float().unsqueeze(0) + offset_y) / max(h - 1, 1) - 1,
        ), dim=-1)
        aligned_ir = F.grid_sample(ir.float(), grid, mode="bilinear", padding_mode="border", align_corners=True).to(ir.dtype)
        # Low match confidence leaves the original IR branch available. The sixth
        # gate descriptor lets detection loss learn whether to trust that branch.
        strength = self.alignment_logit.sigmoid() * confidence.to(ir.dtype)
        ir_used = ir + strength * (aligned_ir - ir)

        rgb_energy = rgb.abs().mean(1, keepdim=True)
        ir_energy = ir_used.abs().mean(1, keepdim=True)
        rgb_contrast = (rgb - F.avg_pool2d(rgb, 3, 1, 1)).abs().mean(1, keepdim=True)
        ir_contrast = (ir_used - F.avg_pool2d(ir_used, 3, 1, 1)).abs().mean(1, keepdim=True)
        disagreement = (rgb - ir_used).abs().mean(1, keepdim=True)
        descriptors = torch.cat(
            [self._relative_map(z) for z in (rgb_energy, ir_energy, rgb_contrast, ir_contrast, disagreement)]
            + [confidence.to(rgb.dtype)], dim=1
        )
        rgb_weight = torch.sigmoid(self.gate(descriptors))
        return self.project(torch.cat((2 * rgb_weight * rgb, 2 * (1 - rgb_weight) * ir_used), dim=1))


class SmallTargetDetailFusion(nn.Module):
    """Preserve B1 P3 output at initialization, then learn a local detail residual."""

    def __init__(self, old: SpatialReliabilityFusion):
        super().__init__()
        channels = old.project.conv.out_channels
        self.base = old
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.pointwise.weight)
        self.detail_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.base(x)
        highpass = fused - F.avg_pool2d(fused, 3, stride=1, padding=1)
        return fused + self.detail_scale * self.pointwise(F.silu(self.depthwise(highpass)))


class DualStreamAlignmentDetectionModel(DualStreamReliabilityDetectionModel):
    @classmethod
    def from_b1(cls, source: DualStreamReliabilityDetectionModel):
        result = deepcopy(source).float()
        result.__class__ = cls
        result.model[3] = ConfidenceAlignmentFusion(result.model[3])
        result.yaml["architecture"] = "B1 + P4 confidence-aware IR-to-RGB local alignment"
        return result


class DualStreamSmallTargetDetectionModel(DualStreamReliabilityDetectionModel):
    @classmethod
    def from_b1(cls, source: DualStreamReliabilityDetectionModel):
        result = deepcopy(source).float()
        result.__class__ = cls
        channels = result.model[2].project.conv.out_channels
        result.model[2] = SmallTargetDetailFusion(result.model[2])
        result.small_center_head = nn.Conv2d(channels, 1, 1)
        nn.init.normal_(result.small_center_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(result.small_center_head.bias, -4.0)
        result.small_loss_weight = 0.25
        result.yaml["architecture"] = "B1 + P3 detail + small-ship center supervision"
        return result

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected BCHW RGB+IR input, got {tuple(x.shape)}")
        rgb_p3, rgb_p4, rgb_p5 = self.model[0](x[:, :3])
        ir_p3, ir_p4, ir_p5 = self.model[1](x[:, 3:])
        p3 = self.model[2](torch.cat((rgb_p3, ir_p3), dim=1))
        p4 = self.model[3](torch.cat((rgb_p4, ir_p4), dim=1))
        p5 = self.model[4](torch.cat((rgb_p5, ir_p5), dim=1))
        self._small_logits = self.small_center_head(p3) if self.training else None
        x = self.model[5](p5)
        neck_p4 = self.model[7](self.model[6]([x, p4]))
        x = self.model[8](neck_p4)
        head_p3 = self.model[10](self.model[9]([x, p3]))
        x = self.model[11](head_p3)
        head_p4 = self.model[13](self.model[12]([x, neck_p4]))
        x = self.model[14](head_p4)
        head_p5 = self.model[16](self.model[15]([x, p5]))
        return self.model[17]([head_p3, head_p4, head_p5])

    @staticmethod
    def _small_center_targets(batch, shape, device):
        n, _, h, w = shape
        targets = torch.zeros(shape, device=device, dtype=torch.float32)
        boxes = batch["bboxes"].to(device).float()
        indices = batch["batch_idx"].to(device).long().reshape(-1)
        # Normalized xywh boxes; 48 input pixels is the explicit small-ship cutoff.
        image_h, image_w = batch["img"].shape[-2:]
        for i in range(boxes.shape[0]):
            cx, cy, bw, bh = boxes[i]
            if max(float(bw * image_w), float(bh * image_h)) > 48:
                continue
            bi = int(indices[i])
            if bi < 0 or bi >= n:
                continue
            px = min(max(int(cx * w), 0), w - 1)
            py = min(max(int(cy * h), 0), h - 1)
            x0, x1 = max(px - 1, 0), min(px + 2, w)
            y0, y1 = max(py - 1, 0), min(py + 2, h)
            targets[bi, 0, y0:y1, x0:x1] = 1.0
        return targets

    def loss(self, batch, preds=None):
        base_loss, items = super().loss(batch, preds)
        logits = getattr(self, "_small_logits", None)
        if not self.training or logits is None:
            return base_loss, items
        target = self._small_center_targets(batch, logits.shape, logits.device)
        # Balanced focal-style BCE: background pixels cannot swamp small centers.
        probability = logits.float().sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
        positive = target * (1 - probability).pow(2) * bce
        negative = (1 - target) * probability.pow(2) * bce
        auxiliary = (positive.sum() + 0.25 * negative.sum()) / target.sum().clamp_min(1.0)
        self._small_logits = None
        # Ultralytics' detection loss is batch-scaled; preserve its 3-item logging API.
        return base_loss + self.small_loss_weight * auxiliary * batch["img"].shape[0], items


class _B2Trainer(RGBIRReliabilityTrainer):
    variant_class = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, self.variant_class):
            return weights.float()
        if not isinstance(weights, DualStreamReliabilityDetectionModel):
            raise TypeError("B2 requires a trained B1 checkpoint, not yolo11m.pt")
        LOGGER.info("Initializing %s from the trained B1 detector", self.variant_class.__name__)
        return self.variant_class.from_b1(weights)


class RGBIRAlignmentTrainer(_B2Trainer):
    variant_class = DualStreamAlignmentDetectionModel


class RGBIRSmallTargetTrainer(_B2Trainer):
    variant_class = DualStreamSmallTargetDetectionModel


# Ultralytics checkpoints pickle custom classes under ultralytics.nn.tasks.
for custom_class in (
    ConfidenceAlignmentFusion, SmallTargetDetailFusion,
    DualStreamAlignmentDetectionModel, DualStreamSmallTargetDetectionModel,
):
    custom_class.__module__ = ultralytics_tasks.__name__
    setattr(ultralytics_tasks, custom_class.__name__, custom_class)
