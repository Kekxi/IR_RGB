"""Reject a debug run with invalid losses or no optimizer update."""

import argparse
import csv
import math
from pathlib import Path

import yaml
from ultralytics import YOLO

import rgb_ir_midfusion  # noqa: F401 - register checkpoint classes
import fog_reliability  # noqa: F401
import b2_variants  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("control", "align", "small"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    results = args.run / "results.csv"
    weights = args.run / "weights" / "last.pt"
    config = args.run / "args.yaml"
    assert results.is_file() and weights.is_file(), "Debug results or checkpoint missing"
    assert config.is_file(), "Debug training configuration missing"
    actual = yaml.safe_load(config.read_text(encoding="utf-8"))
    if args.imgsz is not None:
        assert int(actual["imgsz"]) == args.imgsz, "Debug image size differs from training"
    if args.batch is not None:
        assert int(actual["batch"]) == args.batch, "Debug batch differs from training"
    rows = list(csv.DictReader(results.open(newline="", encoding="utf-8-sig")))
    assert len(rows) == 1, f"Expected one debug epoch, got {len(rows)}"
    loss = {k.strip(): float(v) for k, v in rows[0].items() if "loss" in k.strip()}
    assert loss and all(math.isfinite(v) for v in loss.values()), f"Invalid debug losses: {loss}"
    trained = YOLO(str(weights)).model
    original = YOLO(str(args.b1)).model
    if args.variant == "small":
        assert isinstance(trained, b2_variants.DualStreamSmallTargetDetectionModel)
        assert trained.model[2].pointwise.weight.float().abs().max().item() > 1e-8, "P3 detail branch never updated"
        assert abs(trained.small_center_head.bias.float().mean().item() + 4) > 1e-5, "Center head never updated"
    elif args.variant == "align":
        assert isinstance(trained, b2_variants.DualStreamAlignmentDetectionModel)
        assert isinstance(trained.model[3], b2_variants.ConfidenceAlignmentFusion)
        prior = b2_variants.DualStreamAlignmentDetectionModel.from_b1(original)
        align_before = prior.model[3]
        align_after = trained.model[3]
        changed = [
            (align_after.rgb_key.weight.float() - align_before.rgb_key.weight.float()).abs().max().item(),
            (align_after.ir_key.weight.float() - align_before.ir_key.weight.float()).abs().max().item(),
            (align_after.alignment_logit.float() - align_before.alignment_logit.float()).abs().max().item(),
        ]
        assert max(changed) > 1e-8, "P4 alignment parameters never updated"
    else:
        assert isinstance(trained, fog_reliability.DualStreamReliabilityDetectionModel)
        assert not isinstance(trained, (b2_variants.DualStreamSmallTargetDetectionModel,
                                        b2_variants.DualStreamAlignmentDetectionModel))
    before = original.model[3].project.conv.weight.float()
    after = trained.model[3].project.conv.weight.float()
    assert (after - before).abs().max().item() > 1e-8, "Shared fusion weights never updated"
    print(f"Debug {args.variant} passed: finite losses, checkpoint, optimizer update")


if __name__ == "__main__":
    main()
