# V4 RGB-only, IR-only, and dual-stream B1 baselines

All three models used YOLO11m, seed 42, 960-pixel input, batch 2, and 100 epochs. `training/` stores `args.yaml` and `results.csv`; `evaluation/` stores fog/clear group metrics. `reports/` explains the native coordinate systems and comparisons.

Fog mAP50:95: RGB-only **0.7934** on `labels/rgb`, IR-only **0.8824** on `labels/ir`, dual-stream B1 **0.7949** on `labels/rgb`. The IR-only value is not a fusion improvement over RGB-only because the truth coordinates and label sets differ. The dual-stream B1 gain over RGB-only on RGB labels is about 0.15 percentage points in this single seed. See `reports/V4_单模态双流与B1B2_结果分析_20261023.md` for the full protocol and caveats.
