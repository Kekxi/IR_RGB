# B2 local alignment versus matched control on V4

Both variants continue from the same B1 checkpoint for 100 epochs. `control` retains B1 reliability fusion; `align` adds P4 local 5×5 RGB–IR matching, confidence-controlled IR resampling, and confidence input to the fusion gate. `training/` has complete curves and arguments; `evaluation/` has fog/clear group metrics, size recall, and the RGB-weak/IR-clear review summary.

Fog mAP50:95 was **0.79215 control** versus **0.78930 align**. Fog recall rose 0.84 percentage points, but the reviewed distant RGB-weak/IR-clear set had no align-only recoveries. This is a negative result for the intended haze/fog target, despite a tiny all-validation AP increase. See `reports/B2_align_V4_结果分析_20261023.md` for selection bias and label-matching limits.
