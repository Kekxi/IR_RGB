# B3 phase-2 IR P3 ablation: control, direct, local

All three variants continue 40 epochs from the same B3 first-stage checkpoint with the same seed 42 configuration. `control` adds no feature transfer; `direct` gates the same-position RGB P3 feature into the native IR P3 path; `local` searches a 9×9 RGB P3 neighborhood and confidence-gates the residual. The native IR skip path remains intact. The fused RGB head still uses `labels/rgb`; the IR head uses `labels/ir`.

Fog IR mAP50:95 was **0.8809 control**, **0.8796 direct**, **0.8799 local**, versus **0.8824 IR-only**. At ≤0.5 false positives per image pair, `local` reached **2349/2279** provisional unique targets under the two correspondence assumptions, versus **2362/2291** for independent RGB-only plus IR-only fusion. `local` was slightly above control/direct on that count but below IR-only native AP and independent fusion. The full comparison is `evaluation/comparison.md` and `comparison.json`.

The phase-2 training curves/checkpoints were not present in the downloaded local result folder. This repository includes all downloaded phase-2 native and fog metric summaries, but cannot present missing training curves as if they had been verified. See `reports/B3_phase2_V4_结果分析_20261026.md` for detailed interpretation.
