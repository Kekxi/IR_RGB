# B1 P3/P4/P5 reliability fusion

B1 replaces direct RGB/IR feature fusion at P3/P4/P5 with per-position modality weights learned from feature responses, contrast, and cross-modal differences. The RGB detection head is supervised with `labels/rgb`; the IR backbone receives gradients through that fused RGB-head loss. `training/` contains the original B0/B1 run records, `evaluation/` the group metrics, and `diagnostics/` gate, feature-energy, and modality-mask summaries. The new V4 B1 training curve is also in experiment 01.

The earlier matched B0/B1 fog comparison showed only about **+0.12 percentage points mAP50:95** for B1, with lower recall. The later V4 modality baseline showed **+0.15 percentage points fog mAP50:95** for dual-stream B1 over RGB-only, with different initialization/training. Neither single-seed observation establishes stable fog benefit. See the dated report in `reports/` before citing a number.
