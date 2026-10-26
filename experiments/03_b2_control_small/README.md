# B2 continuation and small-target ablation

`control` continues B1 without the B2 small-target modification. `small` adds a P3 detail residual and small-ship center auxiliary supervision; the fixed-100-epoch control is the appropriate matched reference. V3 and V4 re-evaluation records are kept separately under `training/` and `evaluation/`. The dated `reports/` document label revisions and why unmatched training chains must not be compared as a module effect.

On the final V4 fog grouping, the `small` variant did not improve mAP50:95 over the matched continued B1 control. The historical reports remain important because some older comparisons used different training duration or validation labels.
