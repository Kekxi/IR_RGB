# V4 data and evaluation protocol

The V4 dataset uses paired RGB/IR images, distinct `labels/rgb` and `labels/ir`, and 877 validation pairs. The visual haze/fog subset contains 633 pairs; the clear subset contains 244. The RGB and IR label counts differ because the two native coordinate systems and visibility conditions differ. Dataset images and full label directories are not included in this public experiment record.

`V4_dataset_README.md` describes the local data version, `val_conditions_reviewed.csv` records the visual condition assignment, and `fog_validation_summary.json` records the prepared fog subset counts. All AP comparisons must state which native label set was used. Cross-modal unique-target figures must additionally state the provisional correspondence and false-positive budget.
