# IR/RGB maritime ship detection experiments

This repository records the RGB–IR ship detection experiments through **2026-10-26**. The main question is whether IR can improve ship detection in haze and fog, including targets that are weak in RGB. All reported experiments currently use seed 42; the fog subset contains 633 paired images from a 877-pair V4 validation set.

| Stage | Directory | Main finding |
| --- | --- | --- |
| Early pilots | [00_initial_pilots](experiments/00_initial_pilots/README.md) | Preliminary baselines and OAFA comparison; their data protocols differ from V4. |
| V4 modalities | [01_v4_modality_baselines](experiments/01_v4_modality_baselines/README.md) | IR-only is strong on native IR labels; B1 only slightly improves RGB AP over RGB-only. |
| B1 reliability fusion | [02_b1_reliability](experiments/02_b1_reliability/README.md) | Position-wise RGB/IR gating at P3/P4/P5 has no established stable fog gain. |
| B2 control/small | [03_b2_control_small](experiments/03_b2_control_small/README.md) | Additional small-target branch did not improve fog AP over a matched continuation control. |
| B2 local alignment | [04_b2_align_v4](experiments/04_b2_align_v4/README.md) | Fog recall improved slightly; fog AP and the reviewed RGB-weak/IR-clear subset did not. |
| B3 dual native heads | [05_b3_dual_native](experiments/05_b3_dual_native/README.md) | Independent RGB/IR labels and native outputs work, but the dual-head system trails independent model fusion. |
| B3 P3 ablation | [06_b3_phase2_p3_ablation](experiments/06_b3_phase2_p3_ablation/README.md) | Local RGB→IR P3 transfer has a small count gain over matched controls, not over independent fusion or IR-only native AP. |

The runnable Python and server scripts are in [src](src/README.md); the V4 data protocol and condition annotations are in [data_protocol](data_protocol/README.md). Each experiment folder contains its own README, training configuration where available, machine-readable metrics, and the dated analysis report.

**Evidence boundary:** RGB and IR AP use their respective native labels and are not directly comparable as a fusion gain. Cross-modal unique-target counts use provisional correspondences and validation-set threshold tuning; they are exploratory per-frame counts, not independently tested counts of physical ships across video. No multi-seed stability claim has been established.

The raw RGB/IR images, label files, checkpoint weights, rendered prediction images, raw per-box predictions, and provisional per-object correspondence spreadsheets are not published here. They remain in the local/server study archive. The released metrics and reports document the available results; recreating the cross-modal matching scores from scratch requires those underlying files. Some training curves for B3 phase 2 were not downloaded locally and are therefore absent.

See [FILE_MANIFEST.csv](FILE_MANIFEST.csv) for the original local source and SHA-256 of each copied file. Paths in historical logs and scripts reflect the research environment and require adaptation on another system. No software license is declared by this repository.
