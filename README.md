# IR/RGB maritime ship detection experiments

This repository records the RGB–IR ship detection experiments through **2026-11-21**. The main question is whether IR can improve ship detection in haze and fog, including targets that are weak in RGB. The V4 visual haze/fog subset contains 633 of 877 validation pairs. Later stages use an event-isolated Fold01 and additional controlled diagnostics. The Point 2 B0 v4 proxy experiment used two seeds per fold.

[完整实验归档与阶段总结（2026-11-21）](%E5%AE%9E%E9%AA%8C%E5%BD%92%E6%A1%A3%E4%B8%8E%E9%98%B6%E6%AE%B5%E6%80%BB%E7%BB%93_20261121.md) covers all 34 archived experiment entries, including evidence boundaries and pending work.

| Stage | Directory | Main finding |
| --- | --- | --- |
| Early pilots | [00_initial_pilots](experiments/00_initial_pilots/README.md) | Preliminary baselines and OAFA comparison; their data protocols differ from V4. |
| V4 modalities | [01_v4_modality_baselines](experiments/01_v4_modality_baselines/README.md) | IR-only is strong on native IR labels; B1 only slightly improves RGB AP over RGB-only. |
| B1 reliability fusion | [02_b1_reliability](experiments/02_b1_reliability/README.md) | Position-wise RGB/IR gating at P3/P4/P5 has no established stable fog gain. |
| B2 control/small | [03_b2_control_small](experiments/03_b2_control_small/README.md) | Additional small-target branch did not improve fog AP over a matched continuation control. |
| B2 local alignment | [04_b2_align_v4](experiments/04_b2_align_v4/README.md) | Fog recall improved slightly; fog AP and the reviewed RGB-weak/IR-clear subset did not. |
| B3 dual native heads | [05_b3_dual_native](experiments/05_b3_dual_native/README.md) | Independent RGB/IR labels and native outputs work, but the dual-head system trails independent model fusion. |
| B3 P3 ablation | [06_b3_phase2_p3_ablation](experiments/06_b3_phase2_p3_ablation/README.md) | Local RGB→IR P3 transfer has a small count gain over matched controls, not over independent fusion or IR-only native AP. |
| Early YTMS YOLOv8s | [07_ytms_yolov8s](experiments/07_ytms_yolov8s/README.md) | On the reviewed paired development set, Concat led the other dual-stream variants. |
| E0–E3 and A4 | [08_e0_e3_a4](experiments/08_e0_e3_a4/README.md) | Geometry and IR guard checks on calibration and diagnostic development blocks. |
| Fold01 and A5 | [09_fold01_a5](experiments/09_fold01_a5/README.md) | Event-isolated detection, geometry and fog diagnostics, outer audit, and null ablations. |
| Point 1 association | [10_point1_association](experiments/10_point1_association/README.md) | Four cost ratios and training-side review expansion. |
| Point 2 B0 | [11_point2_b0](experiments/11_point2_b0/README.md) | Regional soft-gate experiments v1–v5, including matched-statistic and cross-event controls. |

Python and server scripts for the October B0–B3 stages are in [src](src/README.md); the V4 data protocol and condition annotations are in [data_protocol](data_protocol/README.md). Each experiment folder contains its own README, training configuration where available, machine-readable metrics, and the dated analysis report.

**Evidence boundary:** RGB and IR AP use their respective native labels and are not directly comparable as a fusion gain. The October B3 unique-target counts use provisional correspondences and threshold tuning on validation data. The later Fold01 outer results cover reviewed identities and report association conditional on detection as well as recovery over all reviewed same-ship targets.

The raw RGB/IR images, label files, checkpoint weights, rendered prediction images, raw per-box predictions, and provisional per-object correspondence spreadsheets are not published here. They remain in the local/server study archive. The released metrics and reports document the available results; recreating the cross-modal matching scores from scratch requires those underlying files. Some training curves for B3 phase 2 were not downloaded locally and are therefore absent.

See [FILE_MANIFEST.csv](FILE_MANIFEST.csv) for the original local source and SHA-256 of each copied file. Paths in historical logs and scripts reflect the research environment and require adaptation on another system. No software license is declared by this repository.
