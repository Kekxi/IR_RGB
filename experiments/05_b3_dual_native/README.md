# B3 first stage: two native detection heads

B3 retains the B1 fused RGB head supervised by `labels/rgb` and adds an IR neck/head supervised by `labels/ir`. It initializes from V4 B1 and trains 100 epochs. `training/` stores the run curve; `evaluation/native/` stores head-specific full/fog/clear metrics; `evaluation/fog_exports/` stores export summaries for all five comparison systems. The unique-target sensitivity outputs use provisional cross-modal label links, not a verified physical-ship census.

On the fog subset, B3 RGB mAP50:95 was **0.7935** and IR mAP50:95 **0.8800**; IR-only was **0.8824** on native IR labels. At a 0.5 false-positive-per-pair budget, B3 dual-head discovery trailed the independent RGB-only plus IR-only union in both provisional correspondence scenarios. The architecture achieved distinct native heads and losses but did not establish a system-level gain. See the reports for the exact caveats.
