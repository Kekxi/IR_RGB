# MTDSP v4: 28-frame LabelImg review

Independent labels; images are hard links to v3. Training labels and train/val split are unchanged. Only 28 RGB validation labels differ: 25 coordinate adjustments and three one-box removals. Use `manifests/fusion_eligible.yaml` for local evaluation. This version is local and must not be confused with the v3 dataset on the server.
