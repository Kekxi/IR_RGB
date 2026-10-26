"""Create a disposable four-pair data copy for local trainer integration checks."""

import csv
import shutil
from pathlib import Path

source = Path(r"E:\HEU_4151\RGB_IR\MTDSP_v2_final_20261019\dataset")
target = Path(r"E:\HEU_4151\RGB_IR\fog_ship_experiments_20261019\integration_check_data")
rows = list(csv.DictReader((source / "pairs.csv").open(encoding="utf-8-sig", newline="")))
selected = {split: [r for r in rows if r["split"] == split and r["same_image_size"] == "1"][:2]
            for split in ("train", "val")}
for split, pair_rows in selected.items():
    for row in pair_rows:
        for column in ("rgb_image", "ir_image", "rgb_label", "ir_label"):
            relative = Path(row[column])
            dst = target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dst)
    (target / f"{split}.txt").write_text(
        "\n".join((target / r["rgb_image"]).as_posix() for r in pair_rows) + "\n", encoding="utf-8")
(target / "tiny.yaml").write_text(
    f"path: {target.as_posix()}\ntrain: {target.as_posix()}/train.txt\n"
    f"val: {target.as_posix()}/val.txt\nchannels: 6\nnames:\n  0: ship\n", encoding="utf-8")
print(target / "tiny.yaml")
