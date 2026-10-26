"""Package only v3->v4 changed RGB validation labels with SHA-256 provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3", type=Path, required=True)
    p.add_argument("--v4", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    v3, v4, out = a.v3.resolve(), a.v4.resolve(), a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    old = v3 / "labels/rgb/val"
    new = v4 / "labels/rgb/val"
    old_names = {x.name for x in old.glob("*.txt")}
    new_names = {x.name for x in new.glob("*.txt")}
    if old_names != new_names:
        raise ValueError("v3/v4 RGB validation label names differ")
    if (v3 / "pairs.csv").read_bytes() != (v4 / "pairs.csv").read_bytes():
        raise ValueError("v3/v4 pairs.csv differ")
    changed = []
    for name in sorted(old_names):
        before, after = (old / name).read_bytes(), (new / name).read_bytes()
        if before != after:
            changed.append({"file": name, "old_sha256": sha256(before), "new_sha256": sha256(after)})
    if len(changed) != 28:
        raise ValueError(f"Expected exactly 28 changed RGB validation labels, got {len(changed)}")
    out.mkdir(parents=True)
    for row in changed:
        shutil.copy2(new / row["file"], out / row["file"])
    manifest = {"base": "MTDSP_v3_labelreview_20261020", "target": "MTDSP_v4_labelreview_20261021",
                "pairs_sha256": sha256((v3 / "pairs.csv").read_bytes()), "changed": changed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Packaged {len(changed)} changed labels in {out}")


if __name__ == "__main__":
    main()
