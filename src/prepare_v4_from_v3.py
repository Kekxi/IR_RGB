"""Create an independent-label v4 dataset from server v3 and a 28-file patch.

Images are hard-linked when possible, with a copy fallback. Labels are always
independent files. Existing v3 files are never written or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--patch", type=Path, required=True)
    args = p.parse_args()
    src, dst, patch = args.source.resolve(), args.destination.resolve(), args.patch.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be separate dataset directories")
    manifest = json.loads((patch / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["changed"]
    if len(rows) != 28 or len({r["file"] for r in rows}) != 28:
        raise ValueError("Patch must contain exactly 28 unique labels")
    if sha256(src / "pairs.csv") != manifest["pairs_sha256"]:
        raise ValueError("Server v3 pairs.csv is not the patch base")
    for row in rows:
        name = row["file"]
        if Path(name).name != name or not name.endswith(".txt"):
            raise ValueError(f"Unsafe patch filename: {name}")
        if sha256(src / "labels/rgb/val" / name) != row["old_sha256"]:
            raise ValueError(f"Server v3 label differs from patch base: {name}")
        if sha256(patch / name) != row["new_sha256"]:
            raise ValueError(f"Corrupt patch label: {name}")

    created = False
    linked = copied = 0
    if not dst.exists():
        dst.mkdir(parents=True)
        created = True
        for source_file in src.rglob("*"):
            if source_file.is_symlink():
                raise ValueError(f"Source symlink unsupported: {source_file}")
            relative = source_file.relative_to(src)
            target = dst / relative
            if source_file.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source_file.suffix != ".cache":
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative.parts[0] == "images":
                    try:
                        os.link(source_file, target)
                        linked += 1
                        continue
                    except OSError:
                        pass
                shutil.copy2(source_file, target)
                copied += 1
        for row in rows:
            target = dst / "labels/rgb/val" / row["file"]
            target.chmod(target.stat().st_mode | stat.S_IWRITE)
            shutil.copy2(patch / row["file"], target)

    if not dst.is_dir() or sha256(dst / "pairs.csv") != manifest["pairs_sha256"]:
        raise ValueError("Destination is incomplete or has a different pairs.csv")
    revised = {r["file"]: r for r in rows}
    label_count = 0
    for old in (src / "labels").rglob("*.txt"):
        relative = old.relative_to(src)
        target = dst / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        expected = (revised.get(old.name) or {}).get("new_sha256") if relative.parts[:3] == ("labels", "rgb", "val") else None
        if expected is None:
            expected = sha256(old)
        if sha256(target) != expected:
            raise ValueError(f"Unexpected label change: {relative}")
        if os.path.samefile(old, target):
            raise ValueError(f"Destination label is hard-linked to source: {relative}")
        label_count += 1
    print(json.dumps({"created": created, "destination": str(dst), "patched_labels": len(rows),
                      "verified_labels": label_count, "hardlinked_images": linked, "copied_files": copied}, ensure_ascii=False))


if __name__ == "__main__":
    main()
