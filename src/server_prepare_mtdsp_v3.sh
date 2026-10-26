#!/usr/bin/env bash
# Clone the frozen v2 dataset, apply three LabelImg-reviewed RGB label files, and make a v3 manifest.
set -euo pipefail
CODE=/work/home/kekexiong/fog_ship_code_20261019
SRC=/work/home/kekexiong/datasets/MTDSP_v2_final_20261019/dataset
DST=/work/home/kekexiong/datasets/MTDSP_v3_labelreview_20261020/dataset
PATCH="$CODE/label_corrections_v3"
MANIFEST="$CODE/manifests_v3"
echo 'MTDSP v3 preparation: resume-safe v5'

[[ -d "$SRC" && -f "$SRC/pairs.csv" ]] || { echo "Missing source dataset: $SRC" >&2; exit 1; }
[[ -f "$PATCH/label_corrections.json" ]] || { echo "Missing LabelImg patch: $PATCH" >&2; exit 1; }
if [[ ! -e "$DST" && ! -e "$MANIFEST" ]]; then
  mkdir -p "$(dirname "$DST")"
  cp -a --reflink=auto "$SRC" "$DST"
elif [[ -d "$DST" && ! -L "$DST" && ( ! -e "$MANIFEST" || -d "$MANIFEST" ) ]]; then
  echo "Existing v3 dataset found; verifying it before resuming preparation."
  if [[ ! -f "$DST/pairs.csv" && -f "$DST/dataset/pairs.csv" ]]; then
    echo "Found a nested dataset at $DST/dataset; inspect this layout before copying more files." >&2
    exit 1
  fi
  # Fill missing files individually; cp -n may skip an existing directory itself.
  python - "$SRC" "$DST" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

src, dst = map(Path, sys.argv[1:])
added = 0
for source in src.rglob("*"):
    relative = source.relative_to(src)
    target = dst / relative
    if source.is_symlink():
        raise RuntimeError(f"Source contains symlink: {relative}")
    if source.is_dir():
        if target.exists() or target.is_symlink():
            assert target.is_dir() and not target.is_symlink(), f"Unsafe destination directory: {relative}"
        else:
            target.mkdir(parents=True)
        continue
    if target.exists() or target.is_symlink():
        assert target.is_file() and not target.is_symlink(), f"Unsafe destination file: {relative}"
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".v3copy_", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    added += 1
print(f"Filled {added} missing files without overwriting existing v3 files.")
PY
else
  echo "Cannot safely resume v3 preparation at $DST or $MANIFEST; inspect both paths." >&2
  exit 1
fi

python - "$SRC" "$DST" "$PATCH" "$CODE/v3_image_repair_log.jsonl" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

src, dst, patch = map(Path, sys.argv[1:4])
repair_log = Path(sys.argv[4])
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

records = json.loads((patch / "label_corrections.json").read_text(encoding="utf-8"))
assert len(records) == 3
assert hashlib.sha256((src / "pairs.csv").read_bytes()).digest() == hashlib.sha256((dst / "pairs.csv").read_bytes()).digest(), "v3 pairs.csv differs from source"
for rec in records:
    name = rec["id"] + ".txt"
    old = src / "labels" / "rgb" / "val" / name
    new = patch / name
    assert hashlib.sha256(old.read_bytes()).hexdigest() == rec["old_sha256"], name
    assert hashlib.sha256(new.read_bytes()).hexdigest() == rec["new_sha256"], name
    assert len(old.read_text().splitlines()) == 3 and len(new.read_text().splitlines()) == 2, name

patched = {Path("labels/rgb/val") / (rec["id"] + ".txt") for rec in records}
for source_file in (src / "labels").rglob("*.txt"):
    relative = source_file.relative_to(src)
    if relative in patched:
        continue
    target = dst / relative
    assert target.is_file() and not target.is_symlink(), f"Missing or linked v3 label: {relative}"
    assert hashlib.sha256(source_file.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest(), f"Unexpected v3 label change: {relative}"
repaired_images = 0
for source_file in (src / "images").rglob("*"):
    if not source_file.is_file():
        continue
    relative = source_file.relative_to(src)
    target = dst / relative
    assert target.is_file() and not target.is_symlink(), f"Missing or linked v3 image: {relative}"
    source_hash = sha256_file(source_file)
    target_hash = sha256_file(target)
    if source_hash != target_hash:
        original_size = target.stat().st_size
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".v3repair_", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copy2(source_file, temporary)
            assert sha256_file(temporary) == source_hash, f"Repair copy failed: {relative}"
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        with repair_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                     "path": relative.as_posix(),
                                     "old_size": original_size,
                                     "source_size": source_file.stat().st_size,
                                     "old_sha256": target_hash,
                                     "source_sha256": source_hash}) + "\n")
        repaired_images += 1
print(f"Verified RGB/IR images against frozen v2; repaired {repaired_images} differing images.")

changed = False
for rec in records:
    name = rec["id"] + ".txt"
    target = dst / "labels" / "rgb" / "val" / name
    assert not target.is_symlink(), f"Refusing to replace symlink: {target}"
    current = hashlib.sha256(target.read_bytes()).hexdigest()
    assert current in (rec["old_sha256"], rec["new_sha256"]), f"Unexpected v3 label content: {name}"
    if current == rec["old_sha256"]:
        staged = target.with_suffix(".txt.v3tmp")
        staged.write_bytes((patch / name).read_bytes())
        staged.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(staged, target)
        changed = True
for cache in (dst / "labels").rglob("*.cache"):
    cache.unlink()
(dst / "label_corrections.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
for rec in records:
    name = rec["id"] + ".txt"
    target = dst / "labels" / "rgb" / "val" / name
    assert hashlib.sha256(target.read_bytes()).hexdigest() == rec["new_sha256"], name
print("Verified three LabelImg corrections; source dataset preserved.")
PY

python "$CODE/prepare_data.py" --dataset "$DST" --output "$MANIFEST"
[[ -f "$MANIFEST/fusion_eligible.yaml" ]] || { echo 'v3 manifest missing' >&2; exit 1; }
python - "$DST" "$MANIFEST" <<'PY'
import sys
from pathlib import Path
import yaml

dst, folder = map(Path, sys.argv[1:])
config = yaml.safe_load((folder / "fusion_eligible.yaml").read_text(encoding="utf-8"))
assert Path(config["path"]).resolve() == dst.resolve(), "Manifest points to another dataset"
for split, expected in (("train", 1266), ("val", 877)):
    listed = Path(config[split])
    assert listed.is_file(), f"Missing {split} list"
    assert len(listed.read_text(encoding="utf-8").splitlines()) == expected, split
print("Verified v3 manifest: train=1266, val=877")
PY
python "$CODE/audit_visibility_conditions.py" --dataset "$DST" \
  --conditions "$CODE/val_conditions_reviewed.csv" --strict
echo "Ready: $DST"
echo "Manifest: $MANIFEST/fusion_eligible.yaml"
