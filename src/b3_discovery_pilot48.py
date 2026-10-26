"""Make and score a fixed 48-pair fog pilot without retraining.

Selection is made from pair IDs alone: nine evenly spaced frames from each of
the five long camera sequences, plus all three frames from camera 13. Model
outputs and label correspondence never influence selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from b3_discovery_export import VARIANTS
from b3_discovery_score import score


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def select_pairs(pairs: list[dict]) -> list[dict]:
    if len(pairs) != 633:
        raise ValueError(f"Expected 633 V4 fog validation pairs, got {len(pairs)}")
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["id"].split("_")[1]].append(pair)
    if set(grouped) != {"01", "03", "05", "07", "09", "13"}:
        raise ValueError(f"Unexpected camera groups: {sorted(grouped)}")
    selected = []
    for camera in sorted(grouped):
        sequence = sorted(grouped[camera], key=lambda pair: pair["id"])
        if camera == "13":
            if len(sequence) != 3:
                raise ValueError("Expected three camera-13 fog pairs")
            selected.extend(sequence)
        else:
            indices = [round(i * (len(sequence) - 1) / 8) for i in range(9)]
            if len(set(indices)) != 9:
                raise ValueError(f"Repeated pilot index in camera {camera}")
            selected.extend(sequence[index] for index in indices)
    if len(selected) != 48:
        raise AssertionError(len(selected))
    return selected


def write_manifests(full_prepared: Path, subset_prepared: Path) -> None:
    dataset_line = next(line for line in (full_prepared / "rgb.yaml")
                        .read_text(encoding="utf-8").splitlines()
                        if line.startswith("path:"))
    dataset = Path(json.loads(dataset_line.removeprefix("path:").strip()))
    pairs = jsonl(subset_prepared / "pairs.jsonl")
    for modality in ("rgb", "ir", "fusion"):
        image_modality = "ir" if modality == "ir" else "rgb"
        manifest = subset_prepared / f"{modality}_val.txt"
        manifest.write_text("\n".join(
            (dataset / pair[f"{image_modality}_image"]).as_posix() for pair in pairs
        ) + "\n", encoding="utf-8")
        yaml_text = (f"path: {json.dumps(dataset.as_posix())}\n"
                     f"train: {json.dumps(manifest.as_posix())}\n"
                     f"val: {json.dumps(manifest.as_posix())}\n"
                     f"channels: {6 if modality == 'fusion' else 3}\n"
                     "names:\n  0: ship\n")
        (subset_prepared / f"{modality}.yaml").write_text(yaml_text, encoding="utf-8")


def prepare(full_prepared: Path, full_draft: Path, subset_root: Path) -> None:
    if subset_root.exists() and any(subset_root.iterdir()):
        raise FileExistsError(f"Pilot output already exists: {subset_root}")
    prepared = subset_root / "prepared"
    images = prepared / "review_images"
    images.mkdir(parents=True)
    selected = select_pairs(jsonl(full_prepared / "pairs.jsonl"))
    ids = {pair["id"] for pair in selected}
    (prepared / "pairs.jsonl").write_text(
        "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in selected),
        encoding="utf-8")
    write_manifests(full_prepared, prepared)
    with full_draft.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["id"] in ids]
        fields = reader.fieldnames
    if len(rows) != sum(len(pair["gt_ir"]) for pair in selected):
        raise ValueError("Pilot correspondence rows do not match selected IR GT")
    # The full draft points to valid absolute local paths. Keep those links so
    # both the original and pilot workbooks open the same source review image.
    with (prepared / "correspondence_review.csv").open(
            "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for pair in selected:
        source = full_prepared / "review_images" / f"{pair['id']}.jpg"
        shutil.copy2(source, images / source.name)
    summary = {
        "selection_rule": "9 evenly spaced pair IDs per camera 01/03/05/07/09, all 3 from camera 13",
        "selection_uses_predictions": False,
        "pairs": len(selected),
        "rgb_gt": sum(len(pair["gt_rgb"]) for pair in selected),
        "ir_gt": len(rows),
        "status": dict(Counter(row["status"] for row in rows)),
        "priority_rows": sum(row["review_priority"] != "spot_check" for row in rows),
        "camera_pairs": dict(Counter(pair["id"].split("_")[1] for pair in selected)),
        "warning": "Exploratory sample from six correlated video sequences; do not treat frames as independent events",
    }
    (subset_root / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_score(full_root: Path, subset_root: Path) -> None:
    prepared = subset_root / "prepared"
    correspondence = prepared / "correspondence_review.csv"
    audit_path = correspondence.with_suffix(".audit.json")
    if not correspondence.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Reviewed correspondence CSV and its audit JSON are required")
    review_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not review_audit.get("formal_human_review") or not review_audit.get("score_audit", {}).get("complete"):
        raise ValueError("Pilot scoring requires all IR boxes to be human checked")
    actual_hash = hashlib.sha256(correspondence.read_bytes()).hexdigest()
    if review_audit.get("csv_sha256") != actual_hash:
        raise ValueError("Reviewed CSV differs from its local audit")
    pairs = jsonl(prepared / "pairs.jsonl")
    ids = {pair["id"] for pair in pairs}
    exports = subset_root / "exports"
    for variant in VARIANTS:
        source = full_root / "exports" / variant
        predictions = jsonl(source / "predictions.jsonl")
        selected = [row for row in predictions if row["id"] in ids]
        if len(selected) != len(ids) or {row["id"] for row in selected} != ids:
            raise ValueError(f"Missing or duplicate pilot predictions: {variant}")
        summary = json.loads((source / "export_summary.json").read_text(encoding="utf-8"))
        summary["images"] = len(selected)
        summary["detections"] = sum(len(row["boxes"]) for row in selected)
        summary["subset_source"] = str(source)
        destination = exports / variant
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
        (destination / "export_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result = score(prepared, exports, correspondence, subset_root / "scores",
                   budgets=[0.1, 0.25, 0.5])
    print(json.dumps({"status": result["status"], "pairs": len(pairs),
                      "scores": str(subset_root / "scores")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--full-prepared", type=Path, required=True)
    prep.add_argument("--full-draft", type=Path, required=True)
    prep.add_argument("--subset-root", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--full-root", type=Path, required=True)
    scoring.add_argument("--subset-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.full_prepared, args.full_draft, args.subset_root)
    else:
        run_score(args.full_root, args.subset_root)


if __name__ == "__main__":
    main()
