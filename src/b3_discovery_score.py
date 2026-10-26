"""Audit native predictions and count unique ships after human GT correspondence review.

This is an offline, GT-assisted evaluation. It does not implement a deployable
cross-camera alert deduplicator. Threshold sweeps on the evaluated images are
descriptive upper envelopes, not unbiased estimates for future scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from b3_discovery_export import VARIANTS, _iou


SHARED = {"shared_rgb_weak", "shared_rgb_clear", "shared_rgb_uncertain"}
IR_ONLY = {"ir_only_common_fov", "ir_only_outside_rgb_fov"}
SYSTEMS = {"rgb_only": ("rgb_only",), "b1": ("b1",), "ir_only": ("ir_only",),
           "independent_union": ("rgb_only", "ir_only"),
           "b3_dual_head": ("b3_rgb", "b3_ir")}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _predictions(prepared: dict[str, dict], root: Path) -> dict[str, dict[str, dict]]:
    result = {}
    for variant in VARIANTS:
        rows = _jsonl(root / variant / "predictions.jsonl")
        indexed = {row["id"]: row for row in rows}
        if len(rows) != len(indexed) or set(indexed) != set(prepared):
            raise ValueError(f"Missing, duplicate, or extra prediction pair IDs for {variant}")
        expected_coord = "ir" if variant in ("ir_only", "b3_ir") else "rgb"
        summary = json.loads((root / variant / "export_summary.json").read_text(encoding="utf-8"))
        if summary["coordinate"] != expected_coord or summary["images"] != len(prepared):
            raise ValueError(f"Wrong export metadata for {variant}")
        result[variant] = indexed
    return result


def _native_matches(pairs: dict[str, dict], exports: dict[str, dict[str, dict]],
                    min_iou: float) -> dict[str, list[tuple[str, float, int | None]]]:
    """Assign each scored prediction to at most one native GT, greedily by confidence."""
    matched = {}
    for variant, images in exports.items():
        modality = "ir" if variant in ("ir_only", "b3_ir") else "rgb"
        detections = []
        for pair_id, pair in pairs.items():
            gt = pair[f"gt_{modality}"]
            used = set()
            boxes = sorted(images[pair_id]["boxes"], key=lambda row: row["conf"], reverse=True)
            for pred in boxes:
                confidence = float(pred["conf"])
                if not (0 <= confidence <= 1):
                    raise ValueError(f"Invalid confidence: {variant}, {pair_id}")
                box = pred["xyxy"]
                if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    raise ValueError(f"Invalid prediction box: {variant}, {pair_id}")
                candidates = [(float(_iou(box, native_box)), index)
                              for index, native_box in enumerate(gt) if index not in used]
                best = max(candidates, default=(0.0, -1))
                native_index = best[1] if best[0] >= min_iou else None
                if native_index is not None:
                    used.add(native_index)
                detections.append((pair_id, confidence, native_index))
        matched[variant] = detections
    return matched


def _read_correspondence(path: Path, pairs: dict[str, dict]) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(pair_id, index) for pair_id, pair in pairs.items()
                for index in range(len(pair["gt_ir"]))}
    seen = set()
    used_rgb: set[tuple[str, int]] = set()
    errors = []
    links = {}
    for row in rows:
        pair_id = row.get("id", "").strip()
        try:
            ir_index = int(row.get("ir_gt_index", ""))
        except ValueError:
            errors.append(f"Invalid IR index: {pair_id}")
            continue
        key = (pair_id, ir_index)
        if key not in expected or key in seen:
            errors.append(f"Unexpected or duplicate IR GT row: {key}")
            continue
        seen.add(key)
        status = row.get("status", "").strip().lower()
        rgb_text = row.get("rgb_gt_index", "").strip()
        if not status or status == "uncertain":
            continue
        if status == "not_ship_or_duplicate":
            errors.append(f"Fix dataset label before scoring: {key}")
            continue
        if status in SHARED:
            try:
                rgb_index = int(rgb_text)
            except ValueError:
                errors.append(f"Missing RGB index for shared ship: {key}")
                continue
            if not 0 <= rgb_index < len(pairs[pair_id]["gt_rgb"]):
                errors.append(f"RGB index out of range: {key} -> {rgb_index}")
                continue
            rgb_key = (pair_id, rgb_index)
            if rgb_key in used_rgb:
                errors.append(f"One RGB GT linked to multiple IR GTs: {rgb_key}")
                continue
            used_rgb.add(rgb_key)
            links[key] = (rgb_index, status)
        elif status in IR_ONLY:
            if rgb_text:
                errors.append(f"IR-only row has an RGB GT index: {key}")
                continue
            links[key] = (None, status)
        else:
            errors.append(f"Unknown status at {key}: {status}")
    missing_rows = sorted(expected - seen)
    incomplete = sorted(seen - set(links))
    audit = {"expected_ir_rows": len(expected), "submitted_rows": len(rows),
             "completed_rows": len(links), "missing_rows": len(missing_rows),
             "incomplete_rows": len(incomplete), "errors": errors[:30],
             "missing_examples": missing_rows[:20], "incomplete_examples": incomplete[:20],
             "complete": not (missing_rows or incomplete or errors)}
    return links, audit


def _unique_gt(pairs: dict[str, dict], links: dict) -> tuple[dict, Counter]:
    gt_uid = {}
    status: dict[str, str] = {}
    for pair_id, pair in pairs.items():
        for index in range(len(pair["gt_rgb"])):
            uid = f"{pair_id}:R{index}"
            gt_uid[(pair_id, "rgb", index)] = uid
            status[uid] = "rgb_only"
        for index in range(len(pair["gt_ir"])):
            rgb_index, category = links[(pair_id, index)]
            uid = f"{pair_id}:R{rgb_index}" if rgb_index is not None else f"{pair_id}:I{index}"
            gt_uid[(pair_id, "ir", index)] = uid
            status[uid] = category
    return gt_uid, Counter(status.values())


def _at_threshold(detections, threshold, gt_uid, modality):
    hit = set()
    false_positive = 0
    true_reports = 0
    for pair_id, confidence, native_index in detections:
        if confidence < threshold:
            continue
        if native_index is None:
            false_positive += 1
        else:
            true_reports += 1
            if gt_uid is not None:
                hit.add(gt_uid[(pair_id, modality, native_index)])
    return {"hit": hit, "fp": false_positive, "true_reports": true_reports}


def _thresholds():
    return [1.0001] + [step / 100 for step in range(100, -1, -1)]


def _system_result(parts: tuple[str, ...], thresholds: tuple[float, ...],
                   evaluated: dict, uid_status: dict[str, str], n_images: int) -> dict:
    states = [evaluated[(part, threshold)] for part, threshold in zip(parts, thresholds)]
    hit = set().union(*(state["hit"] for state in states))
    fp = sum(state["fp"] for state in states)
    reports = sum(state["true_reports"] for state in states)
    totals = Counter(uid_status.values())
    found = Counter(uid_status[uid] for uid in hit)
    return {"thresholds": dict(zip(parts, thresholds)), "unique_tp": len(hit),
            "unique_gt": len(uid_status), "unique_recall": len(hit) / len(uid_status),
            "false_positives": fp, "fppi": fp / n_images,
            "duplicate_true_reports": reports - len(hit),
            "by_status": {key: {"found": found[key], "total": totals[key]}
                          for key in sorted(totals)}}


def score(prepared_dir: Path, exports_dir: Path, correspondence: Path, output: Path,
          budgets: list[float], min_iou: float = 0.5,
          correspondence_provenance: str = "human_reviewed") -> dict:
    if correspondence_provenance not in {"human_reviewed", "assistant_assisted_preliminary"}:
        raise ValueError("Unknown correspondence provenance")
    if not 0 < min_iou <= 1 or any(value < 0 for value in budgets):
        raise ValueError("Invalid IoU or false-positive budget")
    prepared = _jsonl(prepared_dir / "pairs.jsonl")
    pairs = {row["id"]: row for row in prepared}
    if len(prepared) != len(pairs) or not pairs:
        raise ValueError("Prepared pairs have duplicate IDs or are empty")
    exports = _predictions(pairs, exports_dir)
    matched = _native_matches(pairs, exports, min_iou)
    links, audit = _read_correspondence(correspondence, pairs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "correspondence_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    native = {}
    for variant, detections in matched.items():
        modality = "ir" if variant in ("ir_only", "b3_ir") else "rgb"
        state = _at_threshold(detections, 0.25, None, modality)
        gt_count = sum(len(pair[f"gt_{modality}"]) for pair in pairs.values())
        native[variant] = {"threshold": 0.25, "native_tp": state["true_reports"],
                           "native_gt": gt_count, "native_recall": state["true_reports"] / gt_count,
                           "false_positives": state["fp"], "fppi": state["fp"] / len(pairs)}
    (output / "native_fixed_conf025.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not audit["complete"]:
        return {"status": "correspondence_incomplete", "audit": audit,
                "native_fixed_conf025": native,
                "message": "No unique-ship system scores were computed."}

    gt_uid, counts = _unique_gt(pairs, links)
    uid_status = {}
    for (pair_id, modality, index), uid in gt_uid.items():
        if modality == "rgb":
            uid_status[uid] = "rgb_only"
    for (pair_id, modality, index), uid in gt_uid.items():
        if modality == "ir":
            uid_status[uid] = links[(pair_id, index)][1]
    all_thresholds = _thresholds()
    evaluated = {}
    for variant, detections in matched.items():
        modality = "ir" if variant in ("ir_only", "b3_ir") else "rgb"
        for threshold in all_thresholds:
            evaluated[(variant, threshold)] = _at_threshold(detections, threshold, gt_uid, modality)
    fixed = {name: _system_result(parts, (0.25,) * len(parts), evaluated, uid_status, len(pairs))
             for name, parts in SYSTEMS.items()}
    budget_results = {}
    for budget in budgets:
        cap = math.floor(budget * len(pairs))
        per_system = {}
        for name, parts in SYSTEMS.items():
            candidates = ((threshold,) for threshold in all_thresholds) if len(parts) == 1 else (
                (left, right) for left in all_thresholds for right in all_thresholds)
            best = None
            for threshold_tuple in candidates:
                fp = sum(evaluated[(part, threshold)]["fp"]
                         for part, threshold in zip(parts, threshold_tuple))
                if fp > cap:
                    continue
                hit = set().union(*(evaluated[(part, threshold)]["hit"]
                                    for part, threshold in zip(parts, threshold_tuple)))
                key = (len(hit), -fp)
                if best is None or key > best[0]:
                    best = (key, threshold_tuple)
            assert best is not None
            per_system[name] = _system_result(parts, best[1], evaluated, uid_status, len(pairs))
        budget_results[str(budget)] = per_system
    provenance_note = (
        "Cross-modal GT links were reviewed by a person; this is oracle evaluation, not a deployable prediction deduplicator."
        if correspondence_provenance == "human_reviewed" else
        "Cross-modal GT links are geometry- and assistant-assisted provisional judgments, not human-verified ground truth; these scores are exploratory."
    )
    report = {"status": "complete", "pairs": len(pairs), "min_iou": min_iou,
              "correspondence_provenance": correspondence_provenance,
              "unique_gt": len(uid_status), "gt_by_status": dict(counts),
              "fixed_conf025": fixed, "same_fppi_budget": budget_results,
              "limitations": [provenance_note,
                              "Budget thresholds were selected on the evaluated set, so the envelope is optimistic.",
                              "Adjacent video frames are correlated; use an independent event-level test for a final claim."]}
    (output / "unique_ship_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    heading_note = (
        "以下跨模态去重使用人工核实的真值对应表；这不是可部署的报警去重算法。"
        if correspondence_provenance == "human_reviewed" else
        "以下跨模态对应由几何预填与助手看图辅助判定，尚未经完整人工核实。结果只用于探索和决定下一步实验，不作正式独立船舶覆盖率结论。"
    )
    lines = ["# V4 雾天独立船舶验证", "", f"图像对：{len(pairs)}；逐船真值：{len(uid_status)}；IoU≥{min_iou}。", "",
             heading_note, "",
             "| 系统 | conf=0.25 独立船检出 | 召回 | 误报/图像对 | 重复真阳报告 |",
             "|---|---:|---:|---:|---:|"]
    for name, row in fixed.items():
        lines.append(f"| {name} | {row['unique_tp']}/{row['unique_gt']} | {row['unique_recall']:.3f} | "
                     f"{row['fppi']:.3f} | {row['duplicate_true_reports']} |")
    lines.extend(["", "## 关注目标类别（conf=0.25）", "",
                  "| 系统 | RGB 弱可见的同船检出 | IR 独有且在共同视场的船检出 |",
                  "|---|---:|---:|"])
    for name, row in fixed.items():
        weak = row["by_status"].get("shared_rgb_weak", {"found": 0, "total": 0})
        ir_only = row["by_status"].get("ir_only_common_fov", {"found": 0, "total": 0})
        lines.append(f"| {name} | {weak['found']}/{weak['total']} | "
                     f"{ir_only['found']}/{ir_only['total']} |")
    for budget, rows in budget_results.items():
        lines.extend(["", f"## 同等误报预算：≤{budget} FP/图像对", "",
                      "| 系统 | 独立船检出 | 召回 | 实际误报/图像对 | 阈值 |",
                      "|---|---:|---:|---:|---|"])
        for name, row in rows.items():
            thresholds = ", ".join(f"{key}={value:.2f}" for key, value in row["thresholds"].items())
            lines.append(f"| {name} | {row['unique_tp']}/{row['unique_gt']} | "
                         f"{row['unique_recall']:.3f} | {row['fppi']:.3f} | {thresholds} |")
    lines.extend(["", "阈值在同一验证集上搜索，结果是描述性上界；正式结论需要独立事件测试。", ""])
    (output / "unique_ship_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    result = score(args.prepared, args.exports, args.correspondence,
                   args.output, args.budgets, args.iou)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("fixed_conf025", "same_fppi_budget", "native_fixed_conf025")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
