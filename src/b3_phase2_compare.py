"""Compare B3 phase-2 variants with independent RGB/IR at equal FP budgets.

All cross-modal GT links here are provisional geometry/assistant decisions.
The score unit is one target in one paired frame, not a ship tracked over video.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from b3_discovery_prefill import propose
from b3_discovery_score import _jsonl, score


OVERRIDES = {
    ("mtdsp_01_20250117101938", 0): ("ir_only_common_fov", None),
    ("mtdsp_05_20250117104221", 0): ("shared_rgb_uncertain", 3),
    ("mtdsp_07_20250117110915", 4): ("shared_rgb_uncertain", 3),
    # User confirmed I4/I6 are two distinct IR ships. Their precise RGB link
    # is not material because both IR heads match both native IR boxes.
    ("mtdsp_07_20250117111156", 4): ("ir_only_common_fov", None),
    ("mtdsp_07_20250117111156", 6): ("shared_rgb_uncertain", 4),
    ("mtdsp_07_20250117111838", 1): ("shared_rgb_uncertain", 5),
    ("mtdsp_07_20250117111853", 2): ("shared_rgb_uncertain", 2),
    ("mtdsp_07_20250117111904", 1): ("shared_rgb_uncertain", 2),
    ("mtdsp_13_20250314154739", 0): ("ir_only_common_fov", None),
    ("mtdsp_13_20250314154739", 1): ("shared_rgb_uncertain", 0),
    ("mtdsp_13_20250314154744", 0): ("ir_only_common_fov", None),
    ("mtdsp_13_20250314154744", 1): ("shared_rgb_uncertain", 0),
    ("mtdsp_13_20250314154749", 0): ("ir_only_common_fov", None),
    ("mtdsp_13_20250314154749", 1): ("shared_rgb_uncertain", 0),
}


def correspondences(prepared: Path, output: Path) -> dict[str, Path]:
    pairs = _jsonl(prepared / "pairs.jsonl")
    base = []
    for pair in pairs:
        proposals = propose(pair)
        for index in range(len(pair["gt_ir"])):
            candidate = proposals[index]
            status = candidate["status"]
            rgb = candidate.get("rgb")
            key = (pair["id"], index)
            if key in OVERRIDES:
                status, rgb = OVERRIDES[key]
            base.append({"id": pair["id"], "ir_gt_index": index,
                         "rgb_gt_index": "" if rgb is None else rgb,
                         "status": status, "notes": "ASSISTANT_PROVISIONAL_GEOMETRY",
                         "candidate": candidate.get("candidate"),
                         "distance": candidate.get("distance")})
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for scenario in ("all_unresolved_distinct", "linked_where_possible"):
        rows = [dict(row) for row in base]
        used = defaultdict(set)
        for row in rows:
            if row["status"].startswith("shared_rgb_"):
                rgb = int(row["rgb_gt_index"])
                if rgb in used[row["id"]]:
                    raise ValueError(f"Duplicate RGB link: {row['id']} R{rgb}")
                used[row["id"]].add(rgb)
        unresolved = sorted((row for row in rows if row["status"] == "uncertain"),
                            key=lambda row: row["distance"] or 2.0)
        for row in unresolved:
            candidate = row["candidate"]
            distance = row["distance"]
            if scenario == "linked_where_possible" and candidate is not None and distance is not None \
                    and distance <= 0.15 and candidate not in used[row["id"]]:
                row["status"] = "shared_rgb_uncertain"
                row["rgb_gt_index"] = candidate
                used[row["id"]].add(candidate)
            else:
                row["status"] = "ir_only_common_fov"
                row["rgb_gt_index"] = ""
        path = output / f"{scenario}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "ir_gt_index", "rgb_gt_index", "status", "notes"),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        result[scenario] = path
    return result


def assemble_exports(baseline: Path, phase2: Path, output: Path, model: str) -> Path:
    target = output / "assembled_exports" / model
    source = baseline if model == "original_b3" else phase2 / model
    for variant in ("rgb_only", "ir_only", "b1", "b3_rgb", "b3_ir"):
        folder = target / variant
        folder.mkdir(parents=True, exist_ok=True)
        source_folder = baseline / variant if variant in ("rgb_only", "ir_only", "b1") else source / variant
        for filename in ("predictions.jsonl", "export_summary.json"):
            origin = source_folder / filename
            if not origin.is_file():
                raise FileNotFoundError(origin)
            shutil.copyfile(origin, folder / filename)
    return target


def compare(prepared: Path, baseline: Path, phase2: Path, output: Path) -> dict:
    prepared, baseline, phase2, output = (path.resolve() for path in (prepared, baseline, phase2, output))
    maps = correspondences(prepared, output / "provisional_correspondence")
    models = ("original_b3", "control", "direct", "local")
    results = {}
    for model in models:
        exports = assemble_exports(baseline, phase2, output, model)
        result = {"native": {head: json.loads((exports / f"b3_{head}" / "export_summary.json").read_text(encoding="utf-8"))
                             for head in ("rgb", "ir")}, "scenarios": {}}
        for scenario, correspondence in maps.items():
            report = score(prepared, exports, correspondence,
                           output / "scores" / model / scenario,
                           budgets=[0.25, 0.5, 0.75],
                           correspondence_provenance="assistant_assisted_preliminary")
            if report["status"] != "complete":
                raise RuntimeError(f"Incomplete correspondence: {model}/{scenario}: {report['audit']}")
            result["scenarios"][scenario] = {
                "unique_gt_per_paired_frame": report["unique_gt"],
                "fixed_conf025": {name: {key: row[key] for key in ("unique_tp", "fppi")}
                                  for name, row in report["fixed_conf025"].items()
                                  if name in ("ir_only", "independent_union", "b3_dual_head")},
                "same_fppi_budget": {budget: {name: {key: row[key] for key in ("unique_tp", "fppi", "thresholds")}
                                             for name, row in systems.items()
                                             if name in ("ir_only", "independent_union", "b3_dual_head")}
                                     for budget, systems in report["same_fppi_budget"].items()},
            }
        results[model] = result
        print("Compared", model, flush=True)
    summary = {"provenance": "exploratory automatic/assistant GT correspondence; no human-reviewed ship count",
               "unit": "target in one paired frame; adjacent video frames are not deduplicated",
               "models": results}
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# B3 P3 特征补充：V4 雾天对照", "",
             "本报告对应 633 对雾天图像。跨模态标签对应为自动/助手暂定，结果用于实验筛选，不是论文中的跨视频实体船增益。",
             "阈值在同一验证集内搜索，因此相同误报预算下的结果是描述性上界。", "",
             "| 模型 | RGB mAP50:95 | IR mAP50:95 | ≤0.5 FP/对：全分开假设 | ≤0.5 FP/对：尽量配对假设 |", 
             "| --- | ---: | ---: | ---: | ---: |"]
    for model, result in results.items():
        a = result["scenarios"]["all_unresolved_distinct"]["same_fppi_budget"]["0.5"]["b3_dual_head"]["unique_tp"]
        b = result["scenarios"]["linked_where_possible"]["same_fppi_budget"]["0.5"]["b3_dual_head"]["unique_tp"]
        lines.append(f"| {model} | {result['native']['rgb']['mAP50_95']:.4f} | "
                     f"{result['native']['ir']['mAP50_95']:.4f} | {a} | {b} |")
    reference = results["original_b3"]["scenarios"]
    ir_ap = json.loads((baseline / "ir_only" / "export_summary.json").read_text(encoding="utf-8"))["mAP50_95"]
    ir_a = reference["all_unresolved_distinct"]["same_fppi_budget"]["0.5"]["ir_only"]["unique_tp"]
    ir_b = reference["linked_where_possible"]["same_fppi_budget"]["0.5"]["ir_only"]["unique_tp"]
    lines.append(f"| ir_only | — | {ir_ap:.4f} | {ir_a} | {ir_b} |")
    a = reference["all_unresolved_distinct"]["same_fppi_budget"]["0.5"]["independent_union"]["unique_tp"]
    b = reference["linked_where_possible"]["same_fppi_budget"]["0.5"]["independent_union"]["unique_tp"]
    lines.append(f"| independent_union | — | — | {a} | {b} |")
    lines.extend(["", "优先判断 local 是否同时高于 direct、control、independent_union；并检查 IR 原生 AP 是否高于 IR-only。",
                  "若仅在某一对应假设下略有优势，先核对影响排名的案例，不直接宣称新增实体船。", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-exports", type=Path, required=True)
    parser.add_argument("--phase2-exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.prepared, args.baseline_exports, args.phase2_exports, args.output)
