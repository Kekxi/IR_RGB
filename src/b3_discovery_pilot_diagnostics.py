"""Inspect which unique targets differ between B3 and independent baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from b3_discovery_score import (_jsonl, _native_matches, _predictions,
                                _read_correspondence, _unique_gt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.pilot_root
    pairs = {row["id"]: row for row in _jsonl(root / "prepared" / "pairs.jsonl")}
    links, audit = _read_correspondence(root / "prepared" / "correspondence_assisted.csv", pairs)
    if not audit["complete"]:
        raise ValueError(audit)
    uid, _ = _unique_gt(pairs, links)
    matches = _native_matches(pairs, _predictions(pairs, root / "exports"), 0.5)
    report = json.loads((root / "scores_assisted" / "unique_ship_report.json").read_text(encoding="utf-8"))
    category = {f"{pair_id}:I{index}": status for (pair_id, index), (rgb_index, status) in links.items()
                if rgb_index is None}

    def hits(variant, threshold):
        modality = "ir" if variant in ("ir_only", "b3_ir") else "rgb"
        return {uid[(pair_id, modality, index)]
                for pair_id, conf, index in matches[variant]
                if conf >= threshold and index is not None}

    comparisons = {}
    for setting in ["fixed_conf025", *[f"budget_{x}" for x in (0.1, 0.25, 0.5)]]:
        rows = (report["fixed_conf025"] if setting == "fixed_conf025" else
                report["same_fppi_budget"][setting.removeprefix("budget_")])
        sets = {}
        for name in ("ir_only", "independent_union", "b3_dual_head"):
            sets[name] = set().union(*(hits(variant, threshold)
                                      for variant, threshold in rows[name]["thresholds"].items()))
        independent = sets["independent_union"]
        b3 = sets["b3_dual_head"]
        comparisons[setting] = {
            "b3_extra_vs_independent": sorted(b3 - independent),
            "independent_extra_vs_b3": sorted(independent - b3),
            "b3_extra_vs_ir_only": sorted(b3 - sets["ir_only"]),
            "ir_only_extra_vs_b3": sorted(sets["ir_only"] - b3),
            "b3_ir_only_found": sorted(value for value in b3 if value in category),
            "independent_ir_only_found": sorted(value for value in independent if value in category),
        }
    output = {"provenance": "assistant_assisted_preliminary", "comparisons": comparisons}
    target = root / "scores_assisted" / "case_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
