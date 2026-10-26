"""Audit the user-filled B2 fog-small CSV without modifying it."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "fog_small_miss_b2_s42_v3"
SOURCE = ROOT / "manual_queue_reviewed.csv"
ALL = ROOT / "all_fog_small_boxes.csv"


def read_csv(path: Path, encoding: str):
    with path.open(encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw = SOURCE.read_bytes()
    try:
        raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "gb18030"
    rows = read_csv(SOURCE, encoding)
    all_rows = read_csv(ALL, "utf-8-sig")
    if len(rows) != 84 or len(all_rows) != 330:
        raise ValueError("Unexpected number of reviewed rows or fog-small GT boxes")
    original = {(r["id"], r["gt_index"]): r for r in all_rows}
    if len(original) != len(all_rows):
        raise ValueError("Duplicate GT identity in original fog-small table")
    immutable = ["event", "rgb_image", "ir_image", "gt_xywh_norm",
                 "control_hit", "small_hit", "comparison"]
    altered = []
    duplicate_keys = []
    seen = set()
    normalized = []
    for line, row in enumerate(rows, 2):
        key = (row["id"], row["gt_index"])
        if key in seen:
            duplicate_keys.append(line)
        seen.add(key)
        base = original.get(key)
        if base is None:
            altered.append((line, "unknown_GT"))
            continue
        altered.extend((line, name) for name in immutable if row[name] != base[name])
        if abs(float(row["long_side_at_960"]) - float(base["long_side_at_960"])) > 1e-8:
            altered.append((line, "long_side_at_960"))
        normalized_row = dict(row)
        normalized_row["source_csv_line"] = line
        if not row["human_distance"].strip() and row["human_visibility"].strip() in {
            "far", "middle", "near", "uncertain"
        }:
            normalized_row["human_distance"] = row["human_visibility"].strip()
            normalized_row["human_visibility"] = ""
        normalized.append(normalized_row)
    if altered or duplicate_keys:
        raise ValueError(f"Model/GT fields changed: {altered}; duplicate identities: {duplicate_keys}")

    normalized_fields = ["source_csv_line", *rows[0].keys()]
    write_csv(ROOT / "manual_queue_normalized_for_followup.csv", normalized, normalized_fields)
    # Focus IR/RGB visibility follow-up on model disagreements and shared far misses.
    priority = [r for r in normalized if r["comparison"] in ("small_only", "control_only")
                or (r["comparison"] == "both_miss" and r["human_distance"] == "far")
                or r["human_cause"] in ("possible_label_issue", "uncertain")]
    write_csv(ROOT / "priority_visibility_followup.csv", priority, normalized_fields)

    cause = Counter(r["human_cause"] for r in normalized)
    distance = Counter(r["human_distance"] for r in normalized)
    model = Counter(r["comparison"] for r in normalized)
    label_lines = [r for r in normalized if r["human_cause"] == "possible_label_issue"]
    uncertain_lines = [r for r in normalized if r["human_cause"] == "uncertain"]
    report = [
        "# B2 雾天小船人工审核表核查", "",
        f"原文件：`manual_queue_reviewed.csv`（检测为 {encoding}）；共 {len(rows)} 行。",
        "模型命中标记、真值 ID/序号/坐标、事件和图像路径均与原始导出记录一致；无重复船框。",
        "原表没有被修改。另存 `manual_queue_normalized_for_followup.csv` 为 UTF-8 BOM：",
        "将 84 行 `human_visibility` 中的 far/middle/near 移入 `human_distance`；"
        "`human_visibility` 留空，待补真正的 RGB/IR 可见性。保留原因和备注。", "",
        "## 当前人工分布（原始 84 行，尚未排除疑似误标）", "",
        f"- 模型结果：{dict(model)}。", f"- 人工距离：{dict(distance)}。",
        f"- 人工原因：{dict(cause)}。", "",
        "`background_confusion/weak_target` 是两种原因并列，归纳时不可重复计为两艘船。",
        "没有填 `alignment_issue` 不等于排除了错位：84 行的 RGB/IR 可见性栏实际上尚未审核。", "",
        "## 需重点核对的原表行", "",
    ]
    for r in label_lines:
        report.append(f"- 第 {r['source_csv_line']} 行：{r['id']} / GT {r['gt_index']}，"
                      f"`possible_label_issue`，备注：{r['human_comment']}。"
                      "请用 LabelImg 查看 RGB 真值框，确认是船、浮标/灯塔或仍不确定。")
    for r in uncertain_lines:
        report.append(f"- 第 {r['source_csv_line']} 行：{r['id']} / GT {r['gt_index']}，"
                      f"`uncertain`，备注：{r['human_comment']}。")
    report += ["", "注意：这 3 张图在 v2→v3 已各删除 1 个非船框；本次标疑问的是 v3 中仍保留的另一个框。"
               "旧删除框中心约在 x=0.44，而本次 3 个疑问框中心约在 x=0.53。"
               "不能因为旧记录已处理，就把本次疑问当成同一误标或自动删除。"]
    report += ["", "## 下一步", "",
               f"优先填写 `priority_visibility_followup.csv` 的 {len(priority)} 行 `human_visibility`："
               "25 条两模型结果不一致、17 条两者都漏检且人工认为远的目标，以及其余不确定项（去重）。",
               "建议值：`rgb_weak_ir_clear` / `both_weak` / `rgb_clear` / `ir_weak` / `uncertain`。"
               "若疑似标注问题，先保留 `possible_label_issue`，不要据此直接删除真值。", "",
               "当前核对和归位不改变 MTDSP v3 标签，也不改变 258/330 与 259/330 的模型诊断计数。", ""]
    (ROOT / "review_audit.md").write_text("\n".join(report), encoding="utf-8")
    print("encoding", encoding, "reviewed", len(rows), "priority", len(priority))
    print("model", model, "distance", distance, "cause", cause)
    print("label lines", [r["source_csv_line"] for r in label_lines])
    print("uncertain lines", [r["source_csv_line"] for r in uncertain_lines])


if __name__ == "__main__":
    main()
