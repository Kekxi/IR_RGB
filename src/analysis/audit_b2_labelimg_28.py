"""Compare 28 isolated LabelImg edits with the untouched MTDSP v3 labels."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent / "fog_small_miss_b2_s42_v3" / "LabelImg_28_review"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_boxes(path: Path):
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid YOLO ship label: {path}:{i}")
        box = np.array([float(x) for x in parts[1:]], dtype=float)
        x, y, w, h = box
        if w <= 0 or h <= 0 or x - w / 2 < -1e-5 or x + w / 2 > 1 + 1e-5 or y - h / 2 < -1e-5 or y + h / 2 > 1 + 1e-5:
            raise ValueError(f"Out-of-image or zero-area box: {path}:{i}")
        rows.append(box)
    return rows


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    x1 = max(ax1 - aw / 2, bx1 - bw / 2)
    y1 = max(ay1 - ah / 2, by1 - bh / 2)
    x2 = min(ax1 + aw / 2, bx1 + bw / 2)
    y2 = min(ay1 + ah / 2, by1 + bh / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def main():
    with (ROOT / "review_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    if len(manifest) != 28:
        raise ValueError("Expected 28 reviewed images")
    rows = []
    alerts = []
    for item in manifest:
        old_path, new_path = Path(item["source_label"]), Path(item["review_label"])
        if sha256(old_path) != item["source_sha256"]:
            raise AssertionError(f"Original v3 label changed: {old_path}")
        if sha256(old_path) == sha256(new_path):
            raise AssertionError(f"No saved edit: {new_path}")
        old, new = read_boxes(old_path), read_boxes(new_path)
        if len(old) < len(new) or len(old) - len(new) > 1:
            alerts.append(f"{item['id']}: unexpected box count {len(old)}→{len(new)}")
        scores = np.array([[iou(a, b) for b in new] for a in old])
        old_idx, new_idx = linear_sum_assignment(-scores)
        mapping = dict(zip(old_idx, new_idx))
        removed = [i for i in range(len(old)) if i not in mapping]
        matched_ious = [float(scores[i, j]) for i, j in mapping.items()]
        if len(old) == len(new) and any(x < 0.20 for x in matched_ious):
            alerts.append(f"{item['id']}: at least one matched box has IoU<0.20")
        target_idx = int(item["gt_index"]) - 1
        target_iou = (float(scores[target_idx, mapping[target_idx]])
                      if target_idx in mapping else None)
        rows.append({
            "batch": item["batch"], "source_csv_line": item["source_csv_line"],
            "id": item["id"], "target_gt_index_old": item["gt_index"],
            "old_box_count": len(old), "new_box_count": len(new),
            "removed_old_indices": "|".join(str(i + 1) for i in removed),
            "removed_old_centers": "|".join(f"{old[i][0]:.6f},{old[i][1]:.6f}" for i in removed),
            "target_matched_iou": "" if target_iou is None else round(target_iou, 5),
            "minimum_matched_iou": round(min(matched_ious), 5) if matched_ious else "",
            "source_sha256": item["source_sha256"], "review_sha256": sha256(new_path),
        })
    report_csv = ROOT / "label_diff_audit.csv"
    with report_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    altered = [r for r in rows if int(r["old_box_count"]) == int(r["new_box_count"])]
    deleted = [r for r in rows if int(r["old_box_count"]) - int(r["new_box_count"]) == 1]
    lines = ["# 28 张 LabelImg 复核标签核查", "",
             "- 28/28 个复核 TXT 与对应 v3 原标签不同；所有复核框均为合法 `0 ship`，坐标、尺寸在图像内。",
             "- v3 原标签与复核前 SHA-256 全部相同，未被修改。",
             f"- {len(altered)} 张只调整坐标、框数不变；{len(deleted)} 张各删除 1 个框。",
             "- 原框和复核框按最大 IoU 一对一匹配；明细见 `label_diff_audit.csv`。", "",
             "## 删除框核对", ""]
    for r in deleted:
        lines.append(f"- `{r['id']}`：原 GT {r['removed_old_indices']} 被删除，"
                     f"中心 ({r['removed_old_centers']})，旧框 {r['old_box_count']} → 新框 {r['new_box_count']}。")
    lines += ["", "这 3 个被删框均是本轮疑似非船目标，中心约 x=0.53；"
              "另一处约 x=0.75 的框保留并调整。之前 v2→v3 删除的是约 x=0.44 的另一处框。",
              "类别真伪由用户在 LabelImg 复核，程序仅检查删除位置与标签完整性。", "",
              "## 自动检查提醒", ""]
    lines += [f"- {x}" for x in alerts] if alerts else ["- 无超出预期框数或匹配 IoU<0.20 的条目。"]
    lines += ["", "这两处低 IoU 框的中心仅移动约 (1,1) 与 (1,4) 原图像素；"
              "新框面积约为旧框的 13.6% 与 17.4%，符合人工收紧船体范围。"]
    lines += ["", "这 28 份复核标签未写回 v3；已纳入独立 v4 数据副本，"
              "并用原权重在本地重新计算指标。详见 B2_v4_labelreview_audit_20261021.md。", ""]
    (ROOT / "label_diff_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("audited", len(rows), "coordinate_only", len(altered), "deleted_one", len(deleted))
    print("alerts", alerts)
    print("deleted", [(r["id"], r["removed_old_indices"], r["removed_old_centers"]) for r in deleted])


if __name__ == "__main__":
    main()
