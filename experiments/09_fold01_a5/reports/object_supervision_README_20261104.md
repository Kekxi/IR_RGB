# 对象级监督资产：来源、语义和使用边界

`object_supervision_20261104.csv` 以 `(id, ir_gt_index)` 为唯一键，把原对应表的**非自动**记录、WP3 H1/H2 人审及 outer 身份人审合并。`object_supervision_overrides_20261104.csv` 保存高优先级人工核对覆盖旧记录的每一行，未改写原文件。事后的 34 页失败判断只作为 outer 行上的 `posthoc_*` 诊断字段，不能用于训练、阈值或主表回算。

**真值版本必须按行区分。** 549 行既有对应/WP3 标签的框索引指向 `b3_after_rgb_boxes_20261029/pairs.jsonl`，其中包含人工补过的 RGB 框；120 行 outer 新审标签索引指向制作 outer PDF 时的原始 YOLO 标签。统一表的 `gt_reference` 标明版本，全部 669 行已按各自版本核验索引有效。不能拿原始 YOLO 标签的框数验证 B3 快照中的 R 索引，也不能在不同快照间直接比较索引号。

## 逐行对账

原 `correspondence.csv` 共 2342 行，其中 1888 行带 `AUTO_DRAFT_UNREVIEWED`，**全部排除**；剩 454 行非自动记录。WP3 的 92 个案例展开为 142 个 IR 对象（H1 100、H2 42）；outer 人审 120 个。发生 47 次同键覆盖，最终 **669 个唯一对象**：train 209、inner_val 335、outer_eval 125。47 次覆盖中 RGB GT 身份索引未改变，均为可观测状态的细化或纠正；其中旧 `ir_only_common_fov` 被明确细分为视场内不可辨认 31、视场内缺 GT 框 4、视场外 4、边界不确定 3、原因未指定 1。其余 4 次将已有 shared 对应升级为双侧可见人工确认。完整逐行差异见覆盖表。

| 切分 | 唯一对象 | 主来源 | 标签覆盖的事件 |
|---|---:|---|---:|
| train | 209 | 67 旧非自动、100 H1 双船、42 H2 | **3/9** 个训练事件 |
| inner_val | 335 | 旧非自动用户记录 | 2/2 个开发事件 |
| outer_eval | 125 | 5 旧非自动、120 新身份审核 | 3/3 个 outer 事件 |

训练许可仅按**本轮冻结 WP3 方案**记录在 `train_eligible_frozen_wp3`：H1 100 个同船对象与 H2 31 个视场内不可辨认 null，共 131 个 IR 对象；其中 11 个 H2 为中等置信。该列不是未来新模型的自动准入许可。outer 任何行都不能移入本轮训练或校准。train 其余 6 个事件尚无经本资产确认的对象级身份/可观测状态；不可把自动几何预填当作补充真值。

## 状态字段不要混用

- `correspondence_status=matched_gt`：存在人审或旧非自动记录的 RGB GT 索引。旧 `shared_rgb_uncertain` 只表示**RGB 可见质量未判定**，本表保留为 `shared_rgb_quality_uncertain`，不冒充“清晰可见”。
- `observability_state=in_fov_no_annotated_match_unresolved`：只知道当前 RGB 标注**无对应框**，不能据此推出不可辨认。旧 `ir_only_common_fov` 未经追加实图复核者维持此状态。
- `in_fov_unrecognizable`、`outside_fov`、`in_fov_missing_box` 分列；`in_fov_missing_box` 的物理船可存在，只是 RGB GT 漏标，不可当作 null 正例训练。
- `uncertain_boundary` 与 `uncertain` 保持未知。不得因缺另一检测框而自动改成视场外或不可辨认。
- `posthoc_gt_quality` 只说明 outer 失败页的人工判断。第 16 页为 `gt_unrecognizable`，不是已证实的错标；第 7 页已在 2026-11-05 由用户补充确认 `gt_correct_visible`。34 页现全部有答案。不静默删除第 16 页，也不改动冻结测试指标。

字段 `source`、`confidence`、`evidence`、`source_pdf`、`source_page` 提供每个对象的审查来源；`event`、`split`、`camera` 供事件隔离统计。若以后要研究雾档，应另以 F1 已冻结图像代理分数按 `id` 关联，不能把视觉 `haze_fog` 等同于气象 1–2 km 能见度。

## 研究决策

此资产足以清楚支持**显式 null / 无对应的定义**和小样本负结果，却不足以证明一个跨全部相机与航段稳健的学习可观测性分类器。WP3 的训练监督集中在 3 个事件，outer 的 3 个事件已用于一次固定方案检验。当前最稳妥的论文叙述是条件配对、双原生框保留、检测候选缺失的端到端限制与人工状态细分；若新建方法实验，需先定义新的事件级评估口径，不得再次使用已看的 outer 三事件作未接触终测。
