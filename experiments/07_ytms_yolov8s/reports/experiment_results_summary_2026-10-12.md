# RGB/IR 船舶检测实验结果汇总

更新日期：2026-10-12。本文汇总本地现有结果，不启动新训练，也不改动原始数据、标签或既有运行目录。所有 AP 数字均为单类船舶检测的 mAP；表中“点”指绝对百分点，即指标差值乘以 100。

## 结论先行

1. **旧 YTMS 双流实验：**在重新审核的 609/153 对开发划分上，Concat+1×1 的正常输入 mAP50:95 为 **0.42397**。Common-Diff 为 0.41665，Reliability 为 0.38584，Full 为 0.38322。当前共差分和可靠性机制没有超过简单 Concat。
2. **MTDSP 扩充实验：**在同一批 153 对 YTMS 开发验证图像上，直接联合 YTMS 与 MTDSP 预标注训练，相对 YTMS-only 的 RGB mAP50:95 提高 **2.87 点**，IR 提高 **1.85 点**。先用 MTDSP 单独预训练、再联合训练的方案没有优于直接联合。
3. **重要边界：**这些都是 seed 42 的内部开发结果；YTMS 缺真实航次独立测试，MTDSP 的预留图像尚未完成人工真值，RGB/IR 仍无可信跨模态标定。不能据此声称正式泛化性能、可靠性门控有效、双模态创新成立或 OBB 性能改善。

## 1. 数据版本及比较口径

| 版本 | 用途与规模 | 标注及划分边界 |
|---|---|---|
| 旧 YTMS 原始诊断 | 903 对中的清洁开发池 809 对；train/val = 644/165 | 编号代理组，不是真实航次；早期 IR 框主要沿用 RGB 坐标。详见[早期诊断报告](legacy_diagnostic_five_runs_analysis.md)。 |
| YTMS 人工复核 IR | train 审核 644 对后，611 对为 IR-only 候选；后续成对集排除额外 RGB 错框后为 609/153 | RGB 与 IR 使用各自坐标框；153 对成对集以 RGB 坐标评估融合。详见[IR 审核](ir_train_review_results.md)及[成对构建记录](../artifacts/paired_reviewed_dev_v1/build_summary.json)。 |
| MTDSP 官方预标注导出 | 2,145 对完整 RGB/IR；用于扩充训练的 RGB 1,469 张、IR 1,489 张 | `_DLDete.txt` 暂作训练预框，**不是人工核验真值**。3 个 MMSI 的 381 对全部留出；其中 180 对为待人工审核的身份隔离测试候选。详见[导出审计](../artifacts/mtdsp_official_yolo_v1/build_audit.json)与[审核队列冻结](../artifacts/mtdsp_official_yolo_v1/manual_review_queue_freeze.json)。 |

以下三类结果不能混成一个排行榜：早期 644/165 未配准实验、609/153 人工复核成对实验、以及标准 Ultralytics 单模态 MTDSP 扩充实验。它们的训练集、标签坐标或训练实现并不完全相同。

## 2. 旧 YTMS 未配准方法筛查：历史对照

在早期 644/165 开发划分上，独立验证得到：

| 方法 | 正常输入 mAP50:95 | 与 Concat 差值 |
|---|---:|---:|
| [Concat+1×1](../runs/diagnostics/p2_concat/evaluation.json) | **0.41008** | 参考 |
| [Common-Diff A1](../runs/diagnostics/p2_common_diff/evaluation.json) | 0.38511 | −2.50 点 |
| [Reliability A2](../runs/diagnostics/p2_reliability/evaluation.json) | 0.40377 | −0.63 点 |
| [Full A3](../runs/diagnostics/p2_full/evaluation.json) | 0.36670 | −4.34 点 |

Full 在训练同型的合成 IR 模糊下为 0.35171（Concat 0.33072），合成 RGB 低照下为 0.28751（Concat 0.24546），但正常输入下降明显。历史效用目标中 642/644 个样本 RGB 占优、0 个 IR 占优，门控对质量变化几乎没有可测响应。故此阶段只构成机制诊断，不能证明“学会识别传感器失效”。完整分析见[P2–P4 终审](p2_p3_p4_final_analysis.md)。

## 3. YTMS 标签复核与重新训练

### 3.1 标签本身显著改变 IR 评估

同一模型、同一批 155 张 IR 图像，只把原 RGB 转移框换成人工复核 IR 框，mAP50:95 从 [0.22073](../runs/diagnostics/ir_original_same155_local/evaluation.json) 变为 [0.27067](../runs/diagnostics/ir_reviewed_same155_local/evaluation.json)，增加 **4.99 点**。这是**真值标签变化**，不是模型提升。155 张是完整性筛选后的子集，不能直接与旧 165 张全量结果相减。

IR train 审核完成后，644 对中有 611 对候选完整图像、27 对存在未标 IR 船、6 对不确定。随后成对开发集进一步隔离 `T_474`、`T_10672`、`T_160` 三对 RGB 错框，冻结为 **609 train / 153 val**。验证集中特殊图像和模糊小目标的取舍有记录；不能因模型漏检而继续从验证集删图。

### 3.2 人工复核成对开发集：正常输入

以下采用各运行 `best.pt` 在同一 **153 对、RGB 坐标框**上的本地 960 独立验证。只有 seed 42。

| 方法 | Precision | Recall | mAP50 | mAP50:95 | 与 Concat 差值 |
|---|---:|---:|---:|---:|---:|
| [RGB-only](../runs/diagnostics/paired_reviewed_rgb_s42_normal_val_local960/evaluation.json) | 0.70783 | 0.52916 | 0.58438 | 0.27349 | −15.05 点 |
| [Concat+1×1](../runs/diagnostics/paired_reviewed_concat_s42_normal_val_local960/evaluation.json) | **0.87568** | 0.69546 | **0.79861** | **0.42397** | 参考 |
| [Common-Diff A1](../runs/diagnostics/paired_reviewed_common_diff_s42_normal_val_local960/evaluation.json) | 0.79302 | **0.70626** | 0.77973 | 0.41665 | −0.73 点 |
| [Reliability A2](../runs/diagnostics/paired_reviewed_reliability_s42_normal_val_local960/evaluation.json) | 0.79888 | 0.67603 | 0.74781 | 0.38584 | −3.81 点 |
| [Full A3](../runs/diagnostics/paired_reviewed_full_s42_normal_val_local960/evaluation.json) | 0.85678 | 0.65896 | 0.76267 | 0.38322 | −4.07 点 |

Concat 当前仍是最强双流正常条件基线。虽然它高于这轮 RGB-only，但没有已核验 IR→RGB 标定、独立航次测试和多种子置信区间，不能把差值全部归因于真实跨模态互补。IR-only 的复核标签位于 **IR 坐标系**，与本表 RGB 坐标指标不能直接排名。IR-only `best.pt` 在另一个 154 对 v3 子集上的 mAP50:95 为 [0.46220](../runs/diagnostics/ir_reviewed_v3_excluding_T10792_s42_local960/evaluation.json)，只用于该 IR 验证口径。

### 3.3 IR 缺失与工程回退

同一 153 对 RGB 坐标验证集上，IR 缺失时继续运行原融合网络：Concat [0.28474](../runs/diagnostics/paired_reviewed_concat_s42_ir_missing_val_local960/evaluation.json)，Reliability [0.20658](../runs/diagnostics/paired_reviewed_reliability_s42_ir_missing_val_local960/evaluation.json)，Full [0.24975](../runs/diagnostics/paired_reviewed_full_s42_ir_missing_val_local960/evaluation.json)。Full 低于独立 RGB-only 的 0.27349，未通过“缺失时与可用单模态差距不超过 1 点”的预设门槛。

另行实现的 `SensorFallbackRouter` 在 IR 缺失时**显式切换到 RGB-only 检测器**，不是原 Full 网络自行学会回退。RTX 5060、FP16、batch 1、960 方形输入的 153 图验收：Concat 正常前向中位 [14.044 ms](../artifacts/fallback_acceptance/concat_normal_val153_fp16_v1/summary.json)，Full 正常 [17.824 ms](../artifacts/fallback_acceptance/normal_val153_fp16_v1/summary.json)，IR 缺失路由 RGB-only [8.704 ms](../artifacts/fallback_acceptance/ir_missing_val153_fp16_v1/summary.json)。该计时不含磁盘读取、画框和视频编码，不是艇载全链路时延。

## 4. MTDSP 扩充：八组 YOLOv8s 单模态训练

所有运行 seed 42、960 输入、SGD、物理 batch 4。除预训练阶段外，验证都指向**相同的 153 对人工复核 YTMS 开发图像**。下表取 Ultralytics fitness 选中的最佳 epoch；不是最后一轮，也不是 MTDSP 预留测试集的 AP。每个模态各自比较，不把 RGB 与 IR 的不同坐标标签混为同一 GT。

| 模态 | 训练方案 | 最佳/完成 epoch | Precision | Recall | mAP50 | mAP50:95 | 相对 YTMS-only |
|---|---|---:|---:|---:|---:|---:|---:|
| RGB | [仅 YTMS](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_rgb_ytms_only_s42/results.csv) | 97/137 | 0.77444 | 0.65011 | 0.69634 | 0.32161 | 参考 |
| RGB | [COCO 初始化，直接联合](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_rgb_ytms_plus_mtdsp_s42/results.csv) | 72/112 | **0.80784** | **0.68251** | **0.72939** | **0.35033** | **+2.87 点** |
| RGB | [MTDSP 预训练→联合](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_rgb_ytms_plus_mtdsp_finetune_s42/results.csv) | 69/109 | 0.76625 | 0.65227 | 0.68551 | 0.32551 | +0.39 点 |
| IR | [仅 YTMS](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_ir_ytms_only_s42/results.csv) | 83/123 | **0.87798** | 0.76615 | **0.84739** | 0.43716 | 参考 |
| IR | [COCO 初始化，直接联合](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_ir_ytms_plus_mtdsp_s42/results.csv) | 219/220 | 0.84818 | **0.79510** | 0.84594 | **0.45565** | **+1.85 点** |
| IR | [MTDSP 预训练→联合](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_ir_ytms_plus_mtdsp_finetune_s42/results.csv) | 84/124 | 0.86744 | 0.77240 | 0.83903 | 0.43372 | −0.34 点 |

两项 MTDSP-only 预训练都完成 80 epoch，并产生 `training_complete.json` 与 `last.pt`。阶段末在 **YTMS** 验证集上的 mAP50:95 为 [RGB 0.05590](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_rgb_mtdsp_pretrain_s42/results.csv)、[IR 0.10874](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_ir_mtdsp_pretrain_s42/results.csv)；这不是 MTDSP 自身 AP。联合阶段的 [`args.yaml`](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_rgb_ytms_plus_mtdsp_finetune_s42/args.yaml) 和 [`args.yaml`](../runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_ir_ytms_plus_mtdsp_finetune_s42/args.yaml) 均指向对应预训练 `last.pt`，所以两阶段链路确实执行了。

本次观察支持“直接加入 MTDSP 预标注在 YTMS 内部验证上有帮助”；并不支持“先 MTDSP 预训练再联合更好”。原因尚不能归结为伪框错误、域差异或训练日程中的任一项：预训练增加额外更新，联合训练又会提前停止，三种设置的有效优化步数不同。IR 直接联合在第 219/220 轮才达到最佳，值得在冻结评测集和多种子条件下复核，而不是立即宣布收敛优势。

## 5. 目前不能写进论文主结论的内容

- **独立泛化：**YTMS 仍是编号代理分组；MTDSP 留出图像的框还未人工补全。不能将内部验证 AP 写成独立航次、独立船舶或正式外部测试 AP。
- **模态互补与可靠性：**Concat 的当前优势不等于已校准的 RGB/IR 互补；A2/A3 尚无令人信服的质量响应及缺失模态内生回退证据。显式路由是工程功能，不是可学习门控的效果。
- **小目标和 OBB：**旧样本对极小目标覆盖不足；现有全部结果是水平框检测，尚无人工旋转框真值或 OBB 实验。
- **统计确定性：**主要训练只有 seed 42；没有按真实航次/船舶标识重采样的置信区间。当前差值是点估计，不是显著性结论。

## 6. 下一步顺序

1. **优先形成可信评测真值。**从[冻结的 MTDSP 人工审核队列](../artifacts/mtdsp_official_yolo_v1/manual_review_queue.jsonl)先处理 180 对 `identity_heldout_test_candidate`：逐图补全 RGB/IR 船框、记录不可见与不确定目标、确认跨模态同船对应；保留原始自动预框作对照，不覆盖源文件。其余 120 对只做训练伪框质量抽检，不冒充独立测试。
2. **冻结测试协议后再比较。**在相同的人工测试对、相同评估器和输入尺寸上复核 YTMS-only、直接联合、两阶段联合的 RGB-only 与 IR-only `best.pt`。报告按 MMSI/时段和目标大小分层的 AP、Recall、虚警及失败案例；勿用 `_DLDete.txt` 未审框作真值。
3. **确认扩充收益才进入融合。**先检查 MTDSP RGB/IR 时间配对、视场和标定残差，再用可信双坐标标注训练双流 Concat 与新方法。若直接联合的收益稳定，优先把它作为分支初始化/训练数据对照；两阶段预训练暂作负结果或日程消融。
4. **最后做统计与部署。**关键模型至少三种子；以真实独立采集单位而非相邻帧计算配对区间。工程回退另列，不能替代模型本身的鲁棒性证明。OBB 如要探索，先抽样人工试标，不从现有水平框直接推算旋转角度。

本报告是当前结果索引和判断，不替代各运行目录中的原始 `results.csv`、`evaluation.json`、`args.yaml`、混淆矩阵及冻结清单。
