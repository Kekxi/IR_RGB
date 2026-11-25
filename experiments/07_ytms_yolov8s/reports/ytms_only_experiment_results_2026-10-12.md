# YTMS（RGB/IR）单数据集实验结果总览

更新日期：2026-10-12。本文**只整理 YTMS 历史数据上的训练、人工复核及验证**；不纳入 MTDSP、SMD 或其他数据源的训练结果，也不启动新实验。指标均为单类 `vessel` 水平框检测；“点”表示 AP 等指标的绝对百分点（例如 0.42−0.40 = 2 点）。各表的 Precision、Recall、mAP50、mAP50:95 均取**同一检查点/同一次验证**，不能混用不同 epoch 的单项最高值。

## 1. 结论与比较边界

- 在当前**人工复核成对开发集**（609 对训练、153 对验证）上，同一 153 对、RGB 坐标标签的独立验证中，`Concat+1×1` 的 mAP50:95 为 **0.42397**，高于 RGB-only 的 0.27349、Common-Diff A1 的 0.41665、Reliability A2 的 0.38584 和 Full A3 的 0.38322。A1 比 Concat 低 **0.73 点**，A2 低 **3.81 点**，A3 低 **4.07 点**。
- 早期**未配准诊断集**（644/165 对）也以 Concat 最好；该实验的 IR 标签主要沿用 RGB 框，**不能与复核后的结果合并排名**。
- IR-only 使用 IR 坐标标签。它的 AP 与上述 RGB 坐标的融合表**不直接可比**；同一 155 张 IR 图仅更换评估框，mAP50:95 就从 0.22073 变为 0.27067，说明标签口径本身足以显著影响数值。
- 目前只有 seed 42、编号代理分组，缺真实航次独立测试及核验过的 IR→RGB 标定。不能宣称方法优于强基线、独立泛化成立、识别 USV 身份，或已证明可靠性门控能自主识别失效。

### 1.1 本文涉及的 YTMS 数据版本

| 数据/口径 | 规模与用途 | 关键限制 |
|---|---|---|
| 旧未配准诊断 | 旧 903 对中的清洁开发池 809 对；训练 644、验证 165 | 编号形成的代理组，不是真实航次；IR 大多借用 RGB 坐标框。见[早期实验分析](legacy_diagnostic_five_runs_analysis.md)。 |
| IR 人工复核 | 训练 644 对均经审核，611 对为候选完整图像；验证曾形成 155 对 IR 完整子集 | IR 框在 IR 坐标系；155 对是筛选子集，不是全量 165 对。见[训练审核](../artifacts/ir_train_review/summary.json)。 |
| 冻结成对开发集 | 训练 609 对、验证 153 对；RGB/IR 分别保存各自坐标框 | 进一步隔离 `T_474`、`T_10672`、`T_160` 三对 RGB 错框；融合评估以 RGB 坐标为准，IR-only 以 IR 坐标为准。见[构建摘要](../artifacts/paired_reviewed_dev_v1/build_summary.json)。 |

所有历史结论都属于**内部开发诊断**。没有实测跨相机标定和真实航次留出，因此“成对”不等于“已准确配准”，也不等于独立测试。

## 2. 九种方法具体做了什么

实现位于 [`dual_multimodal/model.py`](../dual_multimodal/model.py) 与 [`dual_multimodal/run.py`](../dual_multimodal/run.py)。双流模式在 YOLOv8s 的 P3/P4/P5 特征处融合，之后接 YOLOv8 检测颈部和头部；此轮没有 P2 或 OBB 实验。

| 方法（代码模式） | 具体操作及其对照意义 |
|---|---|
| RGB-only（`rgb`） | 仅将可见光图像送入单路 YOLOv8s；单模态参考和 IR 缺失时的可用分支。 |
| IR-only（`ir`） | 仅将红外图像送入单路 YOLOv8s；**必须注明所用标签是未核验 RGB 转移框，还是人工 IR 框**。 |
| Early6（`early6`） | 在输入通道把 RGB 3 通道与 IR 3 通道直接拼为 6 通道，由单路骨干提特征；检验像素级早融合。 |
| Concat+1×1（`concat`） | 两路骨干分别提取 P3/P4/P5，逐尺度沿通道拼接，再用 1×1 卷积压回原通道数；最重要的简单双流强基线。 |
| FCEM（`fcem`） | 两路特征拼接后施加通道、空间注意力，再投影融合；检验普通自适应注意力是否足够。此处是项目内的注意力比较实现，不应称为某篇论文的严格复现。 |
| Residual Align / A0（`align`） | 在 IR 特征上预测**受限偏移**作残余对齐，然后执行 Concat+1×1；当前没有实际标定网格输入，不能称“完成物理标定”。 |
| Common-Diff / A1（`common_diff`） | 在 A0 后构造共模 `(R+I′)/2` 与差模 `abs(R−I′)`，连同 RGB、对齐 IR 特征作软融合；有效区外不计算差模。**固定权重**，不含教师效用监督。 |
| Reliability / A2（`reliability`） | 在 A1 上增加空间/尺度可靠性估计；RGB/IR 权重归一化为和为 1，训练时利用冻结单模态教师的相对效用目标及 KL 辅助损失。当前目标的场景区分度需要单独审计。 |
| Full / A3（`full`） | 在 A2 上加入训练期模态缺失及退化：随机 RGB/IR 缺失、RGB 低照、IR 5×5 均值模糊；各约 8% 样本。网络内有显式 availability bypass，但不能把它等同于可靠性估计器“学会识别失效”。 |

## 3. 第一轮：旧未配准 YTMS 644/165 方法筛查

下表直接取各训练 [`results.csv`](../runs/legacy_unregistered_diagnostic_concat_s422/results.csv) 按 Ultralytics fitness 对应的最佳 epoch；**不是最后一个 epoch**。除本地 RGB-only 外，服务器诊断采用 YOLOv8s、960、COCO 初始化、SGD、最多 220 epoch、patience 40、AMP、seed 42；RGB-only 与服务器的物理 batch 不同，但使用相同有效批量设置。目录名 `s422` 是首次失败后的自动重试后缀，成功运行的种子仍为 **42**。

| 方法及原始结果 | 做了什么 | 最佳/完成 epoch | Precision | Recall | mAP50 | mAP50:95 |
|---|---|---:|---:|---:|---:|---:|
| [RGB-only](../runs/legacy_dev_rgb_s42/results.csv) | 仅 RGB 单路基线 | 185/220 | 0.74387 | 0.53645 | 0.63331 | 0.31530 |
| [IR-only 诊断](../runs/legacy_unregistered_diagnostic_ir_s422/results.csv) | 仅 IR，但标签几何未核验 | 22/62 | 0.70819 | 0.54435 | 0.57397 | 0.21644 |
| [Early6](../runs/legacy_unregistered_diagnostic_early6_s422/results.csv) | 输入六通道像素拼接 | 41/81 | 0.72418 | 0.55946 | 0.58902 | 0.25831 |
| [Concat+1×1](../runs/legacy_unregistered_diagnostic_concat_s422/results.csv) | P3/P4/P5 双骨干拼接、1×1 压缩 | 189/220 | 0.80760 | **0.64860** | 0.74094 | **0.40967** |
| [FCEM](../runs/legacy_unregistered_diagnostic_fcem_s422/results.csv) | Concat 前加入通道/空间注意力 | 196/220 | 0.82428 | 0.64486 | 0.73745 | 0.38937 |
| [Residual Align / A0](../runs/legacy_unregistered_diagnostic_align_s422/results.csv) | IR 特征受限残余对齐后 Concat | 194/220 | 0.86607 | 0.61682 | 0.75501 | 0.40125 |
| [Common-Diff / A1](../runs/legacy_unregistered_diagnostic_common_diff_s42/results.csv) | A0 加共模/绝对差模融合 | 155/195 | 0.85026 | 0.62618 | 0.74457 | 0.38493 |
| [Reliability / A2](../runs/legacy_unregistered_diagnostic_reliability_s42/results.csv) | A1 加教师效用监督软门控 | 181/220 | **0.86959** | 0.64810 | **0.76651** | 0.40415 |
| [Full / A3](../runs/legacy_unregistered_diagnostic_full_s42/results.csv) | A2 加缺失/退化样本训练 | 207/220 | 0.83455 | 0.63168 | 0.73032 | 0.36725 |

这轮 Concat 比 RGB-only 高 9.44 点；但 IR 框坐标问题使该差值**不能全部归因于真实 RGB/IR 互补**。A1 比 Concat 低 2.47 点、A2 低 0.55 点、A3 低 4.24 点；Early6 和 IR-only 的较低分数也不能脱离未配准标签讨论。完整收敛、混淆矩阵和图像分析见[九方法详细报告](legacy_diagnostic_five_runs_analysis.md)。

### 3.1 同一旧验证集的 P2 独立复核

下表是独立验证器重新加载 `best.pt` 的结果，因此与上表训练日志的末位小数可能不同；只在**本表内部**计算差值。

| 方法及独立结果 | 做了什么 | Precision | Recall | mAP50 | mAP50:95 | 对 Concat |
|---|---|---:|---:|---:|---:|---:|
| [Concat+1×1](../runs/diagnostics/p2_concat/evaluation.json) | 双流逐尺度拼接基线 | 0.80803 | 0.64860 | 0.74103 | **0.41008** | 参考 |
| [Common-Diff / A1](../runs/diagnostics/p2_common_diff/evaluation.json) | 受限对齐＋共差分支 | 0.85306 | 0.62617 | 0.74471 | 0.38511 | −2.50 点 |
| [Reliability / A2](../runs/diagnostics/p2_reliability/evaluation.json) | A1＋教师效用门控 | **0.86956** | 0.64797 | **0.76649** | 0.40377 | −0.63 点 |
| [Full / A3](../runs/diagnostics/p2_full/evaluation.json) | A2＋缺失/退化训练 | 0.83456 | 0.63174 | 0.72979 | 0.36670 | −4.34 点 |

### 3.2 P3 旧数据合成退化/模态缺失

以下仅是**同一 165 对旧代理验证集**上的合成干扰；RGB 低照是 `0.15×RGB`，IR 模糊是 5×5 均值池化，与 A3 训练扰动同型。表中的数值均为 mAP50:95。方括号链接对应各条件的独立 `evaluation.json`。

| 方法 | 做了什么 | 正常 | IR 模糊 | RGB 低照 | IR 缺失 | RGB 缺失 |
|---|---|---:|---:|---:|---:|---:|
| Concat | 双流 1×1 融合，无专门退化训练 | [0.41008](../runs/diagnostics/p3_concat_normal/evaluation.json) | [0.33072](../runs/diagnostics/p3_concat_ir_blur/evaluation.json) | [0.24546](../runs/diagnostics/p3_concat_rgb_lowlight/evaluation.json) | [0.26472](../runs/diagnostics/p3_concat_ir_missing/evaluation.json) | [0.19690](../runs/diagnostics/p3_concat_rgb_missing/evaluation.json) |
| A2 | A1＋效用监督可靠性门控 | [0.40377](../runs/diagnostics/p3_reliability_normal/evaluation.json) | [0.35111](../runs/diagnostics/p3_reliability_ir_blur/evaluation.json) | [0.24025](../runs/diagnostics/p3_reliability_rgb_lowlight/evaluation.json) | [0.25042](../runs/diagnostics/p3_reliability_ir_missing/evaluation.json) | [0.19884](../runs/diagnostics/p3_reliability_rgb_missing/evaluation.json) |
| A3 | A2＋同型缺失/退化样本训练 | [0.36670](../runs/diagnostics/p3_full_normal/evaluation.json) | [0.35171](../runs/diagnostics/p3_full_ir_blur/evaluation.json) | [0.28751](../runs/diagnostics/p3_full_rgb_lowlight/evaluation.json) | [0.22201](../runs/diagnostics/p3_full_ir_missing/evaluation.json) | [0.22373](../runs/diagnostics/p3_full_rgb_missing/evaluation.json) |

正常单模态参考分别是 [RGB-only 0.31534](../runs/diagnostics/p3_rgb_reference_normal/evaluation.json) 和 [IR-only 0.21598](../runs/diagnostics/p3_ir_reference_normal/evaluation.json)；后者仍受旧 IR 标签影响。A3 的 IR 模糊、RGB 低照绝对 AP 分别高于 Concat 2.10、4.21 点，但正常输入低 4.34 点，且 IR 缺失时低于 RGB-only 9.33 点，**未满足失效回退门槛**。这只显示已知合成扰动下的取舍，不证明真实传感器故障鲁棒性。

### 3.3 P4 教师目标及门控机制审计

按[效用审计](../artifacts/legacy_dev_server/utility_audit.json)，644 张训练图像的效用赢家为 RGB **642**、IR **0**、持平 **2**。A2 的正常前景 RGB 权重约 0.651151，在 RGB 低照/IR 模糊后为 0.651097/0.651168；A3 正常约 0.592068，退化后约 0.592082/0.592053，几乎不响应质量变化。缺失模态下权重变成 0/1 来自显式 availability mask，而非学习到的故障判别。旧 IR 教师又受未核验 RGB→IR 坐标影响，因此不能把这轮监督解释为有效场景自适应效用估计。详细证据见[P2–P4 终审](p2_p3_p4_final_analysis.md)。

## 4. IR 框人工复核：结果为何需要重新评估

训练池 644 对的审核结果：611 对候选完整，27 对发现 IR 中额外未框船，6 对不确定；2275 个 IR 框导出到候选集，**没有改写原图/原始标签**。见[审核汇总](../artifacts/ir_train_review/summary.json)与[导出记录](../artifacts/ir_train_reviewed_v1/export_summary.json)。下表**同一批 155 张图、同一个旧 IR 检查点**，仅真值标签不同。

| 评估口径及结果 | 做了什么 | Precision | Recall | mAP50 | mAP50:95 |
|---|---|---:|---:|---:|---:|
| [原 RGB 框投到 IR 的标签](../runs/diagnostics/ir_original_same155_local/evaluation.json) | 未核验转移框作 IR 真值；同图同模型控制组 | 0.71004 | 0.55061 | 0.58299 | 0.22073 |
| [人工复核 IR 框](../runs/diagnostics/ir_reviewed_same155_local/evaluation.json) | 仅换为人工 IR 坐标框 | 0.74830 | 0.60812 | 0.67930 | 0.27067 |

**+4.99 点不是训练改进，而是标签口径变化。**后续 IR-only 新模型需要在自己的复核 IR 标签上评估。验证集先后形成 v1、v2、v3：其中 v2 为 155 对复核子集，v3 依人工不确定性记录剔除 `T_10792` 后为 154 对；**不能把不同版本 AP 差值当模型增益**。

### 4.1 新 IR-only 模型在不同复核版本上的诊断

新 IR-only 训练目录为 [`legacy_reviewed_ir_s42`](../runs/legacy_reviewed_ir_s42/results.csv)。其在随后冻结的成对 609/153 版本还另有 [`paired_reviewed_ir_s42`](../runs/paired_reviewed_dev_v1/paired_reviewed_ir_s42/results.csv)。以下 v1/v2/v3、输入尺寸变化属于**评估口径或尺寸敏感性**，并非独立多次训练。

| 检查点与验证 | 做了什么 | 图像数 | Precision | Recall | mAP50 | mAP50:95 |
|---|---|---:|---:|---:|---:|---:|
| [旧 IR 新训练，v1/960](../runs/diagnostics/ir_reviewed_s42_local960/evaluation.json) | 复核 IR 框上训练的模型，v1 IR 验证 | 155 | 0.84832 | 0.70490 | 0.79795 | 0.44846 |
| [同检查点，v1/1280](../runs/diagnostics/ir_reviewed_s42_local1280/evaluation.json) | 只将评估输入从 960 改至 1280 | 155 | 0.80273 | 0.65909 | 0.75085 | 0.39737 |
| [同检查点，v2/960](../runs/diagnostics/ir_reviewed_v2_s42_local960/evaluation.json) | v2 人工 IR 框验证 | 155 | 0.84841 | 0.69818 | 0.79316 | 0.44632 |
| [同检查点，v3/960](../runs/diagnostics/ir_reviewed_v3_excluding_T10792_s42_local960/evaluation.json) | 剔除一张存在可辨识船头但标注边界不确定的图 | 154 | 0.85184 | 0.73424 | 0.81871 | 0.46220 |

`v3` 高于 `v2` 不说明模型改善，因为验证图像集合和框数已改变；`1280` 在 v1 上反而低于 `960` **5.11 点**，是本检查点/预处理组合的尺寸诊断，不可泛化为高分辨率一定更差。[训练目录](../runs/legacy_reviewed_ir_s42/results.csv)及[复核验证指南](reviewed_ir_validation.md)保留完整过程。`ir_reviewed_dev_s42` 是另一次验证器复核输出，与 `ir_reviewed_s42_local960` 数值极近，勿当成第二个独立种子。

### 4.2 v2 IR-only 的合成退化

| 评估条件及结果 | 做了什么 | Precision | Recall | mAP50 | mAP50:95 | 对 v2 正常 |
|---|---|---:|---:|---:|---:|---:|
| [正常 v2](../runs/diagnostics/ir_reviewed_v2_s42_local960/evaluation.json) | 155 张 IR 原图 | 0.84841 | 0.69818 | 0.79316 | **0.44632** | 参考 |
| [IR 均值模糊](../runs/diagnostics/ir_v2_s42_blur_synthetic/evaluation.json) | 同图合成 5×5 模糊 | 0.84002 | 0.63395 | 0.75852 | 0.42946 | −1.69 点 |
| [IR 低对比](../runs/diagnostics/ir_v2_s42_lowcontrast_synthetic/evaluation.json) | 同图合成低对比 | 0.69678 | 0.59680 | 0.69170 | 0.36273 | −8.36 点 |
| [RGB 缺失控制](../runs/diagnostics/ir_v2_s42_rgb_missing_control/evaluation.json) | IR-only 本就不读取 RGB；验证条件控制 | 0.84841 | 0.69818 | 0.79316 | 0.44632 | 0.00 点 |

低对比下降大于均值模糊，但这是指定合成变换和筛选验证集上的观察，不能推断所有真实夜间或雾天情况。

## 5. 复核后的 YTMS 成对实验：同一 153 对正常输入

这是与此前[全项目汇总 3.2](experiment_results_summary_2026-10-12.md#32-人工复核成对开发集正常输入)相同的主比较，但补上“做了什么”。每行均使用各自训练目录的 `best.pt`，在**相同的 153 对、RGB 坐标标签、960 输入**上运行独立验证器；均为 seed 42。`IR-only` 不在此表，因为它用 IR 坐标真值。

| 方法及独立结果 | 做了什么 | Precision | Recall | mAP50 | mAP50:95 | 对 Concat |
|---|---|---:|---:|---:|---:|---:|
| [RGB-only](../runs/diagnostics/paired_reviewed_rgb_s42_normal_val_local960/evaluation.json) | 只用 RGB，单骨干检测 | 0.70783 | 0.52916 | 0.58438 | 0.27349 | −15.05 点 |
| [Concat+1×1](../runs/diagnostics/paired_reviewed_concat_s42_normal_val_local960/evaluation.json) | 双骨干 P3/P4/P5 特征拼接＋1×1 融合 | **0.87568** | 0.69546 | **0.79861** | **0.42397** | 参考 |
| [Common-Diff / A1](../runs/diagnostics/paired_reviewed_common_diff_s42_normal_val_local960/evaluation.json) | 受限对齐＋共模/绝对差模融合 | 0.79302 | **0.70626** | 0.77973 | 0.41665 | −0.73 点 |
| [Reliability / A2](../runs/diagnostics/paired_reviewed_reliability_s42_normal_val_local960/evaluation.json) | A1＋冻结教师相对效用监督门控 | 0.79888 | 0.67603 | 0.74781 | 0.38584 | −3.81 点 |
| [Full / A3](../runs/diagnostics/paired_reviewed_full_s42_normal_val_local960/evaluation.json) | A2＋训练期模态缺失/低照/模糊 | 0.85678 | 0.65896 | 0.76267 | 0.38322 | −4.07 点 |

该表的 YOLOv8s、960、SGD、AMP、seed 42 和统一开发验证协议只支持**单种子内部排序**。Concat 相对 RGB-only 高 15.05 点；但尚无已核验 IR→RGB 标定、多种子区间及独立航次，不能将差值全写为“融合机制的物理互补增益”。A1 虽多 1.08 点 Recall，却低 0.73 点 mAP50:95；A2/A3 更未超过强基线。[成对训练目录](../runs/paired_reviewed_dev_v1/)的各 `results.csv`、`args.yaml`、`results.png` 与混淆矩阵可追溯每次训练。

### 5.1 成对训练日志的最佳 epoch（供检查点核对）

以下是训练时验证日志，**不是**上表的独立验证器输出；两者末位差异来自评估实现，不宜混取。

| 方法及训练 CSV | 做了什么 | 最佳/完成 epoch | 训练时验证 mAP50:95 |
|---|---|---:|---:|
| [RGB-only](../runs/paired_reviewed_dev_v1/paired_reviewed_rgb_s42/results.csv) | 单 RGB 分支 | 35/75 | 0.27373 |
| [IR-only](../runs/paired_reviewed_dev_v1/paired_reviewed_ir_s42/results.csv) | 单 IR 分支，**IR 坐标验证** | 163/203 | 0.45724 |
| [Concat](../runs/paired_reviewed_dev_v1/paired_reviewed_concat_s42/results.csv) | 双流拼接强基线 | 207/220 | 0.42265 |
| [Common-Diff](../runs/paired_reviewed_dev_v1/paired_reviewed_common_diff_s42/results.csv) | 受限对齐＋共差 | 216/220 | 0.41781 |
| [Reliability](../runs/paired_reviewed_dev_v1/paired_reviewed_reliability_s42/results.csv) | 共差＋教师效用门控 | 118/158 | 0.38560 |
| [Full](../runs/paired_reviewed_dev_v1/paired_reviewed_full_s42/results.csv) | Reliability＋缺失/退化训练 | 204/220 | 0.38406 |

## 6. YTMS 成对模型的 IR 缺失与显式回退

在**同一 153 对 RGB 坐标验证集**上，将 IR 标为不可用而继续运行融合网络，得到下表；它衡量模型自带缺失路径，而非另外调度 RGB-only 检测器。

| 方法及独立结果 | 做了什么 | 正常 mAP50:95 | IR 缺失 mAP50:95 | 缺失时相对 RGB-only 0.27349 |
|---|---|---:|---:|---:|
| [Concat](../runs/diagnostics/paired_reviewed_concat_s42_ir_missing_val_local960/evaluation.json) | 双流模型内标记 IR 不可用，保留 RGB 分支 | 0.42397 | **0.28474** | +1.13 点 |
| [Reliability / A2](../runs/diagnostics/paired_reviewed_reliability_s42_ir_missing_val_local960/evaluation.json) | 门控/可用性掩膜处理缺失 IR | 0.38584 | 0.20658 | −6.69 点 |
| [Full / A3](../runs/diagnostics/paired_reviewed_full_s42_ir_missing_val_local960/evaluation.json) | 经随机失效训练的网络内缺失路径 | 0.38322 | 0.24975 | −2.37 点 |

按预设“缺失时与可用单模态差距 ≤1 点”，Full **未通过**。后来实现的 [`SensorFallbackRouter`](../dual_multimodal/fallback.py) 则是工程上**显式调用 RGB-only 检测器**：在 153/153 张 IR 缺失图像上均切换到 RGB-only，不能写成 Full 网络自发学会可靠性回退。RTX 5060、FP16、batch 1、960 输入的前向中位延迟：正常 Concat [14.044 ms](../artifacts/fallback_acceptance/concat_normal_val153_fp16_v1/summary.json)，正常 Full [17.824 ms](../artifacts/fallback_acceptance/normal_val153_fp16_v1/summary.json)，IR 缺失路由 RGB-only [8.704 ms](../artifacts/fallback_acceptance/ir_missing_val153_fp16_v1/summary.json)。详情和调用方式见[回退验收指南](sensor_fallback_guide.md)；这些计时不含磁盘、绘图、视频编码和实际艇载采集链路。

## 7. 可以据此写出的结论与下一步

**当前支持：**在这批 YTMS 内部开发数据上，P3/P4/P5 的双流 Concat 是最强正常输入基线；旧标签会大幅改变 IR-only AP；A3 对与训练同型的合成模糊/低照有一定适应，但以正常精度下降为代价；工程路由能在缺失 IR 时切换到 RGB-only。

**当前不支持：**共差表示、效用门控、失效训练优于简单 Concat；真实标定条件化可靠性成立；独立航次泛化、小目标优势、真实传感器故障识别或 USV 身份识别。旧 P4 教师目标几乎全偏 RGB，配准先验尚未真正落地。不要把 644/165、609/153、155/154 IR 子集和不同标签坐标的 AP 混成同一排行榜。

**下一步指令：**先冻结现有 YTMS 609/153 清单和评估脚本，不再按漏检结果删验证图；若继续在 YTMS 上开发，请先人工核验 RGB↔IR 对应与标定残差，重做平衡的双模态教师效用目标，再按相同的 153 对、相同 RGB 坐标评估 A2/A3 与 Concat 的 seed 42。只有超过强基线并通过 IR 缺失回退门槛，才值得追加 43/44 两种子和融合尺度消融；若不能，保留负结果，不叠加 P2/Transformer。要形成正式论文结论，最终仍需独立采集单位的人工真值测试集。

本报告将训练日志、独立 `evaluation.json` 和审核清单分开列示，作为现有 YTMS 结果索引；不覆盖这些原始文件。
