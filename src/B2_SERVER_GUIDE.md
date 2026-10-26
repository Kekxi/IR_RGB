# B2 两条条件路线：服务器运行手册

## 实验目的

两种 B2 是互斥的研究假设，均从**同一颗已训练 B1 best.pt**继续训练。`control` 从同一 B1 继续训练相同轮数和学习率，排除额外训练带来的假提升。

| 分支 | 修改位置 | 关键假设 | 首要验证 |
| --- | --- | --- | --- |
| `align` | P4 融合 | 局部 RGB/IR 错位限制 B1 雾天收益 | 人工平移、真实错位组的 AP 与召回 |
| `small` | P3 融合及训练损失 | 远处小船特征/监督不足限制 B1 | 小船召回、按尺寸分组 AP |
| `control` | 无新增模块 | 只继续训练 B1 | 与两条 B2 同条件比较 |

`align` 在 P4 的 5×5 特征邻域计算跨模态局部匹配，得到 RGB 坐标下的 IR 位移及基于匹配分布熵的可信度。可信度调节 IR 的空间迁移，并作为 B1 门控的第六个描述量。匹配可信度不是经过校准的概率；低可信度并不等于 IR 没有船。

`small` 在 P3 增加初始为零的细节残差，并对输入尺寸不超过 48 像素的 RGB 标注船框增加中心点辅助损失。48 像素为实验超参数；若尺寸分布分析显示不合适，应在正式比较前固定新阈值。辅助中心损失参与优化，但 Ultralytics 默认 `box/cls/dfl` 三列日志不会单列显示它。

## 上传到服务器

本地文件在 `E:\HEU_4151\RGB_IR\fog_ship_code_20261019\`。把以下**新增/更新**文件上传到服务器 `/work/home/kekexiong/fog_ship_code_20261019/` 同名位置：

```text
b2_variants.py
run_b2.py
server_b2.sh
server_submit_b2.sh
eval_groups.py                 # 更新了 B2 checkpoint 类注册
eval_shift.py                  # B1/B2 人工错位鲁棒性诊断
eval_size_recall.py            # B1/B2 按船框尺寸统计召回率
B2_SERVER_GUIDE.md
```

服务器原有的 `rgb_ir_midfusion.py`、`fog_reliability.py`、`run_experiment.py`、`manifests/fusion_eligible.yaml`、`val_conditions.csv`、`runs/b1_s42/weights/best.pt` 继续使用。`yolo11m.pt` 仅是 B0/B1 初始权重，**B2 必须读取 B1 best.pt**。数据集仍为 `/work/home/kekexiong/datasets/MTDSP_v2_final_20261019/dataset`，脚本只读该目录；不改 LabelImg 框。

## 服务器命令

登录后先激活先前 B1 使用的 Python 环境，然后：

```bash
cd /work/home/kekexiong/fog_ship_code_20261019
bash server_b2.sh check
bash server_b2.sh smoke control 42
bash server_b2.sh smoke align 42
bash server_b2.sh smoke small 42
```

`check` 和 `smoke` 可以在无 GPU 的登录节点运行。`smoke` 需要 `runs/b1_s42/weights/best.pt` 已生成，检查 B2 CPU 前向；不代表训练或损失函数已经通过 GPU 试跑。

对要采用的路线和 `control`，先各跑一次 GPU debug，例如选择 `align`：

```bash
bash server_submit_b2.sh debug control 42
bash server_submit_b2.sh debug align 42
squeue -u "$USER"
```

等 debug 作业完成并确认日志无错误，再提交正式训练：

```bash
bash server_submit_b2.sh train control 42
bash server_submit_b2.sh train align 42
```

如果诊断指向远处小船，将上面 `align` 改成 `small`。要同时研究两个假设，可分别提交 `align`、`small`；算力不足时先选一条。GPU job 使用现有 `xhhgnormal02` Slurm 分区；如学校调整分区，修改 `server_submit_b2.sh` 中的 `--partition`。在显存不足时，三组都用相同设置，如 `BATCH=1 bash server_submit_b2.sh train align 42`；正式试验不要混用图像尺寸。

训练结束、`runs/b2_*/weights/best.pt` 存在后：

```bash
bash server_submit_b2.sh val control 42
bash server_submit_b2.sh val align 42
```

如选小船路线，将 `align` 改为 `small`。整体 AP 见 `eval/b2_*/`。

在验证集上比较相同权重的 IR 水平平移 0/4/8/16 输入像素（RGB 和 RGB 标签固定）：

```bash
bash server_submit_b2.sh shift b1 42
bash server_submit_b2.sh shift control 42
bash server_submit_b2.sh shift align 42
```

如选小船路线，将最后一行改为 `shift small 42`。结果在 `shift_*/shift_metrics.csv`。此测试覆盖全部验证集；要得出**雾天专属**错位结论，还需完成天气分组后进行对应子集测试。

按输入尺寸统计船框召回率，验证小目标路线是否真正帮助小船：

```bash
bash server_submit_b2.sh size b1 42
bash server_submit_b2.sh size control 42
bash server_submit_b2.sh size small 42
```

结果在 `size_*/size_recall.csv`，指标为 `conf≥0.25、IoU≥0.50` 的逐框召回率，尺寸按模型输入中的框长边分为 ≤48、48–128、>128 像素。该指标不是 COCO AP，也不能代替标准 AP50–95。

天气条件表完成全部 877 对验证集分组后，再运行：

```bash
bash server_submit_b2.sh groups control 42
bash server_submit_b2.sh groups align 42
```

分组结果见 `groups_b2_*/group_metrics.csv`。若 `val_conditions.csv` 尚有未复核行，脚本会拒绝正式分组统计。不能把现有 49 对已填天气的子集称为完整雾天验证集。

确认种子 42 的趋势后，对 43、44 重复 B1、control、所选 B2 的配套实验；每颗 B2 使用**同种子对应**的 B1 `best.pt`。报告均值与波动，并保留逐图结果用于配对分析。

## 判断 B2 该选哪条

先固定 B0/B1 权重与 RGB 标注，对 IR 输入做 0、4、8、16 输入像素的人工平移，仅用于鲁棒性诊断，不修改训练数据或标签。如果 B1 雾天收益随平移明显下降，并在真实错位样本上有相似现象，优先 `align`。如果性能相对稳定而漏检集中在远处小船，优先 `small`。人工平移是合成扰动，不能替代真实错位组评估。

当前 RGB 坐标标注中没有框的 IR-only 船，不能作为现有 RGB mAP 的真阳性。若论文要报告“IR 补回 RGB 不可见船”，先另建经复核的联合目标测试标注。

## 与后续文献阅读衔接

读到新的方法时，记录：**参考坐标系、对齐是否需要配对框、对齐可信度定义、雾天/小目标测试切分、是否有 IR-only 目标标注、同等训练预算的消融**。将这些条件与本实验对应起来，避免仅凭模块名称判断创新性。
