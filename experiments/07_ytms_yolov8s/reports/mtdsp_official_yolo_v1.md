# MTDSP 官方检测框转换与单模态扩充实验

## 数据边界

本次转换读取原始 28 个观测目录中的 `Photo_ScreenShot_*_DLDete.txt`，不覆盖原始 MTDSP、旧 `MTDSP_YOLO_pseudo` 或既有训练结果。相同光学照片在 2002/2003 雷达目录中的重复记录经过内容一致性核查后只保留一份。完整 RGB/IR 时间配对为 2,145 对。

官方检测记录中的 6 种船类统一映射为 YOLO 类别 `0: vessel`；浮标类排除。保留所有船类检测框，不再以 0.70 置信度筛图或删框。`_DLDete.txt` 是自动检测预标注，暂按本轮训练的正确框使用，但尚未证明逐框人工复核；不能将其当作正式评测真值。

输出目录为 `artifacts/mtdsp_official_yolo_v1`。其中 `inventory.jsonl` 记录每对图像的原始文件、船型、船舶标识、时间、两模态预标注数量及导出位置；`build_audit.json` 保存统计和冻结哈希。图像是独立副本，原数据未移动或修改。整包约 1.44 GiB。

每个模态下三个 MTDSP 图像目录的意义相同：`train_mtdsp` 是非预留标识且该模态有船框的预标注训练图像；`pending_mtdsp` 是非预留标识但缺检测文件或无船框的待核图像，暂不作为负样本；`holdout_mtdsp` 是三个预留 MMSI 的所有图像，不参与任何训练，等待人工补齐真值后评测。RGB 与 IR 是否进入 `train_mtdsp` 分别判断，因此两目录数量不必一致；联合训练在此阶段指**单模态数据源联合**，并非已完成 RGB/IR 双流配对融合。

| 内容 | RGB | IR |
|---|---:|---:|
| 船类预标注框 | 6,131 | 6,885 |
| 其中置信度 <0.70 的船框 | 2,701 | 2,690 |
| 排除的浮标框 | 990 | 1,285 |
| MTDSP 扩充训练图像 | 1,469 | 1,489 |
| 缺少船类预标注、待核查图像 | 295 | 275 |
| 按 MMSI 隔离的保留图像 | 381 | 381 |
| 人工复核 YTMS 训练／验证图像 | 609／153 | 609／153 |

预留 MMSI 为 `412331720`、`414150000`、`414211000`。这些标识的全部 381 对不参加 MTDSP 扩充训练。另有 1,764 对属于训练候选池；单模态缺失检测文件或无船类框的图像进入 `pending_mtdsp`，不被错误地当成确认无船的负样本。

## 人工评测队列

`manual_review_queue.jsonl` 是固定的 300 对待审核清单，不是已经完成的真值：

- 180 对来自完全预留的三个 MMSI，作为身份隔离的测试候选；其中包含旧 0.70 筛选流程丢弃的全部 46 对。
- 120 对来自训练候选池，仅作标签质量抽检；其中 80 对来自旧流程丢弃的配对。它们**不能**充当独立测试。
- 合计覆盖 12 个标识、7 种来源船型以及旧流程丢弃的 126 对。队列哈希见 `manual_review_queue_freeze.json`。

人工审核时必须补全两模态所有可见船框，判定无法可靠标注的图像，并确认 RGB/IR 框是否指向同一目标。完成前不得用 MTDSP 保留集计算正式 AP、召回率或配准精度。由于身份隔离测试仅覆盖三个船型，其余船型的质量抽检结果须单独报告，不能冒称同等级的身份隔离测试。

## 本地复核

在 `E:\HEU_4151\dual-multimodal` 的 Anaconda Prompt / CMD 中运行：

```bat
python tools\audit_mtdsp_official_yolo.py --pack "E:\HEU_4151\dual-multimodal\artifacts\mtdsp_official_yolo_v1"
python -m unittest tests.test_mtdsp_official_yolo -v
```

六份可移植 YAML 均以自身目录为根。`mtdsp_pretrain` 的**训练集**只有 MTDSP 有船框图像；其 `val` 指向人工复核的 YTMS 开发集，预训练脚本设置 `val=False` 跳过常规逐轮验证，但 Ultralytics 可能在最后一轮仍执行一次验证。预训练始终取 `last.pt` 而非依据验证集选权重。所有阶段的验证图像始终是同一批 153 对旧 YTMS 开发样本：

```
rgb_ytms_only.yaml
rgb_ytms_plus_mtdsp.yaml
rgb_mtdsp_pretrain.yaml
ir_ytms_only.yaml
ir_ytms_plus_mtdsp.yaml
ir_mtdsp_pretrain.yaml
```

这仍是旧数据代理分组的**内部开发验证**，不构成独立航次测试。旧 `MTDSP_YOLO_pseudo` 内的 `ytms_plus_mtdsp_*.yaml` 指向未复核的旧 YTMS 划分，不用于本轮对照。

## 服务器提交

将整个 `artifacts/mtdsp_official_yolo_v1` 连同当前项目代码同步到服务器对应位置；保持包内目录结构不变。登录服务器并进入 `dual-multimodal` 后，在现有 GPU Conda 环境运行：

```bash
export MTDSP_PACK_ROOT="$PWD/artifacts/mtdsp_official_yolo_v1"
export YOLOV8S_WEIGHTS="$PWD/yolov8s.pt"
test -f "$YOLOV8S_WEIGHTS" || { echo "请先将 COCO 初始化的 yolov8s.pt 放到此处，或把变量改成实际绝对路径"; exit 1; }
python tools/audit_mtdsp_official_yolo.py --pack "$MTDSP_PACK_ROOT"
```

如果项目根目录没有 `yolov8s.pt`，先使用已存在的权重文件绝对路径设置 `YOLOV8S_WEIGHTS`。请勿让网络自动下载替代已冻结的初始化权重。

推荐的两阶段顺序如下。先分别提交 RGB/IR 的 MTDSP-only 预训练；等对应作业**成功结束并存在 `weights/last.pt`**后，提交 YTMS+MTDSP 联合训练。两阶段均通过项目根目录 `run.sh` 在单卡 V100 16 GB 上运行：

```bash
bash server/submit_mtdsp_unimodal.sh rgb mtdsp_pretrain
bash server/submit_mtdsp_unimodal.sh ir mtdsp_pretrain
```

用 `squeue -u "$USER"` 和相应 `slurm-mtdsp_*.out` 确认作业已结束；两份完成标志均存在后再进入第二阶段：

```bash
test -f runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_rgb_mtdsp_pretrain_s42/training_complete.json
test -f runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_ir_mtdsp_pretrain_s42/training_complete.json
```

确认预训练成功并生成 `training_complete.json` 后：

```bash
bash server/submit_mtdsp_unimodal.sh rgb ytms_plus_mtdsp_finetune
bash server/submit_mtdsp_unimodal.sh ir ytms_plus_mtdsp_finetune
```

联合阶段从相应预训练 `last.pt` 初始化，采用 `ytms_plus_mtdsp.yaml`，即 YTMS 与 MTDSP 图像在同一个单模态训练集中参与更新。作为对照，另跑相同 YTMS 验证集上的 COCO 初始化 YTMS-only；旧的 COCO 初始化直接联合训练可作可选消融：

```bash
bash server/submit_mtdsp_unimodal.sh rgb ytms_only
bash server/submit_mtdsp_unimodal.sh ir ytms_only
bash server/submit_mtdsp_unimodal.sh rgb ytms_plus_mtdsp
bash server/submit_mtdsp_unimodal.sh ir ytms_plus_mtdsp
```

如果队列限制不足，等待前面的完成后再提交剩余任务，不要重复提交同名实验。默认 YOLOv8s、960 输入、SGD、seed 42、物理 batch 4、`nbs=16`、AMP；预训练固定 80 epochs、不使用验证选权重；联合阶段 220 epochs、patience 40。显存不足时统一设置 `YTMS_BATCH=2`，可比组保持相同配置。输出目录为 `runs/mtdsp_official_prelabel_v1/mtdsp_prelabel_<mode>_<arm>_s42`。联合阶段的 `plots=True` 应生成 `results.png`、混淆矩阵和验证批次的预测框图；运行中请先检查 `slurm-mtdsp_*.out` 是否有错误。

这些对照只回答“MTDSP 伪框预训练或联合训练后，人工复核 YTMS 内部验证是否改善”。由于预训练增加了额外优化步数，不能把与 YTMS-only 的差异全部归因于数据；如要严格归因，应增加等步数 YTMS-only 对照。不能据此宣称 MTDSP 正式测试提升或双模态融合创新成立。完成人工复核和跨模态对应／配准检查后，才能用预留 MTDSP 评测集比较双流 Concat 与新方法。
