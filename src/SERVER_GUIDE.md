# 第一项创新实验：雾天 RGB/IR 模态可靠性门控

## 1. 这次实验回答什么

使用同一个 YOLO11m 双骨干、P3/P4/P5 融合、检测头和 RGB 坐标真值：

- **B0**：原来的 `Concat + 1×1 Conv` 融合。
- **B1**：在三个融合尺度前预测逐位置 RGB 权重。输入是两路特征的响应强度、局部对比度、跨模态差异；RGB 权重低时 IR 权重相应升高。初始权重均为 0.5，B1 起点等价于 B0。

第一轮只检验门控是否有收益，不把局部对齐、雾化增强、小波同时加入。门控是从特征中学习可靠性，不保证自动识别真实雾；需要实验结果和可视化验证。

## 2. 上传文件

从 Windows 上传下列三项到同一台 Linux GPU 服务器：

| 本机位置 | 服务器示例位置 | 用途 |
|---|---|---|
| `E:\HEU_4151\RGB_IR\MTDSP_v2_final_20261019\dataset` 整个目录 | `~/fog_ship/dataset` | 最终只读数据，含 `pairs.csv`、RGB/IR 图像和两路标签 |
| `E:\HEU_4151\RGB_IR\fog_ship_code_20261019.zip` | `~/fog_ship/code.zip`，解压成 `~/fog_ship/code` | 本实验脚本和说明；压缩包不含本机生成的 Windows 路径清单 |
| `E:\HEU_4151\HEU_yolo11\ultralytics-8.3.163\yolo11m.pt` | `~/fog_ship/yolo11m.pt` | B0/B1 相同的官方单路初始化 |

不需要上传 `MTDSP_v1`、`MTDSP_v2_review`、旧 `best.pt`、原始雷达文件或整个 Windows 项目。旧 YOLOv8m/Fusion 权重不能用于这份 YOLO11m B0/B1 的公平初始条件。服务器至少需要 Python、PyTorch、CUDA 和 `ultralytics==8.3.163`；保留服务器原有可用的 PyTorch/CUDA 组合。

Windows PowerShell 示例（先把 `USER@HOST` 换成真实用户名和服务器地址）：

```powershell
ssh USER@HOST 'mkdir -p ~/fog_ship'
Set-Location 'E:\HEU_4151\RGB_IR\MTDSP_v2_final_20261019'
scp -r .\dataset USER@HOST:~/fog_ship/
Set-Location 'E:\HEU_4151\RGB_IR'
scp .\fog_ship_code_20261019.zip USER@HOST:~/fog_ship/code.zip
Set-Location 'E:\HEU_4151\HEU_yolo11\ultralytics-8.3.163'
scp .\yolo11m.pt USER@HOST:~/fog_ship/
```

如果服务器登录使用端口或跳板机，在 `ssh/scp` 中按学校的连接方式增加参数。传输时不要把 `dataset` 放进 `code` 目录。

## 3. 服务器环境与数据预检

以下命令均在服务器终端执行：

```bash
cd ~/fog_ship
mkdir -p code
unzip -o code.zip -d code
chmod -R a-w dataset
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install 'ultralytics==8.3.163'
python -c "import ultralytics; print(ultralytics.__version__)"
python code/prepare_data.py --dataset "$HOME/fog_ship/dataset" --output "$HOME/fog_ship/manifests"
cat manifests/manifest_summary.json
```

预期 `train=1266`、`val=877`，并排除两对原图尺寸不同的样本。脚本只写 `manifests/`，不修改图像和标签。若 PyTorch/CUDA 原先不可用，先用服务器管理员提供的环境激活方式；不要盲目重装 GPU 版 PyTorch。
Ultralytics 在扫描 YOLO 标签时可能尝试创建 `.cache`；上面的 `chmod` 使最终数据目录保持只读，缓存无法写入也不影响训练。

## 4. 烟雾测试

```bash
cd ~/fog_ship/code
python run_experiment.py smoke --variant b0 --weights ../yolo11m.pt
python run_experiment.py smoke --variant b1 --weights ../yolo11m.pt
```

两条都出现 `smoke forward OK` 后，先做极小训练以检查数据加载、损失反传和保存：

```bash
python run_experiment.py train --variant b1 --weights ../yolo11m.pt --data ../manifests/fusion_eligible.yaml --project ../runs_debug --name b1_debug --epochs 1 --patience 1 --imgsz 640 --batch 1 --workers 2 --fraction 0.02 --device 0
```

这一步的结果只验证脚本能跑通。`fraction` 仅缩小训练集，验证仍使用完整 877 对；调试目录不能用于正式比较。显存不足先减小 `batch`，必要时再减小 `imgsz`，正式 B0/B1 必须用一致设置。

## 5. 正式 B0/B1 消融

先以 42 为种子训练两组，确保能稳定收敛，再用 43、44 重复。下面是一组命令示例：

```bash
cd ~/fog_ship/code
python run_experiment.py train --variant b0 --weights ../yolo11m.pt --data ../manifests/fusion_eligible.yaml --project ../runs --name b0_s42 --epochs 220 --patience 40 --imgsz 960 --batch 2 --workers 2 --seed 42 --device 0
python run_experiment.py train --variant b1 --weights ../yolo11m.pt --data ../manifests/fusion_eligible.yaml --project ../runs --name b1_s42 --epochs 220 --patience 40 --imgsz 960 --batch 2 --workers 2 --seed 42 --device 0
```

如果正式运行必须使用 640 或 batch 1，对 B0/B1 同时修改。不要把新 YOLO11m 的分数与之前 YOLOv8m 的历史分数直接当成门控消融。两组需保持相同初始化、数据、训练超参数、种子和指标代码。

训练结束后分别用 `best.pt` 重评估：

```bash
python run_experiment.py val --variant b0 --weights ../runs/b0_s42/weights/best.pt --data ../manifests/fusion_eligible.yaml --project ../eval --name b0_s42 --imgsz 960 --batch 2 --device 0
python run_experiment.py val --variant b1 --weights ../runs/b1_s42/weights/best.pt --data ../manifests/fusion_eligible.yaml --project ../eval --name b1_s42 --imgsz 960 --batch 2 --device 0
```

需要保存每组 `args.yaml`、`results.csv`、`best.pt`、验证结果与日志。后续补全 877 对验证图像的人工能见度分组后，按相同图像 ID 另算雾/低能见度、清晰天气的 AP50–95 与召回率；目前 49 对已复核验证样本不能直接作为正式雾天结论。`IR 有船但 RGB 同视场无框`的船不属于现有 RGB 真值中的 TP，需要另立共同视场真值协议。

已有一份 [val_conditions.csv]，包含 877 对验证图像，49 对复核结果已预填（41 `haze_fog`、8 `clear`），其余 828 对留空。请根据原图逐行填 `visibility` 为 `haze_fog` 或 `clear`；无法判断时填 `low_visibility_unknown` 并说明。它只是图像级天气条件表，不是 LabelImg 目标框。填完后重新上传该 CSV 到 `~/fog_ship/code/val_conditions.csv`。`eval_groups.py` 默认要求所有 877 对都有明确的雾/清晰归类；仅探索时可加 `--allow-partial`，且输出会注明排除数量。

```bash
cd ~/fog_ship/code
python eval_groups.py --dataset ../dataset --conditions val_conditions.csv --weights ../runs/b0_s42/weights/best.pt --output ../groups_b0_s42 --imgsz 960 --batch 2 --device 0
python eval_groups.py --dataset ../dataset --conditions val_conditions.csv --weights ../runs/b1_s42/weights/best.pt --output ../groups_b1_s42 --imgsz 960 --batch 2 --device 0
```

分组结果见两个输出目录各自的 `group_metrics.csv`。同样执行种子 43 和 44，再汇总均值与逐种子差异。

## 6. 成功判据与局限

先看三个种子中 B1 是否稳定提高雾/低能见度 AP50–95 与召回率，再看清晰天气和全量 AP、推理耗时、显存；不要只挑单次最高值。若 B1 提升不稳定，检查假阳性和弱可见船的逐例结果，再决定是否继续做对齐可信度模块。当前验证集和训练集按事件分割，但投稿级泛化结论仍需要独立事件或外部测试。

这份代码按本机 `ultralytics 8.3.163` 源码接口编写。本机已完成语法、2145 对数据清单检查、B0/B1 前向测试、B1 门控反传测试，以及**另外复制的 2 训练对 + 2 验证对**的 YOLO11n 单轮训练；B1 的 `best.pt` 已可重新加载验证。初始 B0/B1 预测逐元素差异为 0。这个小数据测试仅验证程序流程，不能说明雾天检测效果；YOLO11m 全数据训练仍需在服务器运行。
