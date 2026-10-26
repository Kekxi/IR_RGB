# B2 v3：服务器只提交两条训练命令

本地已完成 v3 标签与天气表核验、`control` 和 `small` 的 CPU smoke，以及两组 1 epoch GPU debug。修正了短 debug 因梯度累积而没有更新参数的问题；新检查脚本已在两组本地结果上验证 loss 有限、检查点存在、融合层参数发生更新，`small` 的小目标分支也发生更新。

## 上传

将本目录 `E:\HEU_4151\RGB_IR\fog_ship_code_20261019\` 中以下文件，按相同相对路径上传到 `/work/home/kekexiong/fog_ship_code_20261019/`，覆盖服务器上的旧版同名脚本：

```text
server_prepare_mtdsp_v3.sh
server_b2.sh
server_b2_v3.sh
server_submit_b2_v3.sh
run_b2.py
check_b2_debug.py
val_conditions_reviewed.csv
label_corrections_v3/label_corrections.json
label_corrections_v3/mtdsp_13_20250314154739.txt
label_corrections_v3/mtdsp_13_20250314154744.txt
label_corrections_v3/mtdsp_13_20250314154749.txt
```

服务器代码目录还需保留已有的 `prepare_data.py`、`audit_visibility_conditions.py`、`b2_variants.py`、`rgb_ir_midfusion.py`、`fog_reliability.py` 和 `runs/b1_s42/weights/best.pt`。服务器 v2 数据集必须已在 `/work/home/kekexiong/datasets/MTDSP_v2_final_20261019/dataset`。不需上传本地整套图像。

## 服务器执行

激活之前运行 B1 的 Python 环境（命令提示符若已显示 `(pt206_cu124_py311)` 就无需重复），进入代码目录，然后**分两行**执行：

```bash
source /work/home/kekexiong/conda/miniconda3/etc/profile.d/conda.sh
conda activate pt206_cu124_py311
cd /work/home/kekexiong/fog_ship_code_20261019
bash server_submit_b2_v3.sh train control 42
bash server_submit_b2_v3.sh train small 42
```

每条提交命令自动在登录节点校验或创建独立 v3 数据集、校验天气表、检查 Python 依赖和 B1 权重、运行对应分支的 CPU smoke，然后提交一个 Slurm GPU 作业。每个 GPU 作业先用相同的 `IMGSZ` 与 `BATCH` 运行 1 epoch debug；检查有限 loss、模型文件和真实参数更新后，再从同一 B1 `best.pt` 开始最多 100 epoch 的完整训练。默认 `IMGSZ=960`、`BATCH=2`、`WORKERS=2`。两组使用相同设置；若服务器显存不足，两组统一改用 `BATCH=1` 再提交。不要在登录节点直接调用训练脚本。

v3 数据集位于 `/work/home/kekexiong/datasets/MTDSP_v3_labelreview_20261020/dataset`；v3 训练输出在 `runs_v3/b2_control_s42/`、`runs_v3/b2_small_s42/`，自动 debug 输出在 `runs_debug_v3/`。Slurm 日志为 `slurm-fog_train_b2v3_*-*.out`。已有 v3 目录但拷贝未完成时，准备脚本逐文件从冻结的 v2 补齐缺失文件；对已存在图像逐项比对 SHA-256，若与冻结 v2 不同，则从 v2 恢复该图像并记入 `v3_image_repair_log.jsonl`。标签仍按原始标签与三个修订标签逐项校验。随后检查全部 2145 对图像与标签，再生成或刷新 manifest。启动时应打印 `MTDSP v3 preparation: resume-safe v5`，据此确认服务器已更新脚本。

服务器的 Linux/Bash、Slurm 和 `960 × batch 2` 显存条件无法在本地 Windows 上原样验证；这些由已提交作业内部的 debug 检查。作业失败时查看对应 Slurm 日志，不能把“已提交”当作“训练完成”。

训练完毕且两个 `runs_v3/.../weights/best.pt` 均产生后，再提交相同配置的验证：

```bash
bash server_submit_b2_v3.sh val control 42
bash server_submit_b2_v3.sh val small 42
bash server_submit_b2_v3.sh groups control 42
bash server_submit_b2_v3.sh groups small 42
bash server_submit_b2_v3.sh size control 42
bash server_submit_b2_v3.sh size small 42
```

重点比较雾天 AP50–95、召回率与小船召回率，并检查清晰天气是否退化。seed 42 是初筛；有效后再做配对多种子重复。
