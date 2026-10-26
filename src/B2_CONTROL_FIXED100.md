# B2 control：从同一 B1 权重固定训练 100 epoch

将 `server_submit_b2_control_fixed100.sh` 上传到服务器 `/work/home/kekexiong/fog_ship_code_20261019/`。已有的 `run_b2.py`、`server_b2_v3.sh`、`server_prepare_mtdsp_v3.sh`、v3 数据和 B1 权重保持原位。

在服务器登录节点，使用此前训练 B1/B2 的环境执行：

```bash
source /work/home/kekexiong/conda/miniconda3/etc/profile.d/conda.sh
conda activate pt206_cu124_py311
cd /work/home/kekexiong/fog_ship_code_20261019
bash server_submit_b2_control_fixed100.sh
```

脚本先核对 v3 数据、Python 依赖和 B1 CPU 前向，再提交一个 GPU Slurm 作业。作业从 `runs/b1_s42/weights/best.pt` **重新初始化** control；不会读取 `runs_v3/b2_control_s42/weights/last.pt`，也不会使用 debug 权重。设置为 seed 42、全量 v3 数据、960 输入、batch 2、SGD、100 epoch、patience 100；其余训练参数来自现有 `run_b2.py`，与先前 control/small 一致。唯一特意改变的是早停耐心值，从 25 提至 100，以保证 100 轮。

新结果写入 `runs_v3/b2_control_fixed100_s42/`，原有 control 和 small 不会被覆盖。Slurm 日志名形如 `slurm-fog_b2_control_fixed100_s42-<jobid>.out`。作业末尾会检查 `results.csv` 是否有 100 轮及 `weights/best.pt` 是否存在。

训练完成后，比较新 control 与现有 small 的最佳权重。优先复核全部验证集 mAP50–95、雾天 AP50–95 和召回率、≤48 px 小船召回；仍需多种子配对实验才能评估稳定性。由于 small 原作业在第 98 轮早停，但其最佳权重来自第 73 轮，该轮仍处于固定 100 轮范围内。
