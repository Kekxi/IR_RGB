# 显式传感器失效回退：本地验证与后续步骤

本文件只针对旧数据代理分组的内部诊断，不构成独立航次测试。

## 已完成的对照

相同的 153 对验证图像、463 个 RGB 坐标框、960 输入、seed 42：

| 路径 | mAP50:95 |
|---|---:|
| Full，双模态正常 | 0.383224 |
| Full，IR 缺失但仍用 Full 网络 | 0.249750 |
| 独立 RGB-only 检测器 | 0.273490 |
| A2，IR 缺失 | 0.206577 |
| Concat，双模态正常 / IR 缺失 | 0.423965 / 0.284740 |

因此 Full 的内部门控没有达到“IR 缺失时与 RGB-only 相差 ≤1 个百分点”；A3
的失效训练比 A2 有帮助，但仍不构成可靠回退。上述每个条件的详细结果见
`runs/diagnostics/paired_reviewed_*_val_local960/evaluation.json`。

## 可用的路由接口

`dual_multimodal/fallback.py` 提供 `SensorFallbackRouter`，不修改任何权重：

- RGB、IR 都可用：运行双模态模型，输出坐标按 RGB。
- IR 缺失：运行独立 RGB-only 权重，输出坐标按 RGB。
- RGB 缺失：只有显式提供 IR-only 权重时才运行它，输出坐标按 IR。
- 两路都缺失：报错；不把伪零图当成可用模态。

最小用法（`rgb`/`ir` 是与训练预处理一致的 BCHW、三通道、`0–1` 张量）：

```python
from dual_multimodal.fallback import SensorFallbackRouter

router = SensorFallbackRouter.from_checkpoints(
    fusion=r"E:\HEU_4151\dual-multimodal\runs\paired_reviewed_dev_v1\paired_reviewed_full_s42\weights\best.pt",
    rgb=r"E:\HEU_4151\dual-multimodal\runs\paired_reviewed_dev_v1\paired_reviewed_rgb_s42\weights\best.pt",
    ir=r"E:\HEU_4151\dual-multimodal\runs\paired_reviewed_dev_v1\paired_reviewed_ir_s42\weights\best.pt",
    device="cuda:0",
)
result = router.predict_modal(rgb=rgb, ir=None)
assert result.branch == "rgb" and result.coordinate_system == "rgb"
raw_yolo_output = result.output
```

模型级路由本身返回原始 YOLO 输出。`tools/predict_fallback.py` 已补齐图像/视频
读取、同步 letterbox、NMS、画框和 JSONL 结果保存。它不会把权重合并为一个。
双模态路径默认使用标定置信度 0（未验证的几何关系）；有可靠标定时，应
显式传入与图像同尺寸的 `calibration_confidence` 张量。缺失 IR 时不需要该张量。

## 可直接运行的推理脚本

在 `E:\HEU_4151\dual-multimodal` 目录，使用 `heu_wurenji` 环境的 Python：

```bat
python tools\predict_fallback.py --rgb "E:\HEU_4151\HEU_yolo11\datasets\RGB_IR_dataset\rgb\T_7833.jpg" --fusion-weights "runs\paired_reviewed_dev_v1\paired_reviewed_full_s42\weights\best.pt" --rgb-weights "runs\paired_reviewed_dev_v1\paired_reviewed_rgb_s42\weights\best.pt" --output "artifacts\fallback_acceptance\my_rgb_image" --device cuda:0 --imgsz 960 --half
```

这条命令只给 RGB，因此自动走 RGB-only。双模态图片还需传 `--ir`；只给 IR
时需传 `--ir-weights`。处理视频时，`--rgb` 或 `--ir` 可改为视频文件。只有在
**已确认两段视频逐帧同步**时，才能同时输入两段视频并加
`--assume-frame-synced`；相同帧率和帧数本身不足以证明同步。

输出目录必须尚不存在，脚本不会覆盖旧结果。目录内包含 `summary.json`、
`predictions.jsonl`、带框图片或 `predictions.mp4`。`branch` 和
`coordinate_system` 会写入每条预测记录。原始图像、标注和权重不变。

对冻结验证清单复现整个 IR 缺失路由（不保存逐张图片）：

```bat
python tools\predict_fallback.py --manifest "artifacts\paired_reviewed_dev_v1\manifest_fusion_rgb_coords.csv" --freeze "artifacts\paired_reviewed_dev_v1\freeze_fusion_rgb_coords.json" --split val --condition ir_missing --fusion-weights "runs\paired_reviewed_dev_v1\paired_reviewed_full_s42\weights\best.pt" --rgb-weights "runs\paired_reviewed_dev_v1\paired_reviewed_rgb_s42\weights\best.pt" --output "artifacts\fallback_acceptance\my_ir_missing_val" --device cuda:0 --imgsz 960 --half --no-images
```

## 2026-10-07 本地接入验收

RTX 5060，batch 1，960 方形 letterbox，FP16，前三次预热后统计 153 对验证图像：

| 输入 | 实际路由 | 模型前向中位 / P95 | 预处理＋前向＋NMS 中位 / P95 |
|---|---|---:|---:|
| Concat，RGB+IR 正常 | Concat 153/153 | 14.044 / 16.319 ms | 41.889 / 45.996 ms |
| RGB+IR 正常 | Full 153/153 | 17.824 / 20.124 ms | 45.541 / 50.202 ms |
| IR 缺失 | RGB-only 153/153 | 8.704 / 10.251 ms | 14.989 / 16.971 ms |

Full 相对 Concat 的模型前向中位延迟增加约 26.9%，超过 20%；但在本脚本的
预处理＋前向＋NMS 口径下增加约 8.7%。两种口径结论不同，应同时报告，
不能用后一项掩盖模型本身的开销。此处前向、前处理口径都不是艇载全链路。

验收原始文件：
`artifacts/fallback_acceptance/normal_val153_fp16_v1/summary.json`、
`artifacts/fallback_acceptance/ir_missing_val153_fp16_v1/summary.json`、
`artifacts/fallback_acceptance/concat_normal_val153_fp16_v1/summary.json`。
单图带框示例：
`artifacts/fallback_acceptance/ir_missing_one_v1/T_7833_rgb.jpg` 和
`artifacts/fallback_acceptance/normal_one_render_v1/T_7833_fusion.jpg`。
还使用三个重复帧生成的单路和双路测试视频验证了写出：
`artifacts/fallback_acceptance/video_rgb_three_v1/predictions.mp4` 与
`artifacts/fallback_acceptance/video_pair_three_v1/predictions.mp4` 均可读出 3 帧；
双路测试的 3/3 帧走 Full。重复静态图像不构成真实时间同步测试。

直接 RGB 图片与“清单中 IR 缺失”的同一帧输出了完全相同的 10 个检测框，
证明路由未把 IR 伪图送入 RGB-only 分支。原有验证器的 RGB-only
`mAP50:95=0.273490` 可作为该分支的检测质量参照；新脚本使用方形 letterbox，
不能把这里的 JSONL 当成与矩形验证器完全相同的 AP 测试。

上述耗时排除了磁盘读取、画框、视频编码和传感器采集；不是端到端实时艇载
延迟。峰值 PyTorch allocated 约 151 MiB（正常）/134 MiB（缺 IR），不是
`nvidia-smi` 总显存占用。旧数据没有可靠的跨模态标定或独立航次测试；本次
通过的是工程路由和本地推理稳定性，不是论文主方法的性能准入门。

## 下一步验收

1. 若使用真实双路视频，先提供采集时间戳和安装标定；只有帧序号不够。
   未经校准的这批旧数据不能用于证明可靠性门控的物理意义。
2. 在实际采集链路测量包括解码、标定变换和数据传输的端到端延迟及显存；
   并与 Concat 采用同样
   外部回退策略作公平比较。当前 Full 在正常和缺 IR 条件均低于 Concat，
   不宜启动更多种子的正式主方法训练。
3. 若继续研究模型内回退，优先检查教师效用目标几乎全偏 IR、RGB/IR
   坐标不一致及共享检测头在单模态时的分布偏移；修正后再跑 seed 42。

本地回归测试：

```bat
cd /d E:\HEU_4151\dual-multimodal
D:\anaconda3\envs\heu_wurenji\python.exe -m unittest discover -s tests -p "test_*.py"
```

当前本地效用缓存的服务器原始清单哈希与已重定位的本地清单哈希不相同。
不得直接用该 JSON 在本地重新训练；但它的 609 个训练 `pair_id` 与本地
清单完全一致，且教师权重哈希匹配，可用于此次诊断。
