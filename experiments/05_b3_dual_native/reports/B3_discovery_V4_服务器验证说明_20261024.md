# B3 第一阶段：雾天“多发现独立船舶”验证

这一步使用已有的五组检测输出：RGB-only、IR-only、B1、B3 RGB 头、B3 IR 头。**不重新训练模型，也不需要上传新的预训练权重。**目的是回答 B3 双头是否比两个独立模型的结果汇总更有效，以及 IR 头是否找到了 RGB 难以看到的独立船舶。

## 交付文件

- `b3_discovery_export.py`：从 V4 的 `pairs.csv` 和天气审核表挑出 633 对雾天验证图，生成带 `R0/I0` 编号的 RGB/IR 对照图与逐船对应 CSV；随后用正常的 Ultralytics 验证流程导出五组模型的逐图预测、置信度、原生坐标框。
- `b3_discovery_score.py`：先审核对应表，计算各模型原生坐标的固定阈值诊断；对应表完整后，统计不重复船舶的检出、误报、重复报告，并按相同误报预算比较四种系统。
- `server_submit_b3_discovery_v4.sh`：服务器 `prepare/submit/status/score` 命令。
- `val_conditions_reviewed.csv`：V4 验证图像的人工天气分类。其余三个 Python 依赖文件 `b3_dual_head.py`、`fog_reliability.py`、`rgb_ir_midfusion.py` 随压缩包提供。

把压缩包内容解压到 `/work/home/kekexiong/fog_ship_code_20261019/`。默认假设 V4 数据集与四个现有 `best.pt` 仍在原服务器位置，并使用 `ultralytics==8.3.163`。如果根目录不同，脚本顶部的环境变量 `CODE_ROOT`、`DATA_ROOT`、`RGB_WEIGHTS`、`IR_WEIGHTS`、`B1_WEIGHTS`、`B3_WEIGHTS` 可以覆盖默认路径。

## 服务器执行

```bash
cd /work/home/kekexiong/fog_ship_code_20261019
bash server_submit_b3_discovery_v4.sh prepare
bash server_submit_b3_discovery_v4.sh submit
bash server_submit_b3_discovery_v4.sh status
```

`prepare` 在 `eval_b3_discovery_v4/prepared/` 生成 633 对雾天清单、2132 个 RGB 框、2342 个 IR 框及 `review_images/` 图。对照图左为 RGB、右为 IR；框号从零开始，例如 `R0` 和 `I0`。`submit` 向 Slurm 提交一张 GPU 的任务，顺序导出五组预测。每组完成后，`exports/<variant>/predictions.jsonl` 和 `export_summary.json` 均应存在。任务最后自动运行一次 `score`：对应表尚未填写时只产生原生检测诊断和待核对数量，**不会**生成独立船舶结论。

完成预测导出后，按 `prepared/review_images/<id>.jpg` 核对 `prepared/correspondence_review.csv`。每一行代表一个 IR 真值框 `I<ir_gt_index>`：

| 情况 | `rgb_gt_index` | `status` |
|---|---|---|
| 与 RGB 的 `Rj` 是同一艘船，RGB 看不清或较弱 | `j` | `shared_rgb_weak` |
| 与 RGB 的 `Rj` 是同一艘船，RGB 也清晰 | `j` | `shared_rgb_clear` |
| 能确认同船，但 RGB 可见程度无法确定 | `j` | `shared_rgb_uncertain` |
| IR 中有船，RGB 共同视场内没有对应船框 | 留空 | `ir_only_common_fov` |
| IR 中有船，但落在 RGB 视场外 | 留空 | `ir_only_outside_rgb_fov` |

`suggested_*` 列只是同像素距离提示。RGB 与 IR 配准并不精确，**不能直接把建议编号当成同船对应**。无法判断身份时，保持 `status` 空白或填 `uncertain`；评分脚本会拒绝输出独立船舶结果。若发现错误或重复的真值框，应先修订数据并重做对应表，不能用 `not_ship_or_duplicate` 悄悄排除。每个 IR 框只能对应至多一个 RGB 框，每个 RGB 框也只能被对应一次；未被任何 IR 框对应的 RGB 框自动算 RGB-only。

填写并保存 CSV 后运行：

```bash
bash server_submit_b3_discovery_v4.sh score
```

结果在 `eval_b3_discovery_v4/scores/`：

- `correspondence_audit.json`：对应表是否完整及具体待核对位置。
- `native_fixed_conf025.json`：不依赖跨模态对应的原生坐标诊断。
- `unique_ship_report.md`、`unique_ship_report.json`：仅在全部对应关系核实后出现。包含 RGB-only、B1、IR-only、两个独立模型汇总、B3 双头五种系统的独立船舶检出与误报；同时列出 `shared_rgb_weak` 与 `ir_only_common_fov` 等目标类别。

同误报预算的阈值在这批验证图上搜索，因此报告是**探索性性能上界**。跨模态去重使用的是人工真值对应，只衡量两头预测是否包含互补信息；它不是可部署的自动去重模块。正式声称系统在新海况下“多发现船舶”，还需要锁定阈值和去重规则后，在独立事件上验证。V4 连续帧相关，不能把相邻帧当作独立船例。

## 本地检查范围

已在本地用 V4 中四对真实雾天图运行 `prepare`、五组权重的 `export`、未填写对应表时的拒算流程，以及**仅用于程序检查的模拟对应表**评分流程；该模拟对应表不代表人工真值结果。完整 633 对预处理计数为 RGB 2132 框、IR 2342 框。服务器 Slurm 任务和 2342 行人工对应审核尚需在服务器及人工复核环节完成。
