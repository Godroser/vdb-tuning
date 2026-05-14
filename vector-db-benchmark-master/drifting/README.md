# Drift scripts relocated

数据漂移脚本已迁移至：

`auto-configure/vdtuner/adapt/`

请在该目录使用：

- `run_drift_cycle.py` — 单轮漂移，可选导出当前 corpus + 精确 KNN 快照 (`--export-snapshot` / `--export-only`)
- `run_drift_test.sh` — 端到端漂移性能测试（结果仍在 `vector-db-benchmark-master/results/`；快照写入 `adapt/drift_exports/`）
- `knob_adapt.py` — 同上目录，已指向新的 `run_drift_cycle.py` 与 `.drift_state.json`
