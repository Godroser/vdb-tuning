# Milvus 任务画像与相似度（OtterTune 第 4 节思路）

对**不同调优任务**在**相同标准负载**下采集 Prometheus 指标，用聚合后的指标向量距离近似任务相似度，便于后续做迁移 / 先验（与论文中 workload characterization + 相似任务检索对应）。

## 依赖

```bash
pip install requests numpy typer scikit-learn
```

`scikit-learn` 仅在使用 `run_task_characterization.py compare --pca N` 时需要；仅用欧氏/余弦距离时可不装。

## 1. 配置 PromQL

Milvus 指标名与 `job` 标签随版本与抓取配置变化。可复制并修改：

- `milvus_ottertune/metrics_prometheus.example.json` → 例如 `metrics_prometheus.json`

在 Prometheus UI 中先验证每条查询有非空结果。

## 2. 采集单个任务的画像

**方式 A：与负载并行采集（推荐）**  
对**每个任务**使用**同一条** `--workload-cmd`（标准负载），仅改 Milvus 数据/集合或任务语义；负载结束即停止轮询（避免采到空闲状态）。

```bash
cd /path/to/ottertune-configure

python run_task_characterization.py characterize \
  --prometheus http://127.0.0.1:9090 \
  --task-id glove-100-angular \
  --workload-cmd "timeout 900 ./run_engine_test.sh milvus-single-node milvus-p10 glove-100-angular" \
  --workload-cwd /path/to/vector-db-benchmark-master \
  --profiles-dir ./task_profiles \
  --samples 200 \
  --interval 10 \
  --metrics-json ./milvus_ottertune/metrics_prometheus.json
```

**方式 B：仅定时拉取**（负载由你手动或其它脚本启动）

```bash
python run_task_characterization.py characterize \
  --prometheus http://127.0.0.1:9090 \
  --task-id manual-run-1 \
  --samples 30 \
  --interval 10 \
  --profiles-dir ./task_profiles
```

输出：`task_profiles/<task_id>.json`，内含各指标在时间窗上的 **mean / std**。

## 3. 任务间相似度（距离矩阵）

对同一目录下多个 `*.json` 画像：

```bash
python run_task_characterization.py compare \
  --profiles-dir ./task_profiles \
  --metric euclidean \
  --out-json ./similarity_matrix.json
```

- 默认会对**跨任务的特征**做列方向 z-score，再算距离（避免量纲主导）。
- `--metric cosine`：余弦距离 `1 - cos_sim`。
- `--pca 5`：先用 PCA 降维再算距离（需 sklearn）。
- `--fa 5`：用 **FactorAnalysis**（与论文中因子分析更贴近）；若指定则优先于 `--pca`。
- `--no-standardize` / `--no-std-features`：关闭标准化或去掉 std 特征向量。

## 4. 与 `main_ottertune.py` 的关系

当前 `main_ottertune.py` 仍是**离线 GPRGD**；本模块提供**任务侧画像与相似度**，可后续接：按距离选历史任务、对 GPR 先验或初始 LHS 做加权。需要时可再把这些 JSON 读入调优主循环。
