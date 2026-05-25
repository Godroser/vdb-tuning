# Dataset Sampling Benchmark

对指定数据集按比例采样 base vectors，保持 query 向量不变，分别测试原始数据集与采样后数据集的吞吐、延迟和 recall，并输出对比结果。

## 目录结构

```
sampling/
├── README.md
├── generate_sampled_dataset.py   # 生成采样数据集
└── run_sampling_benchmark.sh       # 一键采样 + benchmark + 对比
└── sampling_param_sweep.py #自动化设置不同的sample比例，索引参数，自动跑sampling并收集前后的性能。
└── train_perfscale_model.py # 基于多组sampling数据，训练一个性能scale的预测模型。
```

依赖同目录上一级的公共脚本：

- `../generate_drift_dataset.py`：数据集解析与读取
- `../run_custom_benchmark.py`：运行 vector-db-benchmark

## 工作流程

1. 读取指定数据集（支持 `datasets.json` 中的名称、目录路径或 `.hdf5` 文件）
2. 按 `SAMPLE_RATIO` 随机采样 base vectors（保留 `payloads.jsonl`，若存在）
3. 原样复制 `queries.jsonl`
4. 基于采样后的 base vectors 重建 `neighbours.jsonl`（用于 recall 评估）
5. 分别在原始数据集和采样数据集上运行 benchmark
6. 输出性能对比摘要

## 快速开始

在仓库根目录执行：

```bash
bash auto-configure/vdtuner/new_adapt/sampling/run_sampling_benchmark.sh \
  milvus-single-node milvus-p10 random-100
```

参数说明：

| 位置参数 | 默认值 | 说明 |
|---------|--------|------|
| `SERVER_PATH` | `milvus-single-node` | Milvus docker compose 目录名 |
| `ENGINE_NAME` | `milvus-p10` | benchmark 引擎配置名 |
| `SOURCE_DATASET_PATH` | `random-100` | 源数据集（名称/路径/h5） |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLE_RATIO` | `0.5` | 采样比例，范围 `(0, 1]` |
| `NEIGHBORS_TOP_K` | `0` | 重建 ground truth 的 K；`0` 表示从源数据集推断，否则默认 100 |
| `SEED` | `42` | 随机采样种子 |
| `DISTANCE` | 空 | 距离度量；空则使用 `datasets.json` 或 `cosine` |
| `VECTOR_SIZE` | `0` | 向量维度；`0` 表示自动检测 |
| `HOST` | `127.0.0.1` | Milvus 地址 |
| `RUN_TAG` | 自动时间戳 | 本次运行标识 |
| `VENV_PATH` | `/talas-pool/home/z78ding/venv` | Python 虚拟环境路径 |

示例：采样 30% 数据

```bash
SAMPLE_RATIO=0.3 SEED=123 \
bash auto-configure/vdtuner/new_adapt/sampling/run_sampling_benchmark.sh \
  milvus-single-node milvus-p10 random-100
```

## 单独生成采样数据集

不跑 benchmark，仅生成采样数据：

```bash
python3 auto-configure/vdtuner/new_adapt/sampling/generate_sampled_dataset.py \
  --source-path random-100 \
  --output-name my-sample-run \
  --dataset-name-prefix new-adapt-my-sample-run \
  --sample-ratio 0.5
```

输出目录：

```
vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/
├── sample_info.json
└── sampled/
    ├── vectors.jsonl
    ├── queries.jsonl
    └── neighbours.jsonl
```

## 输出指标

终端和 `perf_compare_sampling_summary.json` 中包含：

| 指标 | 含义 |
|------|------|
| `rps` | 吞吐（queries/s） |
| `p95_time` / `p99_time` | P95 / P99 延迟 |
| `mean_time` | 平均单次查询延迟 |
| `mean_precisions` | 平均 recall（@K） |
| `total_time` | 总查询耗时 |

## 输出文件

每次运行在以下目录生成结果：

```
vector-db-benchmark-master/datasets/new_adapt/<RUN_TAG>/
├── sample_info.json                      # 采样元信息
├── original_result_meta.json             # 原始数据集 benchmark 元信息
├── sampled_result_meta.json              # 采样数据集 benchmark 元信息
└── perf_compare_sampling_summary.json    # 性能对比摘要
```

## 注意事项

- 每次 benchmark 前会重启 Milvus（`docker compose down -v && up -d`），确保索引状态一致。
- 采样后 query 不变，但 `neighbours.jsonl` 会基于采样后的 base 重新计算，recall 反映的是采样后 ground truth 下的检索质量。
- 需要已安装 Python 依赖（`numpy`, `h5py`, `scikit-learn`）以及可运行的 Milvus docker 环境。
