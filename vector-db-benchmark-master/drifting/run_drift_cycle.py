#!/usr/bin/env python3
"""
数据漂移模拟脚本：向 Milvus 插入新向量并删除旧向量，模拟数据更新场景。
不修改原始数据集文件，所有新向量在内存中随机生成。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 项目根目录 = vector-db-benchmark-master（drifting 的父目录）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pymilvus import Collection, connections, utility

from engine.clients.milvus.config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DEFAULT_ALIAS,
    MILVUS_DEFAULT_PORT,
)


def load_dataset_config(dataset_name: str) -> dict:
    """从 datasets.json 加载数据集配置"""
    datasets_path = ROOT / "datasets" / "datasets.json"
    with open(datasets_path, "r") as f:
        configs = json.load(f)
    for cfg in configs:
        if cfg["name"] == dataset_name:
            return cfg
    raise ValueError(f"Dataset '{dataset_name}' not found in datasets.json")


def get_initial_count(dataset_name: str) -> int:
    """从数据集文件推断初始向量数量（不修改原文件）"""
    cfg = load_dataset_config(dataset_name)
    base = ROOT / "datasets"
    path_str = cfg["path"]
    if cfg["type"] in ("tar",) or any(
        x in dataset_name for x in ("random-geo", "random-match", "random-range", "h-and-m", "arxiv", "yandex", "dbpedia", "laion")
    ):
        # AnnCompoundReader 格式：path 指向目录，内含 vectors.npy
        vectors_file = base / path_str / "vectors.npy"
        if vectors_file.exists():
            return int(np.load(vectors_file).shape[0])
    elif cfg["type"] == "h5":
        import h5py
        h5_path = base / path_str
        with h5py.File(h5_path, "r") as f:
            return int(f["train"].shape[0])
    elif cfg["type"] == "jsonl":
        vectors_file = base / path_str / "vectors.jsonl"
        if vectors_file.exists():
            with open(vectors_file) as f:
                return sum(1 for _ in f)
    raise ValueError(f"Cannot infer count for dataset {dataset_name}")


def generate_random_vectors(n: int, dim: int, normalize: bool = True) -> np.ndarray:
    """生成随机向量，可选 L2 归一化（用于 cosine 距离）"""
    vectors = np.random.randn(n, dim).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    return vectors


def run_drift_cycle(
    host: str = "127.0.0.1",
    port: int = 19530,
    dataset_name: str = "random-geo-radius-2048-angular-no-filters",
    batch_size: int = 1000,
    base_id: int = 0,
    max_id: int = 99999,
) -> tuple[int, int]:
    """
    执行一轮数据漂移：插入 batch_size 个新向量，删除 batch_size 个最旧向量。
    返回新的 (base_id, max_id)。
    """
    cfg = load_dataset_config(dataset_name)
    dim = cfg["vector_size"]
    normalize = cfg.get("distance", "").lower() == "cosine"

    # 连接 Milvus
    connections.connect(
        alias=MILVUS_DEFAULT_ALIAS,
        host=host,
        port=str(port),
    )
    collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)

    # 生成并插入新向量
    new_ids = list(range(max_id + 1, max_id + 1 + batch_size))
    vectors = generate_random_vectors(batch_size, dim, normalize)
    collection.insert([new_ids, vectors.tolist()])

    # 删除最旧的向量
    delete_ids = list(range(base_id, base_id + batch_size))
    # Milvus 删除表达式
    ids_str = ",".join(str(i) for i in delete_ids)
    expr = f"id in [{ids_str}]"
    collection.delete(expr)

    # 刷新使变更生效
    collection.flush()

    connections.disconnect(MILVUS_DEFAULT_ALIAS)

    return base_id + batch_size, max_id + batch_size


def get_segment_vector_counts(
    host: str = "127.0.0.1",
    port: int = 19530,
) -> list[dict]:
    """
    获取每个 segment（物理分区）内的向量数量。
    SCANN 等 partition-based 索引在 Milvus 中以 segment 为物理存储单元，
    每个 segment 有独立的索引结构。
    返回: [{"segment_id": ..., "num_rows": ..., "state": ...}, ...]
    """
    connections.connect(
        alias=MILVUS_DEFAULT_ALIAS,
        host=host,
        port=str(port),
    )
    try:
        seg_infos = utility.get_query_segment_info(
            collection_name=MILVUS_COLLECTION_NAME,
            using=MILVUS_DEFAULT_ALIAS,
        )
        result = []
        for seg in seg_infos:
            seg_id = getattr(seg, "segmentID", None) or getattr(seg, "segment_id", "?")
            num_rows = getattr(seg, "num_rows", None) or getattr(seg, "numRows", 0)
            state = getattr(seg, "state", "?")
            part_id = getattr(seg, "partitionID", None) or getattr(seg, "partition_id", "?")
            result.append({
                "segment_id": str(seg_id),
                "num_rows": int(num_rows) if num_rows is not None else 0,
                "state": str(state),
                "partition_id": str(part_id),
            })
        return result
    finally:
        connections.disconnect(MILVUS_DEFAULT_ALIAS)


def main():
    parser = argparse.ArgumentParser(description="执行数据漂移：插入新向量、删除旧向量")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus 主机")
    parser.add_argument("--port", type=int, default=19530, help="Milvus 端口")
    parser.add_argument(
        "--dataset",
        default="random-geo-radius-2048-angular-no-filters",
        help="数据集名称（用于获取 vector_size 和 distance）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="每轮插入/删除的向量数量",
    )
    parser.add_argument(
        "--base-id",
        type=int,
        default=0,
        help="当前最旧未删除的 ID（由主脚本维护）",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=99999,
        help="当前已插入的最大 ID（由主脚本维护）",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="状态文件路径，用于保存/读取 base_id 和 max_id",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="仅获取并输出各 segment 的向量数量，不执行漂移",
    )
    parser.add_argument(
        "--output-stats",
        default=None,
        help="将 segment 统计写入此文件（JSON 格式）",
    )
    parser.add_argument(
        "--get-initial-count",
        metavar="DATASET",
        default=None,
        help="仅输出指定数据集的初始向量数量并退出",
    )
    args = parser.parse_args()

    if args.get_initial_count:
        print(get_initial_count(args.get_initial_count))
        return

    if args.stats_only:
        stats = get_segment_vector_counts(host=args.host, port=args.port)
        total = sum(s["num_rows"] for s in stats)
        print(f"Segment 数量: {len(stats)}, 总向量数: {total}")
        for i, s in enumerate(stats):
            print(f"  Segment {i+1}: segment_id={s['segment_id']}, num_rows={s['num_rows']}, state={s['state']}")
        if args.output_stats:
            out = {"segments": stats, "total_vectors": total, "segment_count": len(stats)}
            with open(args.output_stats, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        return

    base_id = args.base_id
    max_id = args.max_id

    # 从状态文件读取（若存在且指定）
    if args.state_file and os.path.exists(args.state_file):
        with open(args.state_file, "r") as f:
            state = json.load(f)
            base_id = state["base_id"]
            max_id = state["max_id"]

    new_base, new_max = run_drift_cycle(
        host=args.host,
        port=args.port,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        base_id=base_id,
        max_id=max_id,
    )

    # 写回状态文件
    if args.state_file:
        with open(args.state_file, "w") as f:
            json.dump({"base_id": new_base, "max_id": new_max}, f, indent=2)

    print(f"Drift cycle done: inserted {args.batch_size}, deleted {args.batch_size}")
    print(f"New state: base_id={new_base}, max_id={new_max}")


if __name__ == "__main__":
    main()
