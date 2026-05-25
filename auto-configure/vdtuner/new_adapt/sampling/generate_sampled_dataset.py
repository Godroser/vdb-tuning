#!/usr/bin/env python3
"""Generate a sampled dataset while keeping query vectors unchanged."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_ADAPT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = NEW_ADAPT_DIR.parent.parent.parent
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "vector-db-benchmark-master" / "datasets"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASETS_ROOT / "new_adapt"

sys.path.insert(0, str(NEW_ADAPT_DIR))
from generate_drift_dataset import resolve_source


def iter_jsonl(path: Path) -> Iterator[object]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def sanitize_dataset_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    name = name.strip("-._")
    return name or "custom-dataset"


def infer_top_k(source_dir: Path, fallback: int) -> int:
    neighbours_path = source_dir / "neighbours.jsonl"
    if not neighbours_path.exists():
        return fallback
    with neighbours_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            values = json.loads(line)
            if isinstance(values, list) and values:
                return len(values)
    return fallback


def count_train_vectors_jsonl(vectors_path: Path) -> int:
    count = 0
    with vectors_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                count += 1
    return count


def load_queries_jsonl(queries_path: Path) -> np.ndarray:
    queries = [np.asarray(item, dtype=np.float32) for item in iter_jsonl(queries_path)]
    if not queries:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(queries)


def write_queries_and_load_matrix(source, out_dir: Path) -> np.ndarray:
    if source.kind == "jsonl":
        assert source.queries_jsonl is not None
        queries_out = out_dir / "queries.jsonl"
        if not source.queries_jsonl.exists():
            return np.empty((0, source.vector_size), dtype=np.float32)
        matrix = load_queries_jsonl(source.queries_jsonl)
        with source.queries_jsonl.open("r", encoding="utf-8") as src, queries_out.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                dst.write(line if line.endswith("\n") else line + "\n")
        return matrix

    assert source.h5_path is not None
    queries_out = out_dir / "queries.jsonl"
    with h5py.File(source.h5_path, "r") as data, queries_out.open("w", encoding="utf-8") as dst:
        test = data["test"]
        matrix = np.asarray(test, dtype=np.float32)
        for i in range(test.shape[0]):
            dst.write(json.dumps(test[i].tolist()) + "\n")
    return matrix


def write_neighbors(
    sampled_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbours_path: Path,
    distance: str,
    top_k: int,
) -> int:
    if sampled_vectors.shape[0] == 0 or query_vectors.shape[0] == 0:
        return 0
    k = min(top_k, sampled_vectors.shape[0])
    metric = "cosine" if distance == "cosine" else "euclidean"
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(sampled_vectors)
    indices = nn.kneighbors(query_vectors, return_distance=False)
    with neighbours_path.open("w", encoding="utf-8") as fp:
        for row in indices:
            fp.write(json.dumps([int(x) for x in row.tolist()]) + "\n")
    return int(k)


def run(args: argparse.Namespace) -> None:
    datasets_root = Path(args.datasets_root).resolve()
    source = resolve_source(
        datasets_root=datasets_root,
        source_path=args.source_path,
        distance=args.distance,
        vector_size=args.vector_size,
    )

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    output_root = Path(args.output_root).resolve()
    run_dir = output_root / args.output_name
    sampled_dir = run_dir / "sampled"
    sampled_dir.mkdir(parents=True, exist_ok=True)

    source_name = source.dataset_config.get("name") or sanitize_dataset_name(source.label)
    sampled_name = f"{args.dataset_name_prefix}-sampled"

    if source.kind == "jsonl":
        assert source.vectors_jsonl is not None
        total_vectors = count_train_vectors_jsonl(source.vectors_jsonl)
    else:
        assert source.h5_path is not None
        with h5py.File(source.h5_path, "r") as data:
            total_vectors = int(data["train"].shape[0])
    if total_vectors <= 0:
        raise RuntimeError("No train vectors found in source dataset.")

    sample_count = max(1, int(round(total_vectors * args.sample_ratio)))
    sample_count = min(sample_count, total_vectors)
    rng = np.random.default_rng(args.seed)
    selected_ids = sorted(rng.choice(total_vectors, size=sample_count, replace=False).tolist())
    selected_id_set = set(selected_ids)

    vectors_out = sampled_dir / "vectors.jsonl"
    payloads_out = sampled_dir / "payloads.jsonl"

    sampled_vectors: list[np.ndarray] = []
    payload_iter = source.iter_payloads()
    wrote_payloads = False

    with vectors_out.open("w", encoding="utf-8") as vf:
        pf = payloads_out.open("w", encoding="utf-8") if payload_iter is not None else None
        try:
            for idx, vector in source.iter_train_vectors():
                payload = next(payload_iter) if payload_iter is not None else None
                if idx not in selected_id_set:
                    continue
                vf.write(json.dumps(vector) + "\n")
                sampled_vectors.append(np.asarray(vector, dtype=np.float32))
                if pf is not None:
                    pf.write(json.dumps(payload) + "\n")
                    wrote_payloads = True
        finally:
            if pf is not None:
                pf.close()

    if not wrote_payloads and payloads_out.exists():
        payloads_out.unlink()

    sampled_matrix = np.vstack(sampled_vectors)
    query_matrix = write_queries_and_load_matrix(source, sampled_dir)

    top_k = args.neighbors_top_k
    if top_k <= 0:
        top_k = infer_top_k(source.source_dir, fallback=100)
    actual_top_k = write_neighbors(
        sampled_vectors=sampled_matrix,
        query_vectors=query_matrix,
        neighbours_path=sampled_dir / "neighbours.jsonl",
        distance=source.distance,
        top_k=top_k,
    )

    info = {
        "source_kind": source.kind,
        "source_label": source.label,
        "source_dataset_name": source_name,
        "source_dir": str(source.source_dir),
        "sampled_dir": str(sampled_dir),
        "sampled_dataset_name": sampled_name,
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "vector_size": source.vector_size,
        "distance": source.distance,
        "stats": {
            "total_vectors": total_vectors,
            "sampled_vectors": sample_count,
            "sample_ratio_actual": sample_count / total_vectors,
            "queries": int(query_matrix.shape[0]),
            "neighbors_top_k": int(actual_top_k),
        },
    }
    info_path = run_dir / "sample_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "sample_info": str(info_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a dataset's base vectors while keeping query vectors unchanged."
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Root directory containing benchmark datasets.",
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Dataset name (from datasets.json), folder path, or .hdf5 file path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for sampled datasets.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Run folder name under output-root.",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        required=True,
        help="Prefix used to name sampled dataset for benchmark outputs.",
    )
    parser.add_argument("--sample-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distance",
        default="",
        help="Distance metric; empty means use datasets.json or cosine.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=0,
        help="Vector dimension; 0 means auto-detect.",
    )
    parser.add_argument(
        "--neighbors-top-k",
        type=int,
        default=0,
        help="Ground-truth neighbors per query; <=0 means infer from source or use 100.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
