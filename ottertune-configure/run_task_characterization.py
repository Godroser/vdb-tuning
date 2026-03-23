#!/usr/bin/env python3
"""
OtterTune-style Milvus task characterization: same canonical workload + Prometheus observation.

Workflow (per task):
  1. (Optional) Start a subprocess running the same benchmark / load script as other tasks.
  2. Poll Prometheus every --interval for --samples (or until workload exits + optional tail).
  3. Aggregate metrics (mean/std) → TaskProfile JSON.
  4. Compare tasks: pairwise distance matrix (euclidean or cosine).

Example:
  # Record profile for dataset A (run load + observe in parallel)
  python run_task_characterization.py characterize \\
    --prometheus http://127.0.0.1:9090 \\
    --task-id glove-100-angular \\
    --workload-cmd "timeout 600 ./run_engine_test.sh milvus-single-node milvus-p10 glove-100-angular" \\
    --workload-cwd /path/to/vector-db-benchmark-master \\
    --profiles-dir ./task_profiles \\
    --samples 30 --interval 10

  # Poll only (you run load manually for 5 minutes)
  python run_task_characterization.py characterize \\
    --prometheus http://127.0.0.1:9090 \\
    --task-id my-task \\
    --samples 30 --interval 10 \\
    --profiles-dir ./task_profiles

  # Compare all saved profiles
  python run_task_characterization.py compare --profiles-dir ./task_profiles
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import typer

# Package root: ottertune-configure/
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from milvus_ottertune.milvus_prometheus_collector import MilvusMetricsCollector
from milvus_ottertune.task_similarity import (
    TaskProfile,
    aggregate_observation,
    load_profiles_from_dir,
    pairwise_distance_matrix,
    save_profile,
)

app = typer.Typer(help="Milvus Prometheus task characterization & similarity")
LOG = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


@app.command()
def characterize(
    prometheus: str = typer.Option(..., help="Prometheus base URL, e.g. http://localhost:9090"),
    task_id: str = typer.Option(..., help="Unique task / workload label (e.g. dataset name)"),
    profiles_dir: Path = typer.Option(Path("./task_profiles"), help="Directory for profile JSON"),
    metrics_json: Optional[Path] = typer.Option(
        None,
        help="Optional JSON: {\"metric_name\": \"promql\", ...} to override defaults",
    ),
    samples: int = typer.Option(30, help="Number of Prometheus polls (observation samples)"),
    interval: float = typer.Option(10.0, help="Seconds between polls"),
    workload_cmd: Optional[str] = typer.Option(
        None,
        help="Shell command for canonical load; if set, observation runs in parallel until it exits",
    ),
    workload_cwd: Optional[Path] = typer.Option(
        None,
        help="Working directory for workload subprocess",
    ),
    use_sudo_workload: bool = typer.Option(
        False,
        help="Prepend sudo to workload (only if your script needs it)",
    ),
    tail_after_workload_sec: float = typer.Option(
        0.0,
        help="After workload exits, keep polling for this many seconds (0 = stop immediately)",
    ),
    no_tls_verify: bool = typer.Option(False, help="Disable TLS verify for Prometheus"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Run (optional) canonical workload and collect Prometheus snapshots → save TaskProfile.
    """
    _setup_logging(verbose)
    collector = _make_collector(prometheus, metrics_json, no_tls_verify)

    history: List[dict] = []
    stop_poll = threading.Event()
    workload_rc: List[Optional[int]] = [None]

    def poll_loop() -> None:
        idx = 0
        while not stop_poll.is_set() and idx < samples:
            snap = collector.collect_all_metrics()
            snap["timestamp_epoch"] = time.time()
            snap["sample_index"] = idx
            history.append(snap)
            LOG.info("Sample %s/%s: %s keys", idx + 1, samples, len(snap) - 2)
            idx += 1
            if stop_poll.wait(timeout=interval):
                break

    if workload_cmd:
        cmd = workload_cmd
        if use_sudo_workload:
            cmd = f"sudo {cmd}"
        cwd = str(workload_cwd) if workload_cwd else None
        LOG.info("Starting workload: %s (cwd=%s)", cmd, cwd)
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        poll_thread = threading.Thread(target=poll_loop, daemon=True)
        poll_thread.start()
        _, err = proc.communicate()
        workload_rc[0] = proc.returncode
        if proc.returncode != 0:
            LOG.warning("Workload exited with code %s; stderr (truncated): %s", proc.returncode, (err or b"")[:500])
        # Stop polling as soon as workload exits (avoid sampling idle cluster).
        stop_poll.set()
        poll_thread.join(timeout=30.0)
        if tail_after_workload_sec > 0:
            deadline = time.time() + tail_after_workload_sec
            while time.time() < deadline and len(history) < samples:
                time.sleep(interval)
                snap = collector.collect_all_metrics()
                snap["timestamp_epoch"] = time.time()
                snap["sample_index"] = len(history)
                history.append(snap)
    else:
        LOG.info("No workload_cmd: collecting %s samples every %ss", samples, interval)
        poll_loop()

    names, mean_v, std_v, counts = aggregate_observation(history)
    profile = TaskProfile(
        task_id=task_id,
        metric_names=names,
        mean_vector=mean_v,
        std_vector=std_v,
        num_samples=len(history),
        raw_sample_count_per_metric=counts,
        meta={
            "prometheus": prometheus,
            "samples_requested": samples,
            "interval_sec": interval,
            "workload_cmd": workload_cmd,
            "workload_rc": workload_rc[0],
        },
    )
    out = profiles_dir / f"{_safe_filename(task_id)}.json"
    save_profile(profile, out)
    LOG.info("Saved profile: %s (metrics=%s, rows=%s)", out, len(names), len(history))
    typer.echo(json.dumps({"profile": str(out), "metrics": len(names), "rows": len(history)}, indent=2))


@app.command()
def compare(
    profiles_dir: Path = typer.Option(Path("./task_profiles"), help="Directory with *.json profiles"),
    metric: str = typer.Option("euclidean", help="euclidean or cosine"),
    no_standardize: bool = typer.Option(False, help="Disable cross-task z-score on features"),
    no_std_features: bool = typer.Option(False, help="Do not append std vector to features"),
    pca: int = typer.Option(0, help="If >0 and sklearn installed, PCA n_components before distance"),
    fa: int = typer.Option(
        0,
        help="If >0, use FactorAnalysis (OtterTune-style) instead of PCA; overrides --pca",
    ),
    out_json: Optional[Path] = typer.Option(None, help="Write distance matrix JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Load all profiles in directory and print pairwise distance matrix."""
    _setup_logging(verbose)
    profiles = load_profiles_from_dir(profiles_dir)
    if len(profiles) < 2:
        typer.echo("Need at least 2 profile JSON files in the directory.", err=True)
        raise typer.Exit(1)
    D, ids = pairwise_distance_matrix(
        profiles,
        metric=metric,  # type: ignore[arg-type]
        zscore_across_tasks=not no_standardize,
        include_std_in_features=not no_std_features,
        pca_components=pca if pca > 0 and fa <= 0 else None,
        fa_components=fa if fa > 0 else None,
    )
    typer.echo("Task IDs: " + ", ".join(ids))
    typer.echo("Distance matrix (lower = more similar under same probe workload):")
    typer.echo(_format_matrix(D, ids))
    if out_json:
        payload = {"task_ids": ids, "metric": metric, "matrix": D.tolist()}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"Wrote {out_json}")


def _make_collector(
    prometheus: str,
    metrics_json: Optional[Path],
    no_tls_verify: bool,
) -> MilvusMetricsCollector:
    if metrics_json:
        return MilvusMetricsCollector.from_json(prometheus, metrics_json, verify_tls=not no_tls_verify)
    return MilvusMetricsCollector(prometheus, verify_tls=not no_tls_verify)


def _safe_filename(task_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:200]


def _format_matrix(D, ids: List[str]) -> str:
    w = max(len(s) for s in ids) if ids else 8
    head = " " * (w + 2) + " ".join(f"{j:>8}" for j in range(len(ids)))
    lines = [head]
    for i, rid in enumerate(ids):
        row = f"{rid:<{w}}  " + " ".join(f"{D[i, j]:8.4f}" for j in range(len(ids)))
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    app()
