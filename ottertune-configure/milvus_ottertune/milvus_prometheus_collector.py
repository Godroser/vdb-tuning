"""
Prometheus-based Milvus metrics collection for OtterTune-style workload characterization.

Metric names / labels depend on your Milvus and Prometheus scrape config; override via
MilvusMetricsCollector(metrics_map=...) or a JSON file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

LOG = logging.getLogger(__name__)

# Default PromQL snippets aligned with OtterTune-style numeric workload features (Sec. 4.1).
# Adjust job labels / metric names to match your deployment (e.g. job="milvus", pod=...).
DEFAULT_MILVUS_METRICS: Dict[str, str] = {
    "query_nq": "sum(milvus_proxy_search_vectors_count_total)",
    "search_latency_avg": "avg(milvus_proxy_search_latency_ms)",
    "cache_hit_rate": "avg(milvus_querynode_cache_hit_ratio)",
    "memory_usage_bytes": "sum(process_resident_memory_bytes{job='milvus'})",
    "cpu_usage_percent": "avg(rate(process_cpu_seconds_total{job='milvus'}[1m])) * 100",
    "segment_count": "sum(milvus_datacoord_segment_count)",
    "index_build_latency": "avg(milvus_rootcoord_index_build_latency_ms)",
    "s3_read_bytes": "sum(rate(milvus_storage_download_size_bytes_total[1m]))",
    "msg_size_in": "sum(rate(milvus_proxy_req_size_bytes_total[1m]))",
}


class MilvusMetricsCollector:
    """Fetch instant-vector values from Prometheus HTTP API."""

    def __init__(
        self,
        prometheus_url: str,
        metrics_map: Optional[Dict[str, str]] = None,
        timeout_sec: float = 30.0,
        verify_tls: bool = True,
    ):
        """
        :param prometheus_url: Base URL, e.g. http://localhost:9090
        :param metrics_map: name -> PromQL instant query
        """
        self.prometheus_url = prometheus_url.rstrip("/")
        self.query_api = f"{self.prometheus_url}/api/v1/query"
        self.metrics_map = metrics_map if metrics_map is not None else dict(DEFAULT_MILVUS_METRICS)
        self.timeout_sec = timeout_sec
        self.verify_tls = verify_tls

    @classmethod
    def from_json(cls, prometheus_url: str, path: str | Path, **kwargs) -> "MilvusMetricsCollector":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics") or data
        if not isinstance(metrics, dict):
            raise ValueError("JSON must be an object mapping metric_name -> promql")
        # Drop comment / non-string values
        metrics = {k: v for k, v in metrics.items() if isinstance(v, str) and not k.startswith("_")}
        return cls(prometheus_url, metrics_map=metrics, **kwargs)

    def fetch_metric(self, prom_ql: str) -> Optional[float]:
        """Execute instant query; return scalar value or None on empty/error."""
        try:
            r = requests.get(
                self.query_api,
                params={"query": prom_ql},
                timeout=self.timeout_sec,
                verify=self.verify_tls,
            )
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") != "success":
                LOG.warning("Prometheus non-success: %s", payload)
                return None
            results = (payload.get("data") or {}).get("result") or []
            if not results:
                return None
            val = results[0].get("value")
            if not val or len(val) < 2:
                return None
            return float(val[1])
        except Exception as e:
            LOG.warning("Error fetching %s: %s", prom_ql, e)
            return None

    def collect_all_metrics(self) -> Dict[str, float]:
        """One snapshot: all configured metrics that returned a value."""
        current: Dict[str, float] = {}
        for name, ql in self.metrics_map.items():
            v = self.fetch_metric(ql)
            if v is not None:
                current[name] = v
        return current

    def observe_period(
        self,
        num_samples: int,
        interval_sec: float,
        on_sample: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Observation period [cite OtterTune: repeated samples over workload window].

        :param num_samples: number of polling rounds
        :param interval_sec: sleep between rounds
        :param on_sample: optional callback(dict_snapshot) after each sample
        :return: list of rows, each row includes metric keys + timestamp_epoch
        """
        history: List[Dict[str, Any]] = []
        for i in range(num_samples):
            snap = self.collect_all_metrics()
            snap["timestamp_epoch"] = time.time()
            snap["sample_index"] = i
            history.append(snap)
            if on_sample:
                on_sample(snap)
            if i < num_samples - 1 and interval_sec > 0:
                time.sleep(interval_sec)
        return history
