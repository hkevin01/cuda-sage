"""SQLite-backed persistent cache for tuning results.

Keyed by (kernel_source_hash, arch, params_json) so identical kernel code
on the same architecture never needs re-benchmarking.

Cache file location: ~/.cache/cuda-sage/tune.db  (or $CUDA_SAGE_CACHE_DIR).
"""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .benchmark import BenchmarkPoint


def _default_db_path() -> Path:
    base = Path(os.environ.get("CUDA_SAGE_CACHE_DIR", Path.home() / ".cache" / "cuda-sage"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "tune.db"


class TuneCache:
    """Persistent SQLite cache for KernelAutoTuner results.

    Schema:
        results (source_hash TEXT, arch TEXT, params_json TEXT,
                 time_ms REAL, occupancy REAL, source_kind TEXT,
                 error TEXT)

    Thread safety: Each connection is created per-operation (not shared).
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _default_db_path()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    source_hash TEXT NOT NULL,
                    arch        TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    time_ms     REAL NOT NULL,
                    occupancy   REAL NOT NULL,
                    source_kind TEXT NOT NULL,
                    error       TEXT,
                    PRIMARY KEY (source_hash, arch, params_json)
                )
            """)

    @staticmethod
    def _hash(source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def get(
        self,
        source: str,
        arch: str,
        params: dict[str, Any],
    ) -> Optional[BenchmarkPoint]:
        """Return cached BenchmarkPoint or None if not found."""
        key = json.dumps(params, sort_keys=True)
        h = self._hash(source)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM results WHERE source_hash=? AND arch=? AND params_json=?",
                (h, arch, key),
            ).fetchone()
        if row is None:
            return None
        return BenchmarkPoint(
            params=json.loads(row["params_json"]),
            time_ms=row["time_ms"],
            occupancy=row["occupancy"],
            source=row["source_kind"],           # type: ignore[arg-type]
            error=row["error"],
        )

    def put(
        self,
        source: str,
        arch: str,
        params: dict[str, Any],
        point: BenchmarkPoint,
    ) -> None:
        """Insert or replace a BenchmarkPoint in the cache."""
        key = json.dumps(params, sort_keys=True)
        h = self._hash(source)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO results
                   (source_hash, arch, params_json, time_ms, occupancy, source_kind, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (h, arch, key, point.time_ms, point.occupancy, point.source, point.error),
            )

    def clear(self, source: Optional[str] = None, arch: Optional[str] = None) -> int:
        """Delete cached entries.  Returns number of rows deleted."""
        with self._connect() as conn:
            if source and arch:
                h = self._hash(source)
                cur = conn.execute(
                    "DELETE FROM results WHERE source_hash=? AND arch=?", (h, arch)
                )
            elif source:
                h = self._hash(source)
                cur = conn.execute("DELETE FROM results WHERE source_hash=?", (h,))
            else:
                cur = conn.execute("DELETE FROM results")
            return cur.rowcount

    def stats(self) -> dict[str, int]:
        """Return summary statistics about the cache."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            gpu_only = conn.execute(
                "SELECT COUNT(*) FROM results WHERE source_kind='gpu'"
            ).fetchone()[0]
        return {"total": total, "gpu": gpu_only, "model": total - gpu_only}
