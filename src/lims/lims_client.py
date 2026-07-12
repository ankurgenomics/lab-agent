"""M3: Local LIMS (Laboratory Information Management System).

Stores analysis results with full provenance — plate map, well coordinates,
metric name, value, unit, instrument, timepoint, and experiment lineage.

The LLM reads from here via the query_lims() tool. Writing happens
automatically after every analyze_images() and run_flow_cytometry() call.

In production: swap SQLite URL for Benchling API or your institution's LIMS.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class LocalLIMS:
    """
    Flat SQLite LIMS — one row per (experiment, plate, well, timepoint, metric).
    Designed to be queryable by the LLM via simple filters.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS lims_results (
            id          TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            plate_id    TEXT NOT NULL,
            well        TEXT,
            timepoint   INTEGER DEFAULT 0,
            metric      TEXT NOT NULL,
            value       REAL,
            unit        TEXT,
            instrument  TEXT,
            method      TEXT,
            acquired_at TEXT,
            provenance  TEXT    -- JSON: protocol_id, operator, software_version
        );
        CREATE INDEX IF NOT EXISTS idx_lims_exp
            ON lims_results(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_lims_plate
            ON lims_results(plate_id, well, timepoint);
        CREATE INDEX IF NOT EXISTS idx_lims_metric
            ON lims_results(metric);
    """

    def __init__(self, db_path: str = "lab_lims.db"):
        self.db_path = str(db_path)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(self.SCHEMA)
        conn.close()

    # ── Writes ─────────────────────────────────────────────────────────────

    def write_result(self, experiment_id: str, plate_id: str, well: str,
                     timepoint: int, metric: str, value: float,
                     unit: str = "", instrument: str = "",
                     method: str = "", provenance: dict = None):
        """Insert one measurement. Idempotent on (exp, plate, well, tp, metric)."""
        import json
        row_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        prov_str = json.dumps(provenance or {})
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO lims_results
                (id, experiment_id, plate_id, well, timepoint,
                 metric, value, unit, instrument, method, acquired_at, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (row_id, experiment_id, plate_id, well, timepoint,
             metric, value, unit, instrument, method, now, prov_str),
        )
        conn.commit()
        conn.close()

    def write_plate_results(self, experiment_id: str, plate_id: str,
                            timepoint: int, per_well: dict[str, dict],
                            instrument: str = "microscope", method: str = "watershed"):
        """
        Bulk-write per-well analysis results.

        per_well format:
          {
            "A1": {"cells": 142, "mitotic_pct": 14.7, "live_pct": 85.0,
                   "mean_dapi": 4200, "mean_fitc": 2100},
            ...
          }
        """
        METRIC_UNITS = {
            "cells": "count",
            "mitotic_pct": "%",
            "live_pct": "%",
            "mean_dapi": "AU",
            "mean_fitc": "AU",
        }
        for well, metrics in per_well.items():
            for metric, value in metrics.items():
                if value is None:
                    continue
                self.write_result(
                    experiment_id=experiment_id,
                    plate_id=plate_id,
                    well=well,
                    timepoint=timepoint,
                    metric=metric,
                    value=float(value),
                    unit=METRIC_UNITS.get(metric, ""),
                    instrument=instrument,
                    method=method,
                )

    # ── Queries (called by the LLM via query_lims tool) ────────────────────

    def query(self, experiment_id: str = None, plate_id: str = None,
              well: str = None, timepoint: int = None,
              metric: str = None,
              min_value: float = None, max_value: float = None,
              limit: int = 200) -> list[dict]:
        """Flexible query — any combination of filters."""
        sql = "SELECT experiment_id, plate_id, well, timepoint, metric, value, unit, instrument, acquired_at FROM lims_results WHERE 1=1"
        params: list[Any] = []

        if experiment_id:
            sql += " AND experiment_id=?"; params.append(experiment_id)
        if plate_id:
            sql += " AND plate_id=?"; params.append(plate_id)
        if well:
            sql += " AND well=?"; params.append(well)
        if timepoint is not None:
            sql += " AND timepoint=?"; params.append(timepoint)
        if metric:
            sql += " AND metric=?"; params.append(metric)
        if min_value is not None:
            sql += " AND value>=?"; params.append(min_value)
        if max_value is not None:
            sql += " AND value<=?"; params.append(max_value)

        sql += " ORDER BY plate_id, well, timepoint, metric LIMIT ?"
        params.append(limit)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        return [
            {
                "experiment_id": r[0],
                "plate_id": r[1],
                "well": r[2],
                "timepoint": r[3],
                "metric": r[4],
                "value": r[5],
                "unit": r[6],
                "instrument": r[7],
                "acquired_at": r[8],
            }
            for r in rows
        ]

    def summary(self, experiment_id: str, plate_id: str = None,
                metric: str = "mitotic_pct") -> dict:
        """Return per-well summary for a metric — min, max, mean, and top well."""
        rows = self.query(experiment_id=experiment_id, plate_id=plate_id, metric=metric)
        if not rows:
            return {"error": f"No LIMS data for metric '{metric}' in experiment {experiment_id}."}

        values = [r["value"] for r in rows if r["value"] is not None]
        if not values:
            return {"error": "All values are None."}

        top = max(rows, key=lambda r: r["value"] or 0)
        return {
            "metric": metric,
            "n_measurements": len(values),
            "mean": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "top_well": top["well"],
            "top_timepoint": top["timepoint"],
            "top_value": round(top["value"], 2),
            "unit": rows[0]["unit"],
        }

    def timecourse(self, experiment_id: str, plate_id: str,
                   well: str, metric: str = "mitotic_pct") -> list[dict]:
        """Return a well's metric over all timepoints — for trend analysis."""
        rows = self.query(
            experiment_id=experiment_id,
            plate_id=plate_id,
            well=well,
            metric=metric,
        )
        return sorted(rows, key=lambda r: r["timepoint"])
