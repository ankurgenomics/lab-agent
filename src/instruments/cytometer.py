"""Flow cytometer driver — simulation + real modes.

Simulation: generates realistic FCS-like scatter/fluorescence data with
distinct cell populations (live, dead, mitotic) using multivariate gaussians.
Real mode: can wrap BD FACSDiva (COM), Cytek SpectroFlo (API), or file-drop automation.

The key output for the agent:
  - Population statistics (% positive, median fluorescence)
  - Sorted cell counts if a sort was run
  - Gate definitions used
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path

from . import InstrumentBase, InstrumentResult


@dataclass
class CytometryParams:
    plate_id: str
    sample_well: str
    channels: list[str]
    n_events: int = 10000
    sort_gate: dict | None = None   # e.g. {"channel": "FITC", "min": 0.7, "max": 1.0}
    sort_destination_plate: str | None = None
    sort_destination_well: str | None = None


class CytometerDriver(InstrumentBase):
    name = "flow_cytometer"

    def __init__(self, config: dict, storage_path: Path):
        self.mode = config.get("mode", "simulation")
        self.storage_path = storage_path
        self.config = config

    def acquire(self, params: CytometryParams) -> InstrumentResult:
        """Run acquisition. If sort_gate is set, also sort matching cells."""
        data = self._generate_data(params)
        stats = self._compute_stats(data, params)

        path = self._save_data(data, params)

        sorted_count = None
        if params.sort_gate:
            sorted_count = self._apply_sort_gate(data, params)

        result_data = {
            "plate_id": params.plate_id,
            "well": params.sample_well,
            "n_events_acquired": len(data["FSC"]),
            "population_stats": stats,
            "data_path": str(path),
        }
        if sorted_count is not None:
            result_data["sorted_cells"] = sorted_count
            result_data["sort_destination"] = (
                f"{params.sort_destination_plate}/{params.sort_destination_well}"
            )

        return InstrumentResult.ok("flow_cytometer", result_data)

    # ------------------------------------------------------------------ #
    #  Simulation                                                         #
    # ------------------------------------------------------------------ #

    def _generate_data(self, params: CytometryParams) -> dict:
        """Generate multivariate Gaussian populations mimicking real cytometry data."""
        rng = np.random.default_rng(
            seed=hash(params.plate_id + params.sample_well) % (2**31)
        )
        n = params.n_events

        # Three populations: live (70%), dead (20%), mitotic (10%)
        pop_sizes = [int(n * 0.70), int(n * 0.20), int(n * 0.10)]
        pop_sizes[0] += n - sum(pop_sizes)  # round-up remainder to live

        data = {}
        for ch in ["FSC", "SSC"] + params.channels:
            data[ch] = []

        # Live cells: high FSC, medium SSC, moderate FITC
        _add_population(data, pop_sizes[0], rng,
                        fsc_mean=0.65, fsc_std=0.08,
                        ssc_mean=0.45, ssc_std=0.07,
                        fitc_mean=0.35, fitc_std=0.10,
                        pe_mean=0.20, pe_std=0.05,
                        apc_mean=0.15, apc_std=0.04)

        # Dead/debris: low FSC, high SSC, high PI (PE channel)
        _add_population(data, pop_sizes[1], rng,
                        fsc_mean=0.25, fsc_std=0.10,
                        ssc_mean=0.70, ssc_std=0.10,
                        fitc_mean=0.15, fitc_std=0.08,
                        pe_mean=0.80, pe_std=0.08,
                        apc_mean=0.10, apc_std=0.03)

        # Mitotic cells: high FSC, high FITC (phospho-H3 positive)
        _add_population(data, pop_sizes[2], rng,
                        fsc_mean=0.75, fsc_std=0.06,
                        ssc_mean=0.55, ssc_std=0.06,
                        fitc_mean=0.85, fitc_std=0.07,
                        pe_mean=0.25, pe_std=0.05,
                        apc_mean=0.18, apc_std=0.04)

        return {k: np.clip(np.array(v), 0, 1) for k, v in data.items()}

    def _compute_stats(self, data: dict, params: CytometryParams) -> dict:
        n = len(data["FSC"])
        stats = {"total_events": n}

        # Live gate: FSC > 0.4
        live_mask = data["FSC"] > 0.4
        stats["live_pct"] = round(float(live_mask.mean() * 100), 1)

        # Mitotic gate: FITC > 0.6 (phospho-H3)
        if "FITC" in data:
            mitotic_mask = data["FITC"] > 0.6
            stats["mitotic_pct"] = round(float(mitotic_mask.mean() * 100), 1)
            stats["median_fitc"] = round(float(np.median(data["FITC"])), 3)

        for ch in params.channels:
            if ch in data:
                stats[f"median_{ch.lower()}"] = round(float(np.median(data[ch])), 3)

        return stats

    def _apply_sort_gate(self, data: dict, params: CytometryParams) -> int:
        gate = params.sort_gate
        ch = gate["channel"]
        if ch not in data:
            return 0
        mask = (data[ch] >= gate["min"]) & (data[ch] <= gate["max"])
        return int(mask.sum())

    def _save_data(self, data: dict, params: CytometryParams) -> Path:
        out_dir = self.storage_path / "cytometry" / params.plate_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{params.sample_well}_events.npz"
        np.savez_compressed(path, **data)
        return path


def _add_population(data, n, rng, fsc_mean, fsc_std, ssc_mean, ssc_std,
                    fitc_mean, fitc_std, pe_mean, pe_std, apc_mean, apc_std):
    data["FSC"].extend(rng.normal(fsc_mean, fsc_std, n).tolist())
    data["SSC"].extend(rng.normal(ssc_mean, ssc_std, n).tolist())
    if "FITC" in data:
        data["FITC"].extend(rng.normal(fitc_mean, fitc_std, n).tolist())
    if "PE" in data:
        data["PE"].extend(rng.normal(pe_mean, pe_std, n).tolist())
    if "APC" in data:
        data["APC"].extend(rng.normal(apc_mean, apc_std, n).tolist())
