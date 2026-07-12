"""Tests — run with pytest -q. No API key needed (mocks the LLM calls)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- instrument drivers ----

def test_microscope_simulation(tmp_path):
    from src.instruments.microscope import MicroscopeDriver, AcquisitionParams
    driver = MicroscopeDriver({"mode": "simulation"}, tmp_path)
    params = AcquisitionParams(
        plate_id="TEST01", wells=["A1"], channels=["DAPI", "FITC"]
    )
    result = driver.acquire_single(params)
    assert result.success
    assert "A1" in result.data["image_paths"]
    # Check TIFF was written
    path = Path(result.data["image_paths"]["A1"]["DAPI"])
    assert path.exists()


def test_microscope_timelapse(tmp_path):
    from src.instruments.microscope import MicroscopeDriver, AcquisitionParams
    driver = MicroscopeDriver({"mode": "simulation"}, tmp_path)
    params = AcquisitionParams(
        plate_id="TEST02", wells=["A1", "B1"], channels=["DAPI"],
        timelapse_interval_min=30, timelapse_duration_h=1.0,
    )
    result = driver.acquire_timelapse(params)
    assert result.success
    assert result.data["n_timepoints"] == 3  # T0, T30, T60
    assert "T000" in result.data["image_paths"]
    assert "T002" in result.data["image_paths"]


def test_cytometer_simulation(tmp_path):
    from src.instruments.cytometer import CytometerDriver, CytometryParams
    driver = CytometerDriver({"mode": "simulation"}, tmp_path)
    params = CytometryParams(
        plate_id="TEST01", sample_well="A1",
        channels=["FITC", "PE", "APC"], n_events=5000,
    )
    result = driver.acquire(params)
    assert result.success
    stats = result.data["population_stats"]
    assert 0 < stats["live_pct"] < 100
    assert "mitotic_pct" in stats


def test_cytometer_sort(tmp_path):
    from src.instruments.cytometer import CytometerDriver, CytometryParams
    driver = CytometerDriver({"mode": "simulation"}, tmp_path)
    params = CytometryParams(
        plate_id="TEST01", sample_well="A1",
        channels=["FITC", "PE"], n_events=10000,
        sort_gate={"channel": "FITC", "min": 0.6, "max": 1.0},
        sort_destination_plate="P002",
        sort_destination_well="A1",
    )
    result = driver.acquire(params)
    assert result.success
    assert "sorted_cells" in result.data
    assert result.data["sorted_cells"] > 0


def test_liquid_handler_simulation(tmp_path):
    from src.instruments.liquid_handler import LiquidHandlerDriver, LiquidHandlerParams
    driver = LiquidHandlerDriver({"mode": "simulation"}, tmp_path)
    result = driver.execute(LiquidHandlerParams(
        operation="add_reagent",
        dest_plate="P001",
        wells=["A1", "A2", "A3"],
        reagent="DAPI",
        volume_ul=5.0,
    ))
    assert result.success
    assert result.data["status"] == "complete"


# ---- analysis pipeline ----

def test_analysis_pipeline(tmp_path):
    from src.instruments.microscope import MicroscopeDriver, AcquisitionParams
    from src.analysis import AnalysisPipeline

    # Acquire images first
    driver = MicroscopeDriver({"mode": "simulation"}, tmp_path)
    params = AcquisitionParams(
        plate_id="PTEST", wells=["A1", "A2"], channels=["DAPI", "FITC"]
    )
    acq = driver.acquire_single(params)
    assert acq.success

    # Analyze
    pipeline = AnalysisPipeline({"mode": "simulation"}, tmp_path)
    result = pipeline.analyze_plate("PTEST", acq.data["image_paths"], timepoint=0)
    assert len(result.well_results) == 2
    for wr in result.well_results:
        assert wr.n_cells > 0
        assert 0.0 <= wr.mitotic_index <= 1.0

    summary = result.summary()
    assert "avg_mitotic_index_pct" in summary
    assert summary["n_wells_analyzed"] == 2


# ---- database ----

def test_database(tmp_path):
    from src.data import Database
    db = Database(f"sqlite:///{tmp_path}/test.db")

    db.create_experiment("EXP-001", "Test experiment", "run mitosis assay on P001")
    db.log_event("EXP-001", "tool_call", "acquire_images", {"plate": "P001"})
    db.complete_experiment("EXP-001")

    exps = db.list_experiments()
    assert len(exps) == 1
    assert exps[0]["status"] == "completed"
