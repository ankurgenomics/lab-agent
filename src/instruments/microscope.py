"""Microscope driver — simulation + real (pymmcore-plus) modes.

Simulation mode: generates realistic numpy arrays representing fluorescence images
with synthetic cells (bright spots = nuclei, ring structures = cytoplasm).
Real mode: wraps pymmcore-plus which speaks to Micro-Manager, which controls
Leica/Zeiss/Nikon/Olympus hardware via the same interface.

Swap simulation → real:
  config.yaml: instruments.microscope.mode = "pymmcore"
  pip install pymmcore-plus
  Set config_file to your Micro-Manager .cfg
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import InstrumentBase, InstrumentResult


@dataclass
class AcquisitionParams:
    plate_id: str
    wells: list[str]            # e.g. ["A1", "A2", "B3"]
    channels: list[str]         # e.g. ["DAPI", "FITC"]
    objective: str = "20x"
    z_slices: int = 1
    z_step_um: float = 1.0
    exposure_ms: dict = None    # per-channel exposure
    timelapse_interval_min: float | None = None
    timelapse_duration_h: float | None = None

    def __post_init__(self):
        if self.exposure_ms is None:
            self.exposure_ms = {ch: 100 for ch in self.channels}


class MicroscopeDriver(InstrumentBase):
    name = "microscope"

    def __init__(self, config: dict, storage_path: Path):
        self.mode = config.get("mode", "simulation")
        self.storage_path = storage_path
        self.config = config
        self._core = None   # pymmcore-plus core (real mode only)

        if self.mode == "pymmcore":
            self._init_real(config.get("config_file"))

    def _init_real(self, config_file: str):
        try:
            import pymmcore_plus
            self._core = pymmcore_plus.CMMCorePlus()
            if config_file:
                self._core.loadSystemConfiguration(config_file)
        except ImportError:
            raise RuntimeError(
                "pymmcore-plus not installed. Run: pip install pymmcore-plus"
            )

    # ------------------------------------------------------------------ #
    #  Public interface (same signature in both modes)                    #
    # ------------------------------------------------------------------ #

    def acquire_single(self, params: AcquisitionParams) -> InstrumentResult:
        """Acquire a single timepoint across specified wells and channels."""
        images = {}
        for well in params.wells:
            images[well] = {}
            for channel in params.channels:
                img = self._acquire_field(well, channel, params)
                path = self._save_image(img, params.plate_id, well, channel, t=0)
                images[well][channel] = str(path)

        return InstrumentResult.ok("microscope", {
            "plate_id": params.plate_id,
            "wells_acquired": params.wells,
            "channels": params.channels,
            "timepoint": 0,
            "image_paths": images,
        })

    def acquire_timelapse(self, params: AcquisitionParams,
                          progress_callback=None) -> InstrumentResult:
        """Acquire time-lapse series. Returns paths for all timepoints."""
        if not params.timelapse_interval_min or not params.timelapse_duration_h:
            return InstrumentResult.fail("microscope",
                                         "timelapse_interval_min and timelapse_duration_h required")

        n_timepoints = int(
            (params.timelapse_duration_h * 60) / params.timelapse_interval_min
        ) + 1
        all_images = {}

        for t in range(n_timepoints):
            t_images = {}
            for well in params.wells:
                t_images[well] = {}
                for channel in params.channels:
                    img = self._acquire_field(well, channel, params, timepoint=t)
                    path = self._save_image(img, params.plate_id, well, channel, t=t)
                    t_images[well][channel] = str(path)
            all_images[f"T{t:03d}"] = t_images

            if progress_callback:
                progress_callback(t, n_timepoints, t_images)

            # In real mode this would be a real wait; simulation skips it
            if self.mode != "simulation" and t < n_timepoints - 1:
                time.sleep(params.timelapse_interval_min * 60)

        return InstrumentResult.ok("microscope", {
            "plate_id": params.plate_id,
            "wells_acquired": params.wells,
            "channels": params.channels,
            "n_timepoints": n_timepoints,
            "interval_min": params.timelapse_interval_min,
            "duration_h": params.timelapse_duration_h,
            "image_paths": all_images,
        })

    # ------------------------------------------------------------------ #
    #  Internal acquisition                                               #
    # ------------------------------------------------------------------ #

    def _acquire_field(self, well: str, channel: str,
                       params: AcquisitionParams, timepoint: int = 0) -> np.ndarray:
        if self.mode == "pymmcore" and self._core:
            return self._acquire_real(channel, params.exposure_ms.get(channel, 100))
        return self._simulate_image(well, channel, timepoint)

    def _acquire_real(self, channel: str, exposure_ms: int) -> np.ndarray:
        self._core.setExposure(exposure_ms)
        self._core.snapImage()
        return self._core.getImage()

    def _simulate_image(self, well: str, channel: str,
                        timepoint: int = 0) -> np.ndarray:
        """Generate a synthetic fluorescence image with realistic cell-like structures."""
        rng = np.random.default_rng(
            seed=hash(well + channel) % (2**31) + timepoint
        )
        img = np.zeros((512, 512), dtype=np.uint16)

        # Background + noise
        img += rng.integers(100, 300, size=(512, 512), dtype=np.uint16)

        # Place synthetic cells (bright nuclei in DAPI, ring in FITC)
        n_cells = rng.integers(40, 100)
        for _ in range(n_cells):
            cx, cy = rng.integers(20, 492, size=2)
            radius = rng.integers(8, 18)
            yy, xx = np.ogrid[:512, :512]
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

            if channel == "DAPI":
                # Solid nucleus
                mask = dist < radius
                intensity = rng.integers(3000, 8000)
                # ~15% mitotic: brighter, rounder
                if rng.random() < 0.15 + timepoint * 0.03:
                    intensity = rng.integers(7000, 12000)
                img[mask] = np.clip(img[mask] + intensity, 0, 65535)

            elif channel == "FITC":
                # Ring (cytoplasm)
                ring = (dist > radius) & (dist < radius + 6)
                img[ring] = np.clip(
                    img[ring] + rng.integers(1500, 4000), 0, 65535
                )

            elif channel == "TRITC":
                # Punctate (e.g. mitochondria)
                if dist.min() < radius:
                    for _ in range(rng.integers(3, 8)):
                        px = min(511, max(0, cx + rng.integers(-10, 10)))
                        py = min(511, max(0, cy + rng.integers(-10, 10)))
                        img[py, px] = np.clip(
                            img[py, px] + rng.integers(5000, 15000), 0, 65535
                        )

        return img

    def _save_image(self, img: np.ndarray, plate_id: str,
                    well: str, channel: str, t: int) -> Path:
        from PIL import Image
        out_dir = self.storage_path / "images" / plate_id / well
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"T{t:03d}_{channel}.tiff"
        # Save as 16-bit TIFF
        pil_img = Image.fromarray(img.astype(np.int32), mode="I")
        pil_img.save(path)
        return path
