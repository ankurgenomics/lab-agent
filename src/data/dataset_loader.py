"""Real dataset loader for demo runs.

Supports:
  BBBC004 — Broad synthetic fluorescent cells (DAPI-like, grayscale TIFF)
  BBBC020 — Real HeLa cells, time-lapse, DAPI (c1) + tubulin (c5), RGB TIFF

Maps dataset images to the plate/well/channel/timepoint structure the
orchestrator and analysis pipeline already understand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"


# ------------------------------------------------------------------ #
#  BBBC004 loader                                                     #
# ------------------------------------------------------------------ #

def load_bbbc004(subset: str = "000") -> dict:
    """
    Map BBBC004 images to a fake 96-well plate.

    Returns:
        {well: {"DAPI": path_str}, ...}
    """
    img_dir = DATASETS_DIR / "bbbc004" / f"synthetic_{subset}_images"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"BBBC004 not found at {img_dir}. Run: python data/download_demo.py"
        )

    tifs = sorted(img_dir.glob("*.tif"))
    wells = [f"{chr(65 + i // 12)}{(i % 12) + 1}" for i in range(len(tifs))]

    return {
        well: {"DAPI": str(p)}
        for well, p in zip(wells, tifs)
    }


# ------------------------------------------------------------------ #
#  BBBC020 loader                                                     #
# ------------------------------------------------------------------ #

# The 6 biological conditions in order, as they appear in folder name prefixes.
# Each condition may have multiple replicate folders (e.g. "jw-1h 1", "jw-1h 2").
# Replicates within one condition are treated as separate wells on the same plate.
TIMEPOINT_ORDER = [
    ("jw-Kontrolle", "T000", "untreated (0 min)"),
    ("jw-15min",     "T001", "15 min Nocodazole"),
    ("jw-30min",     "T002", "30 min Nocodazole"),
    ("jw-1h",        "T003", "1 h Nocodazole"),
    ("jw-2h",        "T004", "2 h Nocodazole"),
    ("jw-24h",       "T005", "24 h Nocodazole"),
]

# Human-readable label for each timepoint key, used in reports
TIMEPOINT_LABELS: dict[str, str] = {tk: label for _, tk, label in TIMEPOINT_ORDER}


def load_bbbc020() -> dict:
    """
    Load BBBC020 as a time-lapse plate with 6 biological timepoints.

    BBBC020 has 25 folders named "jw-<condition> <replicate>".
    We group by the 6 biological conditions. Replicates within each
    condition become separate wells (A1, B1, C1 ...).

    Returns:
        {"T000": {well: {"DAPI": path, "FITC": path}}, ..., "T005": {...}}
        T000 = untreated control
        T001 = 15 min Nocodazole
        T002 = 30 min Nocodazole
        T003 = 1 h Nocodazole
        T004 = 2 h Nocodazole
        T005 = 24 h Nocodazole
    """
    root = DATASETS_DIR / "bbbc020" / "BBBC020_v1_images"
    if not root.exists():
        raise FileNotFoundError(
            f"BBBC020 not found at {root}. Run: python data/download_demo.py"
        )

    # Collect all valid folders once
    all_folders = [f for f in root.iterdir() if f.is_dir()]

    timelapse = {}
    for prefix, tp_key, _label in TIMEPOINT_ORDER:
        # Find all replicate folders for this condition
        replicates = sorted(
            [f for f in all_folders if f.name.startswith(prefix)],
            key=lambda p: p.name,
        )
        if not replicates:
            continue

        wells_at_tp = {}
        for rep_idx, folder in enumerate(replicates):
            well = f"{chr(65 + rep_idx)}1"   # A1, B1, C1 ...
            c1 = folder / f"{folder.name}_c1.TIF"
            c5 = folder / f"{folder.name}_c5.TIF"
            if c1.exists() and c5.exists():
                wells_at_tp[well] = {
                    "DAPI": str(c1),   # c1 = DAPI (nuclei, yellow pseudocolor)
                    "FITC": str(c5),   # c5 = tubulin (cytoskeleton, blue pseudocolor)
                }
        if wells_at_tp:
            timelapse[tp_key] = wells_at_tp

    return timelapse


def load_image_as_grayscale_uint16(path: str) -> np.ndarray:
    """Load any TIFF as uint16 grayscale.

    BBBC020 encoding:
      c1 (DAPI/nuclei):   RGB image, signal in R+G channels (yellow pseudocolor)
      c5 (tubulin):       RGB image, signal in B channel (blue pseudocolor)
    General fallback: luminance conversion.
    """
    img = Image.open(path)
    arr = np.array(img)
    path_str = str(path)

    if arr.ndim == 3:
        # Detect BBBC020 pseudocolor encoding
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        r_max, g_max, b_max = r.max(), g.max(), b.max()

        if b_max == 0 and r_max > 0:
            # Yellow channel (DAPI c1): use R or G
            arr = r.astype(np.uint16) * 256
        elif r_max == 0 and g_max == 0 and b_max > 0:
            # Blue channel (tubulin c5)
            arr = b.astype(np.uint16) * 256
        else:
            # Standard luminance
            arr = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint16)
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.uint16) * 256

    return arr.astype(np.uint16)


def describe_bbbc020() -> dict:
    """Quick summary of the loaded dataset for the CLI header."""
    try:
        tl = load_bbbc020()
        timepoints = list(tl.keys())
        first_tp_wells = list(tl[timepoints[0]].keys()) if timepoints else []
        return {
            "dataset": "BBBC020",
            "description": "Real HeLa cells, time-lapse, Nocodazole treatment (mitosis arrest)",
            "channels": ["DAPI (c1, nuclei)", "FITC/tubulin (c5, cytoskeleton)"],
            "n_timepoints": len(timepoints),
            "timepoint_keys": timepoints,
            "timepoint_labels": TIMEPOINT_LABELS,
            "n_wells_per_timepoint": len(first_tp_wells),
            "wells": first_tp_wells,
            "biology": (
                "Nocodazole arrests cells in mitosis. "
                "Mitotic index is expected to increase from T000 (untreated) to T005 (24h)."
            ),
        }
    except FileNotFoundError as e:
        return {"error": str(e)}
