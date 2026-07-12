"""M2: Protocol Validator — safety layer between the LLM and instruments.

Every tool call passes through here before dispatch. Physical limits are
enforced; impossible requests are rejected with a human-readable reason
so the LLM can self-correct.

No LLM is involved — this is 100% deterministic. That's the point.
"""
from __future__ import annotations

from typing import Any


# ── Physical instrument limits ─────────────────────────────────────────────
LIMITS = {
    "microscope": {
        "valid_channels": {"DAPI", "FITC", "TRITC", "CY5", "GFP", "mCherry", "BF"},
        "valid_objectives": {"4x", "10x", "20x", "40x", "60x", "100x"},
        "max_wells_per_run": 96,
    },
    "liquid_handler": {
        "min_volume_ul": 0.5,
        "max_volume_ul": 1000.0,
        "valid_well_formats": None,   # checked dynamically via _parse_well
    },
    "cytometer": {
        "valid_channels": {"FSC", "SSC", "FITC", "PE", "APC", "BV421", "BV605", "PE-Cy7"},
        "min_events": 100,
        "max_events": 1_000_000,
    },
    "incubator": {
        "temp_range_c": (4.0, 42.0),
        "co2_range_pct": (0.0, 10.0),
        "humidity_range_pct": (0.0, 99.0),
    },
}

VALID_WELL_RE_ROWS = "ABCDEFGH"
VALID_WELL_RE_COLS = set(range(1, 13))   # 1–12 for 96-well plate


def _parse_well(well: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Accepts 'A1'..'H12' or 'all'."""
    if well == "all":
        return True, ""
    if not well or len(well) < 2:
        return False, f"Well '{well}' too short — expected format like 'A1' or 'H12'."
    row = well[0].upper()
    col_str = well[1:]
    if row not in VALID_WELL_RE_ROWS:
        return False, f"Well row '{row}' invalid — must be A–H for a 96-well plate."
    try:
        col = int(col_str)
    except ValueError:
        return False, f"Well column '{col_str}' is not a number."
    if col not in VALID_WELL_RE_COLS:
        return False, f"Well column {col} out of range — 96-well plates have columns 1–12."
    return True, ""


# ── Protocol step model (plain dataclass, no Pydantic dependency) ──────────
class ValidationResult:
    def __init__(self, valid: bool, errors: list[str] = None):
        self.valid = valid
        self.errors = errors or []

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": self.errors}


class ProtocolValidator:
    """Validates a single tool call or an entire protocol (list of steps)."""

    # ── Single tool call validation ────────────────────────────────────────

    def validate_tool_call(self, tool_name: str, params: dict) -> ValidationResult:
        """Check one LLM tool call against physical limits.

        Returns ValidationResult(valid=True) if safe to dispatch,
        or ValidationResult(valid=False, errors=[...]) with reasons.
        """
        errors: list[str] = []

        if tool_name == "setup_plate":
            errors.extend(self._check_setup_plate(params))
        elif tool_name == "acquire_images":
            errors.extend(self._check_acquire_images(params))
        elif tool_name == "analyze_images":
            errors.extend(self._check_analyze_images(params))
        elif tool_name == "run_flow_cytometry":
            errors.extend(self._check_cytometry(params))
        elif tool_name == "transfer_cells":
            errors.extend(self._check_transfer(params))
        elif tool_name in ("query_results", "query_lims", "wait_for_job",
                           "plan_experiment", "execute_protocol", "list_protocols"):
            pass  # read-only or control tools — always safe
        else:
            errors.append(f"Unknown tool '{tool_name}'. Cannot validate.")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    # ── Protocol-level validation (M4) ────────────────────────────────────

    def validate_protocol(self, protocol: dict) -> ValidationResult:
        """Validate an LLM-generated protocol (list of steps).

        Each step must have: instrument, operation, params, expected_duration_min.
        """
        errors: list[str] = []

        if not isinstance(protocol.get("steps"), list) or not protocol["steps"]:
            errors.append("Protocol must contain a non-empty 'steps' list.")
            return ValidationResult(valid=False, errors=errors)

        for i, step in enumerate(protocol["steps"]):
            step_errors = self._validate_protocol_step(i, step)
            errors.extend(step_errors)

        # Sanity: total estimated duration must be > 0
        total_min = sum(
            s.get("expected_duration_min", 0) for s in protocol["steps"]
        )
        if total_min <= 0:
            errors.append(
                "Total protocol duration is 0 minutes — each step must have "
                "a positive expected_duration_min."
            )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate_protocol_step(self, idx: int, step: dict) -> list[str]:
        errors: list[str] = []
        prefix = f"Step {idx + 1}"

        instrument = step.get("instrument", "")
        operation = step.get("operation", "")
        params = step.get("params", {})

        if not instrument:
            errors.append(f"{prefix}: missing 'instrument' field.")
        if not operation:
            errors.append(f"{prefix}: missing 'operation' field.")

        dur = step.get("expected_duration_min")
        if dur is None:
            errors.append(f"{prefix}: missing 'expected_duration_min'.")
        elif not isinstance(dur, (int, float)) or dur < 0:
            errors.append(f"{prefix}: expected_duration_min must be a non-negative number.")

        # Map operation → tool_name for reuse of single-call validators
        OPERATION_MAP = {
            "acquire": "acquire_images",
            "acquire_images": "acquire_images",
            "image": "acquire_images",
            "stain": "setup_plate",
            "setup_plate": "setup_plate",
            "prepare": "setup_plate",
            "cytometry": "run_flow_cytometry",
            "flow_cytometry": "run_flow_cytometry",
            "transfer": "transfer_cells",
            "transfer_cells": "transfer_cells",
            "analyze": "analyze_images",
        }
        mapped = OPERATION_MAP.get(operation.lower())
        if mapped and params:
            r = self.validate_tool_call(mapped, params)
            for e in r.errors:
                errors.append(f"{prefix} ({operation}): {e}")

        return errors

    # ── Per-tool checkers ──────────────────────────────────────────────────

    def _check_setup_plate(self, p: dict) -> list[str]:
        errors: list[str] = []
        volume = p.get("volume_ul", 100)
        lims = LIMITS["liquid_handler"]
        if not (lims["min_volume_ul"] <= volume <= lims["max_volume_ul"]):
            errors.append(
                f"volume_ul={volume} out of range "
                f"[{lims['min_volume_ul']}, {lims['max_volume_ul']}] µL."
            )
        wells = p.get("wells", [])
        if isinstance(wells, str):
            wells = [wells]
        for w in wells:
            valid, reason = _parse_well(w)
            if not valid:
                errors.append(reason)
        return errors

    def _check_acquire_images(self, p: dict) -> list[str]:
        errors: list[str] = []
        mscope = LIMITS["microscope"]

        channels = p.get("channels", [])
        for ch in channels:
            if ch not in mscope["valid_channels"]:
                errors.append(
                    f"Channel '{ch}' not available on this microscope. "
                    f"Valid channels: {sorted(mscope['valid_channels'])}."
                )

        obj = p.get("objective")
        if obj and obj not in mscope["valid_objectives"]:
            errors.append(
                f"Objective '{obj}' not available. "
                f"Valid objectives: {sorted(mscope['valid_objectives'])}."
            )

        wells = p.get("wells", [])
        if isinstance(wells, str):
            wells = [wells]
        if wells != ["all"] and len(wells) > mscope["max_wells_per_run"]:
            errors.append(
                f"Requested {len(wells)} wells but microscope supports max "
                f"{mscope['max_wells_per_run']} per run."
            )
        for w in wells:
            valid, reason = _parse_well(w)
            if not valid:
                errors.append(reason)

        interval = p.get("timelapse_interval_min")
        duration = p.get("timelapse_duration_h")
        if (interval is None) != (duration is None):
            errors.append(
                "For timelapse, both timelapse_interval_min and timelapse_duration_h "
                "must be provided together."
            )
        if interval is not None and interval < 1:
            errors.append(
                f"timelapse_interval_min={interval} is too short — "
                "minimum is 1 minute (instrument settle time)."
            )

        return errors

    def _check_analyze_images(self, p: dict) -> list[str]:
        errors: list[str] = []
        tp = p.get("timepoint", 0)
        if not isinstance(tp, int) or tp < -1:
            errors.append(
                f"timepoint={tp} invalid — must be an integer >= 0, or -1 for all timepoints."
            )
        return errors

    def _check_cytometry(self, p: dict) -> list[str]:
        errors: list[str] = []
        cyto = LIMITS["cytometer"]

        channels = p.get("channels", ["FITC", "PE", "APC"])
        for ch in channels:
            if ch not in cyto["valid_channels"]:
                errors.append(
                    f"Cytometer channel '{ch}' not available. "
                    f"Valid: {sorted(cyto['valid_channels'])}."
                )

        n = p.get("n_events", 10000)
        if not (cyto["min_events"] <= n <= cyto["max_events"]):
            errors.append(
                f"n_events={n} out of range "
                f"[{cyto['min_events']}, {cyto['max_events']}]."
            )

        well = p.get("well", "")
        valid, reason = _parse_well(well)
        if not valid:
            errors.append(reason)

        gate = p.get("sort_gate")
        if gate:
            for key in ("channel", "min", "max"):
                if key not in gate:
                    errors.append(f"sort_gate missing required field '{key}'.")
            if "min" in gate and "max" in gate:
                if gate["min"] >= gate["max"]:
                    errors.append(
                        f"sort_gate min ({gate['min']}) must be less than max ({gate['max']})."
                    )

        return errors

    def _check_transfer(self, p: dict) -> list[str]:
        errors: list[str] = []
        src_wells = p.get("source_wells", [])
        dst_wells = p.get("dest_wells", [])
        if len(src_wells) != len(dst_wells):
            errors.append(
                f"source_wells ({len(src_wells)}) and dest_wells ({len(dst_wells)}) "
                "must have the same length."
            )
        volume = p.get("volume_ul", 100)
        lims = LIMITS["liquid_handler"]
        if not (lims["min_volume_ul"] <= volume <= lims["max_volume_ul"]):
            errors.append(
                f"volume_ul={volume} out of range "
                f"[{lims['min_volume_ul']}, {lims['max_volume_ul']}] µL."
            )
        for w in src_wells + dst_wells:
            valid, reason = _parse_well(w)
            if not valid:
                errors.append(reason)
        return errors
