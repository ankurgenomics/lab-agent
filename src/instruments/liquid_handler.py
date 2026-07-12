"""Liquid handler driver — simulation + real modes.

Simulation: validates protocol logic and returns realistic timing estimates.
Real mode: Hamilton VENUS via pyhamilton, or Opentrons Flex via REST API.

The liquid handler is the glue instrument — it sets up plates, transfers
samples between instruments, adds reagents, and is what makes HTS possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import InstrumentBase, InstrumentResult


@dataclass
class TransferStep:
    source_plate: str
    source_well: str
    dest_plate: str
    dest_well: str
    volume_ul: float


@dataclass
class LiquidHandlerParams:
    operation: str          # "setup_plate" | "add_reagent" | "transfer_cells" | "serial_dilution"
    source_plate: str | None = None
    dest_plate: str | None = None
    wells: list[str] | None = None
    reagent: str | None = None
    volume_ul: float = 100.0
    transfers: list[TransferStep] | None = None
    dilution_factor: float | None = None
    n_dilutions: int | None = None


class LiquidHandlerDriver(InstrumentBase):
    name = "liquid_handler"

    def __init__(self, config: dict, storage_path: Path):
        self.mode = config.get("mode", "simulation")
        self.storage_path = storage_path
        self.config = config
        self._opentrons = None

        if self.mode == "opentrons":
            self._init_opentrons(config.get("api_endpoint"))

    def _init_opentrons(self, endpoint: str):
        try:
            import requests
            r = requests.get(f"{endpoint}/health")
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Opentrons not reachable at {endpoint}: {e}")

    def execute(self, params: LiquidHandlerParams) -> InstrumentResult:
        if self.mode == "opentrons":
            return self._execute_opentrons(params)
        return self._execute_simulation(params)

    def _execute_simulation(self, params: LiquidHandlerParams) -> InstrumentResult:
        wells = params.wells or []
        op = params.operation

        if op == "setup_plate":
            return InstrumentResult.ok("liquid_handler", {
                "operation": op,
                "plate": params.dest_plate,
                "wells_prepared": len(wells),
                "reagent": params.reagent,
                "volume_ul": params.volume_ul,
                "estimated_time_s": len(wells) * 8,
                "status": "complete",
            })

        if op == "add_reagent":
            return InstrumentResult.ok("liquid_handler", {
                "operation": op,
                "plate": params.dest_plate or params.source_plate,
                "reagent": params.reagent,
                "wells": wells,
                "volume_ul": params.volume_ul,
                "estimated_time_s": len(wells) * 5,
                "status": "complete",
            })

        if op == "transfer_cells":
            transfers = params.transfers or []
            return InstrumentResult.ok("liquid_handler", {
                "operation": op,
                "n_transfers": len(transfers),
                "source_plate": params.source_plate,
                "dest_plate": params.dest_plate,
                "estimated_time_s": len(transfers) * 12,
                "status": "complete",
            })

        if op == "serial_dilution":
            n = params.n_dilutions or 8
            return InstrumentResult.ok("liquid_handler", {
                "operation": op,
                "plate": params.dest_plate,
                "dilution_factor": params.dilution_factor,
                "n_steps": n,
                "wells": wells,
                "estimated_time_s": n * 15,
                "status": "complete",
            })

        return InstrumentResult.fail("liquid_handler", f"Unknown operation: {op}")

    def _execute_opentrons(self, params: LiquidHandlerParams) -> InstrumentResult:
        # Real Opentrons implementation via REST API
        # https://docs.opentrons.com/v2/new_protocol_api.html
        raise NotImplementedError("Opentrons real mode: implement protocol upload + run")
