"""Base instrument class. All instruments inherit from this.

To add a real instrument:
  1. Subclass InstrumentBase
  2. Override the methods with real API calls
  3. Set mode = "real" in config.yaml
  4. The rest of the system is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class InstrumentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class InstrumentResult:
    success: bool
    job_id: str
    instrument: str
    started_at: str
    completed_at: str | None
    data: dict = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, instrument: str, data: dict) -> "InstrumentResult":
        now = datetime.utcnow().isoformat()
        return cls(True, str(uuid.uuid4())[:8], instrument, now, now, data)

    @classmethod
    def fail(cls, instrument: str, error: str) -> "InstrumentResult":
        now = datetime.utcnow().isoformat()
        return cls(False, str(uuid.uuid4())[:8], instrument, now, now, {}, error)


class InstrumentBase:
    name: str = "base"
    mode: str = "simulation"

    def status(self) -> InstrumentStatus:
        return InstrumentStatus.IDLE

    def ping(self) -> bool:
        return True
