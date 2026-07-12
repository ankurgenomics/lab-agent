"""SQLite metadata database — every experiment, acquisition, result, event, job, and protocol.

In production: swap DATABASE_URL for Postgres.
  pip install psycopg2-binary
  export DATABASE_URL=postgresql://user:pass@host:5432/labagent
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, JSON, ForeignKey, Text, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    command = Column(Text)               # original English command
    status = Column(String, default="running")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    acquisitions = relationship("Acquisition", back_populates="experiment")
    results = relationship("AnalysisResult", back_populates="experiment")
    events = relationship("ExperimentEvent", back_populates="experiment")


# ── M1: Async job tracking ─────────────────────────────────────────
class InstrumentJob(Base):
    """Tracks async instrument jobs (acquire, cytometry, liquid handler).

    When an instrument call is dispatched asynchronously, a row is inserted
    with status='running'. The background thread updates it to 'done' or
    'error' when the instrument finishes. The LLM polls via wait_for_job().
    """
    __tablename__ = "instrument_jobs"

    id = Column(String, primary_key=True)           # UUID
    experiment_id = Column(String, ForeignKey("experiments.id"))
    instrument = Column(String)                      # microscope | cytometer | liquid_handler
    tool_name = Column(String)                       # acquire_images | run_flow_cytometry | …
    params = Column(JSON, default=dict)              # original tool input params
    status = Column(String, default="running")       # running | done | error
    result = Column(JSON, nullable=True)             # filled when done
    error_message = Column(Text, nullable=True)      # filled when error
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    eta_seconds = Column(Integer, nullable=True)     # estimated time to completion


# ── M4: Protocol planning and execution ───────────────────────────
class Protocol(Base):
    """An LLM-generated, validated experiment protocol.

    Flow: LLM generates → validator approves → human sets approval_status='approved'
    → executor runs → status='completed'.
    """
    __tablename__ = "protocols"

    id = Column(String, primary_key=True)           # UUID
    experiment_id = Column(String, ForeignKey("experiments.id"))
    goal = Column(Text)                              # original scientific goal
    hypothesis = Column(Text, nullable=True)         # LLM's stated hypothesis
    steps = Column(JSON, default=list)               # list of protocol steps
    success_criteria = Column(Text, nullable=True)
    estimated_duration_h = Column(Float, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    validation_status = Column(String, default="pending")   # pending | valid | invalid
    validation_errors = Column(JSON, default=list)
    approval_status = Column(String, default="pending_approval")  # pending_approval | approved | rejected
    approval_note = Column(Text, nullable=True)
    execution_status = Column(String, default="not_started")  # not_started | running | completed | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class Acquisition(Base):
    __tablename__ = "acquisitions"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id"))
    instrument = Column(String)          # microscope | flow_cytometer | liquid_handler
    plate_id = Column(String)
    well = Column(String, nullable=True)
    timepoint = Column(Integer, default=0)
    channels = Column(JSON)
    storage_path = Column(String)
    acquired_at = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column("metadata", JSON, default=dict)

    experiment = relationship("Experiment", back_populates="acquisitions")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"))
    plate_id = Column(String)
    well = Column(String, nullable=True)
    timepoint = Column(Integer, default=0)
    n_cells = Column(Integer, nullable=True)
    mitotic_index = Column(Float, nullable=True)
    live_cell_pct = Column(Float, nullable=True)
    mean_dapi_intensity = Column(Float, nullable=True)
    mean_fitc_intensity = Column(Float, nullable=True)
    cytometry_stats = Column(JSON, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="results")


class ExperimentEvent(Base):
    __tablename__ = "experiment_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"))
    event_type = Column(String)          # agent_step | tool_call | error | milestone
    message = Column(Text)
    data = Column(JSON, nullable=True)
    occurred_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="events")


class Database:
    def __init__(self, url: str = "sqlite:///lab_agent.db"):
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._cleanup_stale_experiments()

    def _cleanup_stale_experiments(self):
        """Mark any experiment stuck in 'running' for over 1 hour as 'aborted'.

        This handles experiments that were killed mid-run (e.g. Ctrl-C, crash).
        One hour is a conservative upper bound — real experiments complete in minutes.
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=1)
        with self.Session() as s:
            stale = (
                s.query(Experiment)
                .filter(Experiment.status == "running")
                .filter(Experiment.created_at < cutoff)
                .all()
            )
            for exp in stale:
                exp.status = "aborted"
            if stale:
                s.commit()

    def session(self) -> Session:
        return self.Session()

    # ------------------------------------------------------------------ #
    #  Convenience writes                                                 #
    # ------------------------------------------------------------------ #

    def create_experiment(self, exp_id: str, name: str,
                           command: str) -> Experiment:
        with self.session() as s:
            exp = Experiment(id=exp_id, name=name, command=command)
            s.add(exp)
            s.commit()
            s.refresh(exp)
            return exp

    def log_event(self, experiment_id: str, event_type: str,
                  message: str, data: dict = None):
        with self.session() as s:
            ev = ExperimentEvent(
                experiment_id=experiment_id,
                event_type=event_type,
                message=message,
                data=data or {},
            )
            s.add(ev)
            s.commit()

    def save_acquisition(self, experiment_id: str, acq_id: str,
                          instrument: str, plate_id: str,
                          channels: list, storage_path: str,
                          well: str = None, timepoint: int = 0):
        with self.session() as s:
            acq = Acquisition(
                id=acq_id,
                experiment_id=experiment_id,
                instrument=instrument,
                plate_id=plate_id,
                well=well,
                timepoint=timepoint,
                channels=channels,
                storage_path=storage_path,
            )
            s.add(acq)
            s.commit()

    def save_analysis_result(self, experiment_id: str, plate_id: str,
                              well: str, timepoint: int, well_result):
        with self.session() as s:
            r = AnalysisResult(
                experiment_id=experiment_id,
                plate_id=plate_id,
                well=well,
                timepoint=timepoint,
                n_cells=well_result.n_cells,
                mitotic_index=well_result.mitotic_index,
                live_cell_pct=well_result.live_cell_pct,
                mean_dapi_intensity=well_result.mean_dapi_intensity,
                mean_fitc_intensity=well_result.mean_fitc_intensity,
            )
            s.add(r)
            s.commit()

    def complete_experiment(self, experiment_id: str, status: str = "completed"):
        with self.session() as s:
            exp = s.get(Experiment, experiment_id)
            if exp:
                exp.status = status
                exp.completed_at = datetime.utcnow()
                s.commit()

    # ------------------------------------------------------------------ #
    #  Queries (used by the Data Agent tool)                             #
    # ------------------------------------------------------------------ #

    def query_results(self, plate_id: str = None,
                      min_mitotic_pct: float = None) -> list[dict]:
        with self.session() as s:
            q = s.query(AnalysisResult)
            if plate_id:
                q = q.filter(AnalysisResult.plate_id == plate_id)
            if min_mitotic_pct is not None:
                q = q.filter(
                    AnalysisResult.mitotic_index >= min_mitotic_pct / 100
                )
            return [
                {
                    "plate_id": r.plate_id,
                    "well": r.well,
                    "timepoint": r.timepoint,
                    "n_cells": r.n_cells,
                    "mitotic_index_pct": round((r.mitotic_index or 0) * 100, 1),
                    "live_cell_pct": r.live_cell_pct,
                }
                for r in q.all()
            ]

    def list_experiments(self) -> list[dict]:
        with self.session() as s:
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "status": e.status,
                    "created_at": str(e.created_at),
                    "command": e.command,
                }
                for e in s.query(Experiment).order_by(
                    Experiment.created_at.desc()
                ).limit(20).all()
            ]

    # ------------------------------------------------------------------ #
    #  M1: Async job tracking                                            #
    # ------------------------------------------------------------------ #

    def create_job(self, job_id: str, experiment_id: str, instrument: str,
                   tool_name: str, params: dict, eta_seconds: int = None) -> InstrumentJob:
        with self.session() as s:
            job = InstrumentJob(
                id=job_id,
                experiment_id=experiment_id,
                instrument=instrument,
                tool_name=tool_name,
                params=params,
                status="running",
                eta_seconds=eta_seconds,
            )
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    def complete_job(self, job_id: str, result: dict):
        with self.session() as s:
            job = s.get(InstrumentJob, job_id)
            if job:
                job.status = "done"
                job.result = result
                job.completed_at = datetime.utcnow()
                s.commit()

    def fail_job(self, job_id: str, error_message: str):
        with self.session() as s:
            job = s.get(InstrumentJob, job_id)
            if job:
                job.status = "error"
                job.error_message = error_message
                job.completed_at = datetime.utcnow()
                s.commit()

    def get_job(self, job_id: str) -> dict | None:
        with self.session() as s:
            job = s.get(InstrumentJob, job_id)
            if not job:
                return None
            return {
                "id": job.id,
                "experiment_id": job.experiment_id,
                "instrument": job.instrument,
                "tool_name": job.tool_name,
                "params": job.params,
                "status": job.status,
                "result": job.result,
                "error_message": job.error_message,
                "created_at": str(job.created_at),
                "completed_at": str(job.completed_at) if job.completed_at else None,
                "eta_seconds": job.eta_seconds,
            }

    # ------------------------------------------------------------------ #
    #  M4: Protocol CRUD                                                 #
    # ------------------------------------------------------------------ #

    def save_protocol(self, experiment_id: str, goal: str, hypothesis: str,
                      steps: list, success_criteria: str,
                      estimated_duration_h: float,
                      validation_status: str, validation_errors: list) -> str:
        import uuid as _uuid
        protocol_id = f"PROTO-{str(_uuid.uuid4())[:8].upper()}"
        with self.session() as s:
            proto = Protocol(
                id=protocol_id,
                experiment_id=experiment_id,
                goal=goal,
                hypothesis=hypothesis,
                steps=steps,
                success_criteria=success_criteria,
                estimated_duration_h=estimated_duration_h,
                validation_status=validation_status,
                validation_errors=validation_errors,
            )
            s.add(proto)
            s.commit()
        return protocol_id

    def get_protocol(self, protocol_id: str) -> dict | None:
        with self.session() as s:
            p = s.get(Protocol, protocol_id)
            if not p:
                return None
            return {
                "id": p.id,
                "experiment_id": p.experiment_id,
                "goal": p.goal,
                "hypothesis": p.hypothesis,
                "steps": p.steps,
                "success_criteria": p.success_criteria,
                "estimated_duration_h": p.estimated_duration_h,
                "validation_status": p.validation_status,
                "validation_errors": p.validation_errors,
                "approval_status": p.approval_status,
                "approval_note": p.approval_note,
                "execution_status": p.execution_status,
            }

    def approve_protocol(self, protocol_id: str, note: str = ""):
        with self.session() as s:
            p = s.get(Protocol, protocol_id)
            if p:
                p.approval_status = "approved"
                p.approval_note = note
                p.approved_at = datetime.utcnow()
                s.commit()

    def reject_protocol(self, protocol_id: str, note: str = ""):
        with self.session() as s:
            p = s.get(Protocol, protocol_id)
            if p:
                p.approval_status = "rejected"
                p.approval_note = note
                s.commit()

    def update_protocol_execution(self, protocol_id: str, status: str):
        with self.session() as s:
            p = s.get(Protocol, protocol_id)
            if p:
                p.execution_status = status
                if status == "completed":
                    p.completed_at = datetime.utcnow()
                s.commit()

    def list_protocols(self, experiment_id: str = None) -> list[dict]:
        with self.session() as s:
            q = s.query(Protocol)
            if experiment_id:
                q = q.filter(Protocol.experiment_id == experiment_id)
            return [
                {
                    "id": p.id,
                    "goal": (p.goal or "")[:80],
                    "validation_status": p.validation_status,
                    "approval_status": p.approval_status,
                    "execution_status": p.execution_status,
                    "steps": len(p.steps or []),
                    "estimated_duration_h": p.estimated_duration_h,
                    "created_at": str(p.created_at),
                }
                for p in q.order_by(Protocol.created_at.desc()).limit(20).all()
            ]
