"""Lab Agent Orchestrator — M1–M4 closed-loop architecture.

Supports three LLM backends — auto-detected from environment:
  1. Groq (cloud, free tier):  GROQ_API_KEY in environment  → llama-3.3-70b or qwen3-32b
  2. Anthropic (cloud):        ANTHROPIC_API_KEY in environment → claude-sonnet
  3. Ollama (local, no key):   llm.backend: ollama in config.yaml → qwen2.5:14b or any local model

Priority: Groq > Anthropic > Ollama (first available wins, or override in config.yaml)

Architecture (M1–M4):
  M1 — Async job tracking: acquire_images() returns job_id immediately;
        wait_for_job(job_id) polls until done.
  M2 — Protocol validator: every tool call passes through ProtocolValidator
        before dispatch; impossible requests are rejected with a reason.
  M3 — Local LIMS: analysis results written to lab_lims.db with full
        provenance; LLM queries via query_lims() tool.
  M4 — Structured protocol planning: plan_experiment() → LLM generates
        JSON protocol → validator approves → human approves → execute_protocol()
        runs deterministically.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from src.instruments.microscope import MicroscopeDriver, AcquisitionParams
from src.instruments.cytometer import CytometerDriver, CytometryParams
from src.instruments.liquid_handler import LiquidHandlerDriver, LiquidHandlerParams, TransferStep
from src.analysis import AnalysisPipeline
from src.data import Database
from src.validation.protocol_validator import ProtocolValidator
from src.lims.lims_client import LocalLIMS
from src.protocol.executor import ProtocolExecutor


# ------------------------------------------------------------------ #
#  LLM backend abstraction                                           #
# ------------------------------------------------------------------ #

def _build_llm_backend(llm_cfg: dict):
    """
    Returns (backend_name, client, run_fn) where run_fn(messages, system, tools) -> (reply_text, tool_calls).
    Auto-detects from environment; config.yaml can override.
    """
    forced = llm_cfg.get("backend")
    if forced == "auto":
        forced = None  # treat "auto" same as unset — let env-var detection decide

    # ---- Groq (OpenAI-compatible, free tier) ----
    groq_key = os.environ.get("GROQ_API_KEY")
    if (forced == "groq" or (not forced and groq_key)):
        from openai import OpenAI
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = llm_cfg.get("groq_model", "llama-3.3-70b-versatile")
        fallback = llm_cfg.get("groq_model_fallback", "llama-3.1-8b-instant")
        return "groq", client, _make_openai_runner(client, model, fallback_model=fallback)

    # ---- Anthropic ----
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if (forced == "anthropic" or (not forced and anthropic_key)):
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key)
        model = llm_cfg.get("model", "claude-sonnet-4-5")
        return "anthropic", client, _make_anthropic_runner(client, model)

    # ---- Ollama (local, no key needed) ----
    if forced == "ollama" or not forced:
        from openai import OpenAI
        client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
        model = llm_cfg.get("ollama_model", "qwen2.5:14b")
        return "ollama", client, _make_openai_runner(client, model)

    raise RuntimeError(
        "No LLM backend available. Set GROQ_API_KEY or ANTHROPIC_API_KEY, "
        "or run Ollama locally."
    )


def _make_openai_runner(client, model: str, fallback_model: str = None):
    """Returns a runner for OpenAI-compatible APIs (Groq, Ollama)."""
    import time as _time

    def run(messages: list, system: str, tools: list):
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        full_messages = [{"role": "system", "content": system}] + messages

        # Try primary model, then fallback on rate-limit, then wait and retry
        models_to_try = [model]
        if fallback_model and fallback_model != model:
            models_to_try.append(fallback_model)

        response = None
        last_error = None
        for m_name in models_to_try:
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=m_name,
                        messages=full_messages,
                        tools=oai_tools,
                        tool_choice="auto",
                        temperature=0.2,
                        max_tokens=4096,
                    )
                    if m_name != model:
                        print(f"\n[Info] Using fallback model {m_name} (primary model rate-limited)")
                    break
                except Exception as e:
                    last_error = e
                    err = str(e)
                    if "rate_limit" in err or "429" in err or "413" in err or "Request too large" in err:
                        import re as _re
                        m_wait = _re.search(r'try again in (\d+)m(\d+)', err)
                        wait = int(m_wait.group(1)) * 60 + int(m_wait.group(2)) if m_wait else 60
                        wait = min(wait + 5, 120)
                        if attempt == 0:
                            reason = "rate limit" if ("rate_limit" in err or "429" in err) else "request too large"
                            print(f"\n[{reason} on {m_name}] Waiting {wait}s ...")
                            _time.sleep(wait)
                        # else: move to next model
                    else:
                        raise
            if response is not None:
                break

        if response is None:
            raise last_error

        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
        stop = choice.finish_reason
        return text, tool_calls, stop, msg
    return run


def _make_anthropic_runner(client, model: str):
    """Returns a runner for Anthropic Claude API."""
    def run(messages: list, system: str, tools: list):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )
        text = " ".join(b.text for b in response.content if hasattr(b, "text"))
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        stop = response.stop_reason  # "end_turn" or "tool_use"
        return text, tool_calls, stop, response.content
    return run

from src.instruments.microscope import MicroscopeDriver, AcquisitionParams
from src.instruments.cytometer import CytometerDriver, CytometryParams
from src.instruments.liquid_handler import LiquidHandlerDriver, LiquidHandlerParams, TransferStep
from src.analysis import AnalysisPipeline
from src.data import Database


# ------------------------------------------------------------------ #
#  Tool definitions for Claude                                        #
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "name": "setup_plate",
        "description": (
            "Prepare a plate for an experiment using the liquid handler. "
            "Use this to add media, reagents, stains, or compounds to wells "
            "before imaging or sorting. Always call this before imaging if "
            "cells need staining (e.g. DAPI for nuclei, phospho-H3 for mitosis)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {"type": "string", "description": "Plate identifier, e.g. P001"},
                "wells": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Wells to prepare, e.g. ['A1','A2','B1']. Use ['all'] for full plate."
                },
                "reagent": {"type": "string", "description": "Reagent to add, e.g. 'DAPI', 'phospho-H3-antibody', 'media'"},
                "volume_ul": {"type": "number", "description": "Volume in microliters", "default": 100},
            },
            "required": ["plate_id", "wells", "reagent"],
        },
    },
    {
        "name": "acquire_images",
        "description": (
            "Image wells on the confocal/fluorescence microscope. "
            "Supports single timepoint or time-lapse series. "
            "Returns paths to saved images and triggers analysis automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {"type": "string"},
                "wells": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Wells to image, e.g. ['A1','A2']. Use ['all'] to image every well."
                },
                "channels": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Fluorescence channels, e.g. ['DAPI','FITC','TRITC']"
                },
                "objective": {"type": "string", "default": "20x"},
                "timelapse_interval_min": {
                    "type": "number",
                    "description": "Interval between timepoints in minutes. Omit for single acquisition."
                },
                "timelapse_duration_h": {
                    "type": "number",
                    "description": "Total duration in hours for time-lapse. Omit for single acquisition."
                },
            },
            "required": ["plate_id", "wells", "channels"],
        },
    },
    {
        "name": "analyze_images",
        "description": (
            "Run cell image analysis on images that have already been acquired with acquire_images. "
            "IMPORTANT: You must call acquire_images for this plate before calling this tool. "
            "If you call analyze_images without acquiring first, it will return an error. "
            "Counts cells, measures mitotic index (fraction of cells in mitosis), "
            "nuclear morphology, and fluorescence intensities. "
            "Results are flagged as simulation_mode=true — numbers are indicative, not publication-quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {"type": "string"},
                "timepoint": {
                    "type": "integer",
                    "description": "Which timepoint to analyze (0 = first). Use -1 for all timepoints.",
                    "default": 0
                },
            },
            "required": ["plate_id"],
        },
    },
    {
        "name": "run_flow_cytometry",
        "description": (
            "Run flow cytometry on a sample well. Measures cell populations "
            "by fluorescence and scatter. Optionally sorts a subpopulation "
            "into a destination well. Use after imaging to validate imaging "
            "results or to physically separate a cell population."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {"type": "string"},
                "well": {"type": "string", "description": "Source well, e.g. 'A1'"},
                "channels": {
                    "type": "array", "items": {"type": "string"},
                    "default": ["FITC", "PE", "APC"]
                },
                "n_events": {"type": "integer", "default": 10000},
                "sort_gate": {
                    "type": "object",
                    "description": "Optional sort gate. e.g. {'channel':'FITC','min':0.7,'max':1.0}",
                    "properties": {
                        "channel": {"type": "string"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                    },
                },
                "sort_destination_plate": {"type": "string"},
                "sort_destination_well": {"type": "string"},
            },
            "required": ["plate_id", "well"],
        },
    },
    {
        "name": "transfer_cells",
        "description": (
            "Transfer cells or liquid between plates using the liquid handler. "
            "Use after sorting to move selected cells to a new plate, "
            "or to consolidate samples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_plate": {"type": "string"},
                "source_wells": {"type": "array", "items": {"type": "string"}},
                "dest_plate": {"type": "string"},
                "dest_wells": {"type": "array", "items": {"type": "string"}},
                "volume_ul": {"type": "number", "default": 100},
            },
            "required": ["source_plate", "source_wells", "dest_plate", "dest_wells"],
        },
    },
    {
        "name": "query_results",
        "description": (
            "Query the database for experiment results. Use to check what "
            "data has been collected, find wells above a threshold, or "
            "get a summary of a plate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {"type": "string", "description": "Filter by plate ID"},
                "min_mitotic_pct": {
                    "type": "number",
                    "description": "Return only wells with mitotic index above this %"
                },
            },
        },
    },
    # ── M1: Async job tracking ─────────────────────────────────────────────
    {
        "name": "wait_for_job",
        "description": (
            "Wait for an async instrument job to complete. "
            "Call this after acquire_images, run_flow_cytometry, or setup_plate "
            "returns a job_id. Blocks (polls every 2s) until the job finishes or errors. "
            "Returns the job result when done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID returned by the instrument tool"},
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max seconds to wait before giving up. Default 300.",
                    "default": 300,
                },
            },
            "required": ["job_id"],
        },
    },
    # ── M3: LIMS query ─────────────────────────────────────────────────────
    {
        "name": "query_lims",
        "description": (
            "Query the LIMS for structured analysis results with full provenance. "
            "Use this to answer questions like 'what was the mitotic index in B1 over time?' "
            "or 'which well had the highest cell count?'. "
            "More powerful than query_results — supports metric-level filtering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string", "description": "Filter by experiment ID"},
                "plate_id": {"type": "string", "description": "Filter by plate ID"},
                "well": {"type": "string", "description": "Filter by well, e.g. 'A1'"},
                "timepoint": {"type": "integer", "description": "Filter by timepoint index"},
                "metric": {
                    "type": "string",
                    "description": "Metric name: 'cells', 'mitotic_pct', 'live_pct', 'mean_dapi', 'mean_fitc'",
                },
                "min_value": {"type": "number", "description": "Return only rows with value >= this"},
                "summary": {
                    "type": "boolean",
                    "description": "If true, return summary stats (min/max/mean/top well) instead of raw rows",
                    "default": False,
                },
            },
        },
    },
    # ── M4: Protocol planning and execution ────────────────────────────────
    {
        "name": "plan_experiment",
        "description": (
            "Generate a structured, validated experiment protocol from a scientific goal. "
            "The protocol will be validated against physical instrument limits. "
            "A human must approve it before execution via execute_protocol(). "
            "Use this INSTEAD of calling instruments directly when the goal is complex "
            "or involves multiple steps that should be reviewed before running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Scientific goal in plain English, e.g. 'Find IC50 of compound X in HeLa cells using 10-point dose-response'",
                },
                "available_reagents": {
                    "type": "array", "items": {"type": "string"},
                    "description": "List of reagents available, e.g. ['DAPI', 'FITC', 'Nocodazole 100uM']",
                },
                "plate_id": {"type": "string", "description": "Plate to use, e.g. 'P001'"},
                "wells": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Wells to include. Omit to use all wells.",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "execute_protocol",
        "description": (
            "Execute a previously planned and approved protocol. "
            "The protocol must have been created with plan_experiment() and approved. "
            "Runs all steps deterministically in order. "
            "IMPORTANT: Will refuse to run if the protocol has not been approved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protocol_id": {
                    "type": "string",
                    "description": "Protocol ID returned by plan_experiment()",
                },
            },
            "required": ["protocol_id"],
        },
    },
    {
        "name": "list_protocols",
        "description": (
            "List all planned protocols for this experiment. "
            "Shows validation status, approval status, and execution status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
            },
        },
    },
]


# ------------------------------------------------------------------ #
#  Orchestrator                                                       #
# ------------------------------------------------------------------ #

class LabOrchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text())

        storage_path = Path(cfg["data"]["storage_path"])
        storage_path.mkdir(parents=True, exist_ok=True)

        self.db = Database(cfg["data"]["database_url"])
        self.microscope = MicroscopeDriver(
            cfg["instruments"]["microscope"], storage_path
        )
        self.cytometer = CytometerDriver(
            cfg["instruments"]["flow_cytometer"], storage_path
        )
        self.liquid_handler = LiquidHandlerDriver(
            cfg["instruments"]["liquid_handler"], storage_path
        )
        self.analysis = AnalysisPipeline(
            {"mode": "watershed"},
            storage_path,
        )

        llm_cfg = cfg.get("llm", {})
        self.backend_name, self._llm_client, self._llm_run = _build_llm_backend(llm_cfg)
        self._llm_cfg = llm_cfg

        # M2: Protocol validator
        self.validator = ProtocolValidator()

        # M3: Local LIMS (separate db file)
        lims_path = Path(cfg["data"].get("lims_path", "lab_lims.db"))
        self.lims = LocalLIMS(str(lims_path))

        # Track images acquired in this session for analysis
        self._acquired_images: dict[str, dict] = {}

    def run(self, command: str, experiment_id: str = None,
            on_event=None, dataset_images: dict = None) -> str:
        """
        Run a natural language lab command end-to-end.

        command:         English instruction
        experiment_id:   optional, generated if not provided
        on_event:        callback(event_type, message) for live progress
        dataset_images:  pre-loaded real images to use instead of acquiring.
                         Format: {"plate_id": {"T000": {well: {ch: path}}}}
                         or      {"plate_id": {well: {ch: path}}}  (single timepoint)
        """
        exp_id = experiment_id or f"EXP-{str(uuid.uuid4())[:8].upper()}"
        self.db.create_experiment(exp_id, command[:60], command)

        # Pre-load real dataset images if provided
        if dataset_images:
            for plate_id, images in dataset_images.items():
                self._acquired_images[plate_id] = images
            self._log(on_event, "start",
                       f"Experiment {exp_id} started with pre-loaded real images: "
                       f"{list(dataset_images.keys())}")
        else:
            self._log(on_event, "start", f"Experiment {exp_id} started")

        messages = [{"role": "user", "content": command}]
        system = self._system_prompt(exp_id)

        max_turns = 30  # prevent infinite loops with weaker local models
        for turn in range(max_turns):
            text, tool_calls, stop_reason, raw_msg = self._llm_run(
                messages, system, TOOLS
            )

            if text:
                self._log(on_event, "agent_thought", text)

            # Guard: if the model writes text on turn 0 instead of calling a tool,
            # inject a hard reminder to force a tool call next turn.
            if turn == 0 and text and not tool_calls:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",
                                 "content": "You must call a tool now. Do not write text first."})
                continue

            # No tool calls — agent is done
            if not tool_calls:
                final = text or "Experiment complete."
                self.db.complete_experiment(exp_id)
                self._write_report(exp_id, command, final)
                self._log(on_event, "complete", final)
                return final

            # Execute tool calls
            tool_results_content = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc["input"]
                self._log(on_event, "tool_call",
                           f"{tool_name}({json.dumps(tool_input, indent=2)})")

                result = self._dispatch_tool(tool_name, tool_input, exp_id, on_event)
                self._log(on_event, "tool_result",
                           f"{tool_name} → {json.dumps(result, indent=2)}")
                self.db.log_event(exp_id, "tool_call", tool_name,
                                  {"input": tool_input, "output": result})

                tool_results_content.append({
                    "tool_use_id": tc["id"],
                    "result": json.dumps(result),
                })

            # Build next conversation turn — format differs per backend
            if self.backend_name == "anthropic":
                messages.append({"role": "assistant", "content": raw_msg})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result",
                         "tool_use_id": r["tool_use_id"],
                         "content": r["result"]}
                        for r in tool_results_content
                    ],
                })
            else:
                # OpenAI-compatible (Groq / Ollama)
                assistant_dict = {
                    "role": "assistant",
                    "content": raw_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (raw_msg.tool_calls or [])
                    ] or None,
                }
                assistant_dict = {k: v for k, v in assistant_dict.items() if v is not None}
                messages.append(assistant_dict)
                for r in tool_results_content:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": r["tool_use_id"],
                        "content": r["result"],
                    })

        # Reached turn limit
        final = "Experiment reached max turns without completing."
        self.db.complete_experiment(exp_id)
        self._write_report(exp_id, command, final)
        return final

    # ------------------------------------------------------------------ #
    #  Tool dispatch                                                      #
    # ------------------------------------------------------------------ #

    def _dispatch_tool(self, name: str, inputs: dict,
                        exp_id: str, on_event) -> dict:
        # ── M2: Validate before dispatching ──────────────────────────────
        validation = self.validator.validate_tool_call(name, inputs)
        if not validation.valid:
            return {
                "validation_error": True,
                "errors": validation.errors,
                "message": (
                    "Tool call rejected by validator. Fix these issues and try again: "
                    + "; ".join(validation.errors)
                ),
            }

        try:
            if name == "setup_plate":
                return self._tool_setup_plate(inputs, exp_id)
            if name == "acquire_images":
                return self._tool_acquire_images(inputs, exp_id, on_event)
            if name == "analyze_images":
                return self._tool_analyze_images(inputs, exp_id)
            if name == "run_flow_cytometry":
                return self._tool_run_cytometry(inputs, exp_id)
            if name == "transfer_cells":
                return self._tool_transfer_cells(inputs, exp_id)
            if name == "query_results":
                return self._tool_query_results(inputs)
            # M1
            if name == "wait_for_job":
                return self._tool_wait_for_job(inputs)
            # M3
            if name == "query_lims":
                return self._tool_query_lims(inputs, exp_id)
            # M4
            if name == "plan_experiment":
                return self._tool_plan_experiment(inputs, exp_id)
            if name == "execute_protocol":
                return self._tool_execute_protocol(inputs, exp_id, on_event)
            if name == "list_protocols":
                return self._tool_list_protocols(inputs, exp_id)
            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}

    def _tool_setup_plate(self, inputs: dict, exp_id: str) -> dict:
        wells = inputs["wells"]
        if isinstance(wells, str):
            wells = [wells]  # coerce "all" string → ["all"]
        if wells == ["all"]:
            wells = [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]
        result = self.liquid_handler.execute(LiquidHandlerParams(
            operation="add_reagent",
            dest_plate=inputs["plate_id"],
            wells=wells,
            reagent=inputs.get("reagent", "reagent"),
            volume_ul=inputs.get("volume_ul", 100),
        ))
        return result.data if result.success else {"error": result.error}

    def _tool_acquire_images(self, inputs: dict, exp_id: str, on_event) -> dict:
        plate_id = inputs["plate_id"]
        wells = inputs["wells"]
        if isinstance(wells, str):
            wells = [wells]
        inputs = {**inputs, "wells": wells}

        # If pre-loaded real timelapse images already exist for this plate,
        # confirm registration without running a synthetic acquisition.
        existing = self._acquired_images.get(plate_id, {})
        first_key = next(iter(existing), None)
        is_preloaded_timelapse = (
            first_key and isinstance(first_key, str)
            and first_key.startswith("T") and len(first_key) == 4
            and len(existing) > 1
        )
        if is_preloaded_timelapse:
            tp_keys = sorted(existing.keys())
            self.db.save_acquisition(
                exp_id, "preloaded", "microscope",
                plate_id, inputs.get("channels", ["DAPI", "FITC"]),
                str(self.microscope.storage_path / "images" / plate_id),
            )
            return {
                "plate_id": plate_id,
                "status": "pre-loaded real images registered",
                "n_timepoints": len(tp_keys),
                "timepoints": tp_keys,
                "channels": inputs.get("channels", ["DAPI", "FITC"]),
                "note": "Real dataset images already loaded — no simulated acquisition needed. "
                        "Call analyze_images with timepoint=-1 to process all timepoints.",
            }

        # ── M1: Async acquisition — return job_id immediately ──────────────
        job_id = f"JOB-{str(uuid.uuid4())[:8].upper()}"
        params_snap = {**inputs}
        is_timelapse = bool(inputs.get("timelapse_interval_min"))
        eta = int((inputs.get("timelapse_duration_h", 0) * 60) + 5) if is_timelapse else 5

        self.db.create_job(
            job_id=job_id,
            experiment_id=exp_id,
            instrument="microscope",
            tool_name="acquire_images",
            params=params_snap,
            eta_seconds=eta,
        )

        def _bg_acquire():
            try:
                def progress(t, total, imgs):
                    self._log(on_event, "acquisition_progress",
                               f"Timepoint {t+1}/{total} acquired for {plate_id}")

                params = AcquisitionParams(
                    plate_id=plate_id,
                    wells=inputs["wells"],
                    channels=inputs.get("channels", ["DAPI", "FITC"]),
                    objective=inputs.get("objective", "20x"),
                    timelapse_interval_min=inputs.get("timelapse_interval_min"),
                    timelapse_duration_h=inputs.get("timelapse_duration_h"),
                )

                if params.timelapse_interval_min:
                    result = self.microscope.acquire_timelapse(params, progress_callback=progress)
                else:
                    result = self.microscope.acquire_single(params)

                if result.success:
                    new_paths = result.data.get("image_paths", {})
                    first_ex = next(iter(self._acquired_images.get(plate_id, {})), None)
                    is_ex_tl = (
                        first_ex and isinstance(first_ex, str)
                        and first_ex.startswith("T") and len(first_ex) == 4
                    )
                    if not is_ex_tl:
                        self._acquired_images[plate_id] = new_paths
                    self.db.save_acquisition(
                        exp_id, result.job_id, "microscope",
                        plate_id, params.channels,
                        str(self.microscope.storage_path / "images" / plate_id),
                    )
                    self.db.complete_job(job_id, result.data)
                    self._log(on_event, "acquisition_progress",
                               f"Job {job_id} done — plate {plate_id} acquired.")
                else:
                    self.db.fail_job(job_id, result.error or "acquisition failed")
                    self._log(on_event, "error", f"Job {job_id} failed: {result.error}")
            except Exception as exc:
                self.db.fail_job(job_id, str(exc))
                self._log(on_event, "error", f"Job {job_id} exception: {exc}")

        thread = threading.Thread(target=_bg_acquire, daemon=True)
        thread.start()

        return {
            "job_id": job_id,
            "status": "running",
            "plate_id": plate_id,
            "eta_seconds": eta,
            "message": (
                f"Acquisition started (job {job_id}). "
                f"Call wait_for_job('{job_id}') to block until images are ready, "
                f"then call analyze_images."
            ),
        }

    def _tool_analyze_images(self, inputs: dict, exp_id: str) -> dict:
        plate_id = inputs["plate_id"]
        timepoint_req = inputs.get("timepoint", 0)

        images = self._acquired_images.get(plate_id, {})
        if not images:
            return {"error": f"No images found for plate {plate_id}. Run acquire_images first."}

        all_summaries = []
        first_key = list(images.keys())[0]

        # Detect timelapse format: keys like "T000", "T001", ...
        is_timelapse = isinstance(first_key, str) and first_key.startswith("T") and len(first_key) == 4

        if is_timelapse:
            if timepoint_req == -1:
                timepoints_to_analyze = sorted(images.items())
            else:
                tp_key = f"T{timepoint_req:03d}"
                tp_data = images.get(tp_key)
                if not tp_data:
                    return {"error": f"Timepoint {tp_key} not found. Available: {sorted(images.keys())}"}
                timepoints_to_analyze = [(tp_key, tp_data)]

            for tp_key, tp_images in timepoints_to_analyze:
                t_idx = int(tp_key[1:])
                plate_result = self.analysis.analyze_plate(plate_id, tp_images, t_idx)
                for wr in plate_result.well_results:
                    self.db.save_analysis_result(exp_id, plate_id, wr.well, t_idx, wr)
                # M3: write per-well results to LIMS
                self.lims.write_plate_results(
                    exp_id, plate_id, t_idx,
                    {wr.well: {"cells": wr.n_cells,
                               "mitotic_pct": round(wr.mitotic_index * 100, 2),
                               "live_pct": wr.live_cell_pct,
                               "mean_dapi": wr.mean_dapi_intensity,
                               "mean_fitc": wr.mean_fitc_intensity}
                     for wr in plate_result.well_results},
                )
                n_cells_list = [wr.n_cells for wr in plate_result.well_results]
                mit_list = [wr.mitotic_index * 100 for wr in plate_result.well_results]
                well_names = [wr.well for wr in plate_result.well_results]
                all_summaries.append({
                    "timepoint": tp_key,
                    "wells_analyzed": well_names,
                    "mean_cells": round(sum(n_cells_list) / max(len(n_cells_list), 1), 1),
                    "mean_mitotic_pct": round(sum(mit_list) / max(len(mit_list), 1), 1),
                    "per_well": {w: {"cells": c, "mitotic_pct": round(m, 1)}
                                 for w, c, m in zip(well_names, n_cells_list, mit_list)},
                })
        else:
            # Single-timepoint or flat dict: {well: {ch: path}}
            plate_result = self.analysis.analyze_plate(plate_id, images, timepoint_req)
            for wr in plate_result.well_results:
                self.db.save_analysis_result(exp_id, plate_id, wr.well, timepoint_req, wr)
            # M3: write to LIMS
            self.lims.write_plate_results(
                exp_id, plate_id, timepoint_req,
                {wr.well: {"cells": wr.n_cells,
                           "mitotic_pct": round(wr.mitotic_index * 100, 2),
                           "live_pct": wr.live_cell_pct,
                           "mean_dapi": wr.mean_dapi_intensity,
                           "mean_fitc": wr.mean_fitc_intensity}
                 for wr in plate_result.well_results},
            )
            n_cells_list = [wr.n_cells for wr in plate_result.well_results]
            mit_list = [wr.mitotic_index * 100 for wr in plate_result.well_results]
            well_names = [wr.well for wr in plate_result.well_results]
            all_summaries.append({
                "timepoint": f"T{timepoint_req:03d}",
                "wells_analyzed": well_names,
                "mean_cells": round(sum(n_cells_list) / max(len(n_cells_list), 1), 1),
                "mean_mitotic_pct": round(sum(mit_list) / max(len(mit_list), 1), 1),
                "per_well": {w: {"cells": c, "mitotic_pct": round(m, 1)}
                             for w, c, m in zip(well_names, n_cells_list, mit_list)},
            })

        return {
            "plate_id": plate_id,
            "analysis_complete": True,
            "analysis_method": "watershed",
            "note": (
                "Mitotic index method: FITC channel overlap with segmented nuclei (primary), "
                "plus compact+bright DAPI morphology (secondary, for tubulin-stained datasets like BBBC020). "
                "For accurate mitotic index, use phospho-H3 FITC staining. "
                "Current numbers are valid for relative comparisons across wells/timepoints."
            ),
            "n_timepoints_analyzed": len(all_summaries),
            "results": all_summaries,
        }

    def _tool_run_cytometry(self, inputs: dict, exp_id: str) -> dict:
        result = self.cytometer.acquire(CytometryParams(
            plate_id=inputs["plate_id"],
            sample_well=inputs["well"],
            channels=inputs.get("channels", ["FITC", "PE", "APC"]),
            n_events=inputs.get("n_events", 10000),
            sort_gate=inputs.get("sort_gate"),
            sort_destination_plate=inputs.get("sort_destination_plate"),
            sort_destination_well=inputs.get("sort_destination_well"),
        ))
        data = result.data if result.success else {"error": result.error}
        if result.success:
            data["simulation_mode"] = True
            data["note"] = (
                "Population statistics are simulated (random Gaussian populations). "
                "Not derived from real cell images. Use only to verify workflow, not for biology."
            )
        return data

    def _tool_transfer_cells(self, inputs: dict, exp_id: str) -> dict:
        source_wells = inputs["source_wells"]
        dest_wells = inputs["dest_wells"]
        transfers = [
            TransferStep(
                source_plate=inputs["source_plate"],
                source_well=sw,
                dest_plate=inputs["dest_plate"],
                dest_well=dw,
                volume_ul=inputs.get("volume_ul", 100),
            )
            for sw, dw in zip(source_wells, dest_wells)
        ]
        result = self.liquid_handler.execute(LiquidHandlerParams(
            operation="transfer_cells",
            source_plate=inputs["source_plate"],
            dest_plate=inputs["dest_plate"],
            transfers=transfers,
        ))
        return result.data if result.success else {"error": result.error}

    def _tool_query_results(self, inputs: dict) -> dict:
        rows = self.db.query_results(
            plate_id=inputs.get("plate_id"),
            min_mitotic_pct=inputs.get("min_mitotic_pct"),
        )
        return {
            "n_results": len(rows),
            "results": rows[:50],
        }

    # ── M1: Async job tracking ─────────────────────────────────────────────

    def _tool_wait_for_job(self, inputs: dict) -> dict:
        """Poll DB until job is done or timeout reached."""
        job_id = inputs["job_id"]
        timeout = inputs.get("timeout_seconds", 300)
        deadline = time.time() + timeout
        poll_interval = 2  # seconds

        while time.time() < deadline:
            job = self.db.get_job(job_id)
            if not job:
                return {"error": f"Job '{job_id}' not found in database."}

            if job["status"] == "done":
                # If this was an acquisition job, store images in session cache
                if job["tool_name"] == "acquire_images" and job.get("result"):
                    result_data = job["result"]
                    plate_id = job["params"].get("plate_id")
                    new_paths = result_data.get("image_paths", {})
                    if plate_id and new_paths:
                        first_ex = next(iter(self._acquired_images.get(plate_id, {})), None)
                        is_tl = (first_ex and isinstance(first_ex, str)
                                 and first_ex.startswith("T") and len(first_ex) == 4)
                        if not is_tl:
                            self._acquired_images[plate_id] = new_paths
                return {
                    "job_id": job_id,
                    "status": "done",
                    "instrument": job["instrument"],
                    "tool_name": job["tool_name"],
                    "result": job["result"],
                    "message": "Job complete. You can now call analyze_images.",
                }

            if job["status"] == "error":
                return {
                    "job_id": job_id,
                    "status": "error",
                    "error": job["error_message"],
                }

            elapsed = round(time.time() - (deadline - timeout))
            eta = job.get("eta_seconds", "unknown")
            time.sleep(poll_interval)

        return {
            "job_id": job_id,
            "status": "timeout",
            "message": f"Job did not complete within {timeout}s. Call wait_for_job again to keep waiting.",
        }

    # ── M3: LIMS query ─────────────────────────────────────────────────────

    def _tool_query_lims(self, inputs: dict, exp_id: str) -> dict:
        """Query the local LIMS for structured analysis results."""
        experiment_id = inputs.get("experiment_id", exp_id)
        want_summary = inputs.get("summary", False)

        if want_summary:
            metric = inputs.get("metric", "mitotic_pct")
            result = self.lims.summary(
                experiment_id=experiment_id,
                plate_id=inputs.get("plate_id"),
                metric=metric,
            )
            return result

        rows = self.lims.query(
            experiment_id=experiment_id,
            plate_id=inputs.get("plate_id"),
            well=inputs.get("well"),
            timepoint=inputs.get("timepoint"),
            metric=inputs.get("metric"),
            min_value=inputs.get("min_value"),
            limit=100,
        )
        return {
            "n_results": len(rows),
            "results": rows,
            "tip": "Use summary=true for min/max/mean/top-well stats across all results.",
        }

    # ── M4: Protocol planning and execution ────────────────────────────────

    def _tool_plan_experiment(self, inputs: dict, exp_id: str) -> dict:
        """Generate a structured experiment protocol using the LLM, then validate it."""
        goal = inputs["goal"]
        plate_id = inputs.get("plate_id", "P001")
        wells = inputs.get("wells", ["A1", "A2", "A3"])
        reagents = inputs.get("available_reagents", ["DAPI", "FITC"])

        plan_prompt = f"""You are an experiment planner. Generate a JSON experiment protocol.

Scientific goal: {goal}
Plate: {plate_id}
Wells to use: {wells}
Available reagents: {reagents}

Return ONLY valid JSON (no explanation, no markdown) with this exact structure:
{{
  "hypothesis": "one sentence — what you expect to observe",
  "success_criteria": "one sentence — what result would confirm the hypothesis",
  "steps": [
    {{
      "instrument": "liquid_handler|microscope|cytometer",
      "operation": "setup_plate|acquire_images|analyze_images|run_flow_cytometry",
      "params": {{ ... }},
      "expected_duration_min": 5
    }}
  ]
}}

Rules for steps:
- setup_plate params: plate_id, wells (array), reagent (string), volume_ul (number 1-1000)
- acquire_images params: plate_id, wells (array), channels (array of DAPI/FITC/TRITC)
- analyze_images params: plate_id, timepoint (integer, -1 for all)
- run_flow_cytometry params: plate_id, well (string), channels (array), n_events (integer)
- ALL well values must be strings like "A1", "B2" — never integers
- channels must be from: DAPI, FITC, TRITC, CY5
"""
        # H: Retry up to 2 times if LLM returns invalid JSON or validation fails
        protocol_data = None
        last_error = ""
        retry_prompt = plan_prompt
        for attempt in range(3):
            try:
                messages = [{"role": "user", "content": retry_prompt}]
                text, _, _, _ = self._llm_run(
                    messages, "You are a JSON generator. Output only valid JSON.", []
                )
                clean = text.strip()
                if clean.startswith("```"):
                    lines = clean.split("\n")
                    clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                protocol_data = json.loads(clean)
                break
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON (attempt {attempt+1}): {e}"
                retry_prompt = (
                    plan_prompt
                    + f"\n\nYour previous response was not valid JSON. Error: {e}\n"
                    "Output ONLY raw JSON — no markdown, no explanation."
                )
            except Exception as e:
                return {"status": "error", "message": f"Protocol generation failed: {e}"}

        if protocol_data is None:
            return {"status": "error", "message": f"LLM failed to produce valid JSON after 3 attempts. Last error: {last_error}"}

        # M2: Validate the generated protocol — retry once if validation fails
        validation = self.validator.validate_protocol(protocol_data)
        if not validation.valid and len(validation.errors) <= 3:
            fix_prompt = (
                plan_prompt
                + f"\n\nYour previous protocol had validation errors:\n"
                + "\n".join(f"- {e}" for e in validation.errors)
                + "\n\nFix these issues and return corrected JSON only."
            )
            try:
                messages = [{"role": "user", "content": fix_prompt}]
                text, _, _, _ = self._llm_run(
                    messages, "You are a JSON generator. Output only valid JSON.", []
                )
                clean = text.strip()
                if clean.startswith("```"):
                    lines = clean.split("\n")
                    clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                fixed_data = json.loads(clean)
                fixed_validation = self.validator.validate_protocol(fixed_data)
                if fixed_validation.valid:
                    protocol_data = fixed_data
                    validation = fixed_validation
            except Exception:
                pass  # keep original validation result

        estimated_h = round(sum(
            s.get("expected_duration_min", 0)
            for s in protocol_data.get("steps", [])
        ) / 60, 2)

        protocol_id = self.db.save_protocol(
            experiment_id=exp_id,
            goal=goal,
            hypothesis=protocol_data.get("hypothesis", ""),
            steps=protocol_data.get("steps", []),
            success_criteria=protocol_data.get("success_criteria", ""),
            estimated_duration_h=estimated_h,
            validation_status="valid" if validation.valid else "invalid",
            validation_errors=validation.errors,
        )

        if not validation.valid:
            return {
                "protocol_id": protocol_id,
                "status": "validation_failed",
                "validation_errors": validation.errors,
                "message": (
                    "Protocol generated but failed validation. "
                    "Fix these issues before executing: "
                    + "; ".join(validation.errors)
                ),
            }

        return {
            "protocol_id": protocol_id,
            "status": "pending_approval",
            "hypothesis": protocol_data.get("hypothesis"),
            "success_criteria": protocol_data.get("success_criteria"),
            "n_steps": len(protocol_data.get("steps", [])),
            "estimated_duration_h": estimated_h,
            "steps_summary": [
                f"{s.get('instrument')}.{s.get('operation')} (~{s.get('expected_duration_min')}min)"
                for s in protocol_data.get("steps", [])
            ],
            "message": (
                f"Protocol {protocol_id} generated and validated ({len(protocol_data.get('steps', []))} steps). "
                "It requires human approval before execution. "
                "Approve with: agent.approve_protocol(protocol_id) "
                "Then call execute_protocol(protocol_id) to run it."
            ),
        }

    def _tool_execute_protocol(self, inputs: dict, exp_id: str, on_event) -> dict:
        """Execute an approved protocol using the ProtocolExecutor."""
        protocol_id = inputs["protocol_id"]
        protocol = self.db.get_protocol(protocol_id)

        if not protocol:
            return {"error": f"Protocol '{protocol_id}' not found."}

        if protocol["validation_status"] != "valid":
            return {
                "error": "Protocol failed validation and cannot be executed.",
                "validation_errors": protocol["validation_errors"],
            }

        if protocol["approval_status"] != "approved":
            return {
                "status": "blocked",
                "message": (
                    f"Protocol '{protocol_id}' has not been approved. "
                    f"Current approval status: {protocol['approval_status']}. "
                    "A human must call agent.approve_protocol(protocol_id) before execution."
                ),
            }

        if protocol["execution_status"] == "completed":
            return {"status": "already_completed", "message": f"Protocol {protocol_id} was already executed."}

        self.db.update_protocol_execution(protocol_id, "running")
        executor = ProtocolExecutor(
            dispatch_fn=self._dispatch_tool,
            on_event=on_event,
        )
        result = executor.run(protocol, exp_id)
        final_status = "completed" if result["status"] == "completed" else "failed"
        self.db.update_protocol_execution(protocol_id, final_status)
        return {
            "protocol_id": protocol_id,
            "execution_status": final_status,
            **result,
        }

    def _tool_list_protocols(self, inputs: dict, exp_id: str) -> dict:
        protocols = self.db.list_protocols(
            experiment_id=inputs.get("experiment_id", exp_id)
        )
        return {"n_protocols": len(protocols), "protocols": protocols}

    # ── Public helpers for human approval (call from Python / CLI) ─────────

    def approve_protocol(self, protocol_id: str, note: str = "approved by operator") -> dict:
        """Human approval gate — call this from the CLI or a UI before execute_protocol."""
        self.db.approve_protocol(protocol_id, note)
        return {"protocol_id": protocol_id, "status": "approved", "note": note}

    def reject_protocol(self, protocol_id: str, reason: str = "") -> dict:
        self.db.reject_protocol(protocol_id, reason)
        return {"protocol_id": protocol_id, "status": "rejected", "reason": reason}

    # ------------------------------------------------------------------ #
    #  System prompt                                                      #
    # ------------------------------------------------------------------ #

    def _system_prompt(self, exp_id: str) -> str:
        return f"""You are a lab automation orchestrator. Experiment ID: {exp_id}
ALWAYS respond in English only.

CRITICAL: Your FIRST response must be a tool call — not text. No preamble, no plan, no explanation.

== TOOL WORKFLOW ==

For a SIMPLE experiment (direct instrument control):
  1. Call acquire_images → returns job_id (async).
  2. Call wait_for_job(job_id) → blocks until images ready.
  3. Call analyze_images with timepoint=-1 → processes all timepoints.
  4. Call query_lims(summary=true, metric="mitotic_pct") → get summary stats.
  5. Optionally call run_flow_cytometry on the well with the highest mitotic index.
  6. Write final report with ONLY actual numbers from tool results.

For a STRUCTURED experiment (protocol-based, requires human approval):
  1. Call plan_experiment(goal=...) → LLM generates + validates a protocol.
  2. Wait for human to approve (agent.approve_protocol(id) from CLI).
  3. Call execute_protocol(protocol_id) → runs all steps deterministically.
  4. Call query_lims to read results.
  5. Interpret and report.

== RULES ==
- NEVER state conclusions not supported by tool results.
- Quote actual numbers: "Well B1 at T003: 143 cells, 14.7% mitotic index."
- acquire_images is now ASYNC — always call wait_for_job before analyze_images.
- If wait_for_job times out, call it again — the job is still running.
- Flow cytometry results are SIMULATED — always state this clearly.
- If a tool call is rejected by the validator, read the error and fix the params.
- Your final response (no tool call) must be a concise biological interpretation
  with every key number explicitly cited from the tool results above.
"""

    def _write_report(self, exp_id: str, command: str, conclusion: str):
        """Write a plain Markdown report to experiment_data/reports/{exp_id}.md."""
        from datetime import datetime as dt
        reports_dir = Path("experiment_data") / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        rows = self.db.query_results()
        exp_rows = [r for r in rows if True]  # already filtered by session context

        # Collect tool calls from the database for this experiment
        import sqlite3
        db_path = str(self.db.engine.url).replace("sqlite:///", "")
        try:
            conn = sqlite3.connect(db_path)
            events = conn.execute(
                "SELECT message, data FROM experiment_events WHERE experiment_id=? ORDER BY id",
                (exp_id,)
            ).fetchall()
            conn.close()
        except Exception:
            events = []

        now = dt.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# Experiment Report: {exp_id}",
            f"",
            f"**Date:** {now}  ",
            f"**Status:** completed  ",
            f"**LLM backend:** {self.backend_name}  ",
            f"",
            f"## Command",
            f"",
            f"> {command}",
            f"",
            f"## Steps Executed",
            f"",
        ]

        for tool_name, data_json in events:
            import json as _json
            try:
                d = _json.loads(data_json) if data_json else {}
                inp = d.get("input", {})
                out = d.get("output", {})
                sim = out.get("simulation_mode", False)
                sim_flag = " *(simulated)*" if sim else ""
                lines.append(f"### {tool_name}{sim_flag}")
                if inp:
                    lines.append(f"**Input:** `{_json.dumps(inp)}`  ")
                # Extract key numbers
                if "results" in out:
                    for r in out["results"]:
                        s = r.get("summary", {})
                        if s:
                            lines.append(
                                f"- Timepoint {r['timepoint']}: "
                                f"avg mitotic index {s.get('avg_mitotic_index_pct', '?')}%, "
                                f"peak well {s.get('max_mitotic_well', '?')} "
                                f"({s.get('max_mitotic_index_pct', '?')}%)"
                            )
                elif "population_stats" in out:
                    ps = out["population_stats"]
                    lines.append(
                        f"- Events: {out.get('n_events_acquired', '?')} | "
                        f"Live: {ps.get('live_pct', '?')}% | "
                        f"Mitotic: {ps.get('mitotic_pct', '?')}% | "
                        f"Sorted cells: {out.get('sorted_cells', 0)}"
                    )
                lines.append("")
            except Exception:
                lines.append(f"### {tool_name}")
                lines.append("")

        lines += [
            f"## Conclusion",
            f"",
            conclusion,
            f"",
            f"---",
            f"*Analysis mode: thresholding (not Cellpose). "
            f"Mitotic index numbers are indicative. "
            f"Flow cytometry statistics are simulated.*",
        ]

        report_path = reports_dir / f"{exp_id}.md"
        report_path.write_text("\n".join(lines))

    def _log(self, callback, event_type: str, message: str):
        if callback:
            callback(event_type, message)
