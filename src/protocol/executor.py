"""M4: Protocol Executor — deterministic step-by-step execution engine.

The LLM generates a protocol → validator approves it → human approves it
→ this executor runs each step in order, calling instrument drivers directly.

No LLM is involved in execution. That's the point.
"""
from __future__ import annotations

import json
import time
import threading
from typing import Callable, Any


class ProtocolExecutor:
    """
    Runs a validated, approved protocol step by step.

    Each step has:
      instrument  : microscope | liquid_handler | cytometer | incubator
      operation   : acquire_images | setup_plate | run_flow_cytometry | transfer_cells
      params      : dict matching the tool's input schema
      expected_duration_min : float

    The executor calls the orchestrator's tool dispatch for each step,
    collecting results. On error, execution halts and the step error is recorded.
    """

    def __init__(self, dispatch_fn: Callable[[str, dict, str, Any], dict],
                 on_event: Callable[[str, str], None] = None):
        """
        dispatch_fn: orchestrator._dispatch_tool(name, inputs, exp_id, on_event)
        on_event:    optional progress callback(event_type, message)
        """
        self.dispatch = dispatch_fn
        self.on_event = on_event

    def run(self, protocol: dict, experiment_id: str) -> dict:
        """
        Execute all steps in a protocol synchronously.
        Returns execution summary with per-step results.

        In production: replace with Temporal/Prefect workflow for crash-resilience.
        """
        steps = protocol.get("steps", [])
        if not steps:
            return {"status": "error", "message": "Protocol has no steps."}

        results = []
        total_steps = len(steps)

        self._log("execution_start",
                  f"Executing protocol '{protocol.get('id', 'UNKNOWN')}' "
                  f"— {total_steps} steps, est. "
                  f"{protocol.get('estimated_duration_h', '?')}h")

        for idx, step in enumerate(steps):
            instrument = step.get("instrument", "unknown")
            operation = step.get("operation", "unknown")
            params = step.get("params", {})
            eta_min = step.get("expected_duration_min", 0)

            self._log("step_start",
                      f"Step {idx+1}/{total_steps}: {instrument}.{operation} "
                      f"(est. {eta_min} min)")

            start = time.time()
            try:
                result = self.dispatch(operation, params, experiment_id, self.on_event)
            except Exception as exc:
                result = {"error": str(exc)}

            # If the operation returned an async job_id, wait for it to finish
            # before moving to the next step. This ensures acquire → analyze ordering works.
            if isinstance(result, dict) and result.get("status") == "running" and result.get("job_id"):
                job_id = result["job_id"]
                self._log("step_start", f"Waiting for async job {job_id}...")
                wait_result = self.dispatch(
                    "wait_for_job",
                    {"job_id": job_id, "timeout_seconds": 300},
                    experiment_id,
                    self.on_event,
                )
                if wait_result.get("status") == "error":
                    result = {"error": f"Async job {job_id} failed: {wait_result.get('error')}"}
                elif wait_result.get("status") == "timeout":
                    result = {"error": f"Async job {job_id} timed out — still running in background"}
                else:
                    result = wait_result.get("result", result)

            elapsed = round(time.time() - start, 1)

            if "error" in result:
                self._log("step_error",
                          f"Step {idx+1} FAILED: {result['error']}")
                results.append({
                    "step_index": idx + 1,
                    "instrument": instrument,
                    "operation": operation,
                    "status": "error",
                    "error": result["error"],
                    "elapsed_s": elapsed,
                })
                return {
                    "status": "failed",
                    "failed_step": idx + 1,
                    "message": f"Step {idx+1} ({operation}) failed: {result['error']}",
                    "steps_completed": idx,
                    "step_results": results,
                }

            self._log("step_done",
                      f"Step {idx+1}/{total_steps}: {operation} completed in {elapsed}s")
            results.append({
                "step_index": idx + 1,
                "instrument": instrument,
                "operation": operation,
                "status": "done",
                "result": result,
                "elapsed_s": elapsed,
            })

        self._log("execution_done",
                  f"Protocol executed successfully — {total_steps} steps completed.")
        return {
            "status": "completed",
            "steps_completed": total_steps,
            "step_results": results,
        }

    def _log(self, event_type: str, message: str):
        if self.on_event:
            self.on_event(event_type, message)
