"""Lab Agent CLI.

Usage:
  python run.py "Run the mitosis assay on plate P001"
  python run.py "Image wells A1-A4 on plate P002 every 30 minutes for 2 hours, then sort mitotic cells"
  python run.py --list-experiments
  python run.py --demo              # run a built-in demo workflow

Requires: ANTHROPIC_API_KEY in environment or .env file
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich import box

load_dotenv()
console = Console()


def _read_backend_cfg(config: str) -> str | None:
    """Return the llm.backend value from config.yaml, or None on error."""
    try:
        import yaml as _yaml
        cfg = _yaml.safe_load(Path(config).read_text())
        return cfg.get("llm", {}).get("backend")
    except Exception:
        return None

EVENT_STYLES = {
    "start":                ("cyan", "[START]"),
    "agent_thought":        ("dim white", "[AGENT]"),
    "tool_call":            ("yellow", "[TOOL →]"),
    "tool_result":          ("green", "[RESULT]"),
    "acquisition_progress": ("blue", "[SCOPE]"),
    "complete":             ("bold green", "[DONE]"),
    "error":                ("bold red", "[ERROR]"),
    # M1/M4 new events
    "execution_start":      ("cyan", "[EXEC]"),
    "step_start":           ("blue", "[STEP]"),
    "step_done":            ("green", "[STEP ✓]"),
    "step_error":           ("bold red", "[STEP ✗]"),
    "execution_done":       ("bold green", "[EXEC DONE]"),
}


def on_event(event_type: str, message: str):
    style, prefix = EVENT_STYLES.get(event_type, ("white", f"[{event_type.upper()}]"))
    if event_type == "agent_thought":
        # Agent reasoning — show dimmed and truncated
        short = message[:200] + "..." if len(message) > 200 else message
        console.print(f"  {prefix} {short}", style=style)
    elif event_type == "tool_call":
        # Show just the first line of the JSON (tool name + key params)
        first_line = message.split("\n")[0]
        console.print(f"  {prefix} {first_line}", style=style)
    elif event_type == "tool_result":
        # Show compact summary
        short = message[:300] + "..." if len(message) > 300 else message
        console.print(f"  {prefix} {short}", style=style)
    else:
        console.print(f"  {prefix} {message}", style=style)


@click.group(invoke_without_command=True)
@click.argument("command", required=False)
@click.option("--config", default="config.yaml", help="Config file path")
@click.option("--demo", is_flag=True, help="Run built-in demo workflow (simulation)")
@click.option("--demo-real", is_flag=True, help="Run demo with real BBBC020 dataset (requires download)")
@click.option("--list-experiments", is_flag=True, help="List recent experiments")
@click.option("--list-protocols", is_flag=True, help="List all planned protocols")
@click.option("--approve", default=None, metavar="PROTOCOL_ID",
              help="Approve a pending protocol so it can be executed")
@click.option("--reject", default=None, metavar="PROTOCOL_ID",
              help="Reject a pending protocol")
@click.option("--loop", default=0, type=int, metavar="N",
              help="Run plan→approve→execute N times automatically (closed-loop demo). "
                   "Each cycle auto-approves its own protocol (no human gate in loop mode).")
@click.pass_context
def cli(ctx, command, config, demo, demo_real, list_experiments, list_protocols, approve, reject, loop):
    """Lab Agent — control your instruments with natural language."""

    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    ollama_ok = cfg_backend == "ollama" if (cfg_backend := _read_backend_cfg(config)) else False

    if not groq_ok and not anthropic_ok and not ollama_ok:
        console.print(
            "[bold red]Error:[/] No LLM API key found.\n"
            "Set GROQ_API_KEY (free at console.groq.com) or ANTHROPIC_API_KEY in your shell or .env file.\n"
            "Alternatively run Ollama locally and set [bold]backend: ollama[/bold] in config.yaml.",
            style="red",
        )
        sys.exit(1)

    if list_experiments:
        _show_experiments(config)
        return

    if list_protocols:
        _show_protocols(config)
        return

    if approve:
        _approve_protocol(approve, config)
        return

    if reject:
        _reject_protocol(reject, config)
        return

    if loop > 0:
        if not command:
            console.print("[red]--loop requires a goal command, e.g.:[/]\n"
                          "  python run.py --loop 3 \"Measure mitotic index in wells A1-A3\"")
            sys.exit(1)
        _run_closed_loop(command, config, n_cycles=loop)
        return

    if demo_real:
        _run_real_dataset_demo(config)
        return

    if demo:
        command = (
            "Run a mitosis assay on plate P001. "
            "First add DAPI and phospho-H3 antibody staining to wells A1, A2, and A3. "
            "Then image these wells with DAPI and FITC channels. "
            "Analyze the images and report the mitotic index for each well. "
            "If any well has mitotic index above 15%, also run flow cytometry on that well "
            "to validate, gating on FITC-high cells."
        )
        console.print(
            Panel(
                f"[bold]Demo command:[/]\n{command}",
                title="Lab Agent Demo (simulation)",
                border_style="cyan",
            )
        )

    if not command:
        click.echo(ctx.get_help())
        return

    _run_command(command, config)


def _run_real_dataset_demo(config: str):
    """Load real BBBC020 data and run the full mitosis assay pipeline."""
    from orchestrator import LabOrchestrator
    from src.data.dataset_loader import load_bbbc020, describe_bbbc020
    info = describe_bbbc020()
    if "error" in info:
        console.print(f"[red]{info['error']}[/]")
        sys.exit(1)

    console.print(Panel(
        "\n".join([
            f"[bold]Dataset:[/] {info['dataset']}",
            f"[bold]Biology:[/] {info['biology']}",
            f"[bold]Channels:[/] {', '.join(info['channels'])}",
            f"[bold]Timepoints:[/] {info['n_timepoints']} ({', '.join(info['timepoint_keys'])})",
            f"[bold]Replicates/timepoint:[/] {info['n_wells_per_timepoint']}",
        ]),
        title="Real Dataset: BBBC020 (Broad Institute)",
        border_style="magenta",
    ))

    # Load the real timelapse images
    plate_id = "BBBC020"
    timelapse_images = load_bbbc020()

    command = (
        f"Analyze the mitosis time-lapse experiment on plate {plate_id}. "
        f"The plate has {info['n_timepoints']} timepoints (Kontrolle/untreated, 15min, 30min, 1h, 2h, 24h) "
        f"and {info['n_wells_per_timepoint']} wells per timepoint (biological replicates). "
        f"Channels are DAPI (nuclei) and FITC (tubulin/cytoskeleton). "
        f"Analyze ALL timepoints and ALL wells. "
        f"Report how mitotic index changes over time — this is a Nocodazole treatment experiment "
        f"so you expect mitotic index to increase then plateau. "
        f"Flag any wells with unusually high or low cell counts. "
        f"After analysis, run flow cytometry on the well with the highest mitotic index "
        f"to validate the imaging result, gating on FITC-high (tubulin-high) cells as a mitosis marker. "
        f"Give a final biological interpretation of the results."
    )

    console.print(Panel(
        Text(command, style="white"),
        title="Command",
        border_style="blue",
    ))
    console.print()

    try:
        agent = LabOrchestrator(config_path=config)
        console.print(f"  [dim]LLM backend: [bold]{agent.backend_name}[/bold][/dim]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/]")
        sys.exit(1)

    try:
        result = agent.run(
            command,
            on_event=on_event,
            dataset_images={plate_id: timelapse_images},
        )
    except Exception as e:
        err = str(e)
        if "rate_limit" in err or "429" in err or "413" in err:
            console.print("\n[bold red]Groq API quota exhausted.[/bold red]")
            console.print("[yellow]The free-tier daily limit (100k tokens) has been reached.[/yellow]")
            console.print("[yellow]The quota resets at midnight UTC (8 AM UTC+8).[/yellow]")
            console.print("[dim]Re-run after the reset, or set ANTHROPIC_API_KEY to use Anthropic instead.[/dim]")
        else:
            console.print(f"\n[bold red]Experiment failed:[/bold red] {e}")
        sys.exit(1)

    console.print()
    # If the result looks like raw JSON or a tool response leak, show a clean fallback
    display_result = result
    if result.strip().startswith("{") or "<tool_response>" in result:
        display_result = "Analysis complete. See experiment events above for results."
    console.print(Panel(
        Text(display_result, style="bold green"),
        title="Final Biological Report",
        border_style="green",
    ))


def _run_closed_loop(goal: str, config: str, n_cycles: int = 3):
    """
    Automated closed-loop: plan → auto-approve → execute → interpret → repeat.
    Each cycle uses the previous cycle's LIMS results as context for the next plan.
    This is the M4 autonomous loop — no human gate per cycle.
    """
    from orchestrator import LabOrchestrator
    import uuid as _uuid

    agent = LabOrchestrator(config_path=config)
    console.print(Panel(
        f"[bold]Goal:[/] {goal}\n"
        f"[bold]Cycles:[/] {n_cycles}\n"
        f"[bold]Backend:[/] {agent.backend_name}\n\n"
        "[yellow]Auto-approval enabled — no human gate per cycle in loop mode.[/yellow]",
        title="Closed-Loop Experiment",
        border_style="cyan",
    ))

    all_results = []

    for cycle in range(1, n_cycles + 1):
        console.print(f"\n[bold cyan]── Cycle {cycle}/{n_cycles} ──[/bold cyan]")
        exp_id = f"EXP-LOOP-{str(_uuid.uuid4())[:6].upper()}"
        agent.db.create_experiment(exp_id, f"Loop cycle {cycle}", goal)

        # Build goal enriched with prior results
        context = ""
        if all_results:
            last = all_results[-1]
            context = (
                f"\nContext from previous cycle: top mitotic well was {last.get('top_well')} "
                f"with {last.get('top_value')}% mitotic index (mean {last.get('mean')}%)."
            )

        # Plan
        console.print("  [yellow]Planning...[/yellow]")
        plan = agent._dispatch_tool("plan_experiment", {
            "goal": goal + context,
            "plate_id": f"LOOP-P{cycle:02d}",
            "wells": ["A1", "B1", "C1"],
            "available_reagents": ["DAPI", "FITC"],
        }, exp_id, on_event)

        proto_id = plan.get("protocol_id")
        if not proto_id:
            console.print(f"  [red]Plan failed: {plan.get('message', plan)}[/red]")
            continue

        console.print(f"  Protocol: [cyan]{proto_id}[/cyan] ({plan.get('n_steps', 0)} steps) — auto-approving")

        # Auto-approve (no human gate in loop mode)
        agent.approve_protocol(proto_id, f"auto-approved cycle {cycle}")

        # Execute
        console.print("  [yellow]Executing...[/yellow]")
        xr = agent._dispatch_tool("execute_protocol", {"protocol_id": proto_id}, exp_id, on_event)
        exec_status = xr.get("execution_status", "unknown")
        steps_done = len([s for s in xr.get("step_results", []) if s["status"] == "done"])
        console.print(f"  Execution: [{'green' if exec_status == 'completed' else 'red'}]{exec_status}[/] — {steps_done} steps done")

        # Read LIMS for this cycle
        lims = agent._dispatch_tool("query_lims", {
            "experiment_id": exp_id, "summary": True, "metric": "mitotic_pct"
        }, exp_id, None)

        if "top_well" in lims:
            all_results.append(lims)
            console.print(
                f"  LIMS: top_well=[bold]{lims['top_well']}[/bold] "
                f"{lims['top_value']}% mitotic  |  mean={lims['mean']}%  n={lims['n_measurements']}"
            )
        else:
            console.print(f"  [dim]No LIMS data this cycle — {lims.get('error', '')}[/dim]")

        agent.db.complete_experiment(exp_id)

    # Summary across cycles
    console.print(f"\n[bold green]Closed loop complete — {n_cycles} cycles[/bold green]")
    if all_results:
        table = Table(title="Per-Cycle LIMS Summary", box=box.SIMPLE)
        table.add_column("Cycle", justify="right")
        table.add_column("Top Well")
        table.add_column("Top Mitotic %", justify="right")
        table.add_column("Mean %", justify="right")
        table.add_column("N wells", justify="right")
        for i, r in enumerate(all_results, 1):
            table.add_row(
                str(i),
                r.get("top_well", "?"),
                str(r.get("top_value", "?")),
                str(r.get("mean", "?")),
                str(r.get("n_measurements", "?")),
            )
        console.print(table)


def _show_experiments(config: str):
    from src.data import Database
    import yaml

    cfg = yaml.safe_load(Path(config).read_text())
    db = Database(cfg["data"]["database_url"])
    experiments = db.list_experiments()

    if not experiments:
        console.print("[dim]No experiments yet. Run a command to get started.[/]")
        return

    table = Table(title="Recent Experiments", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created")
    table.add_column("Command", max_width=60)

    for e in experiments:
        status_style = "green" if e["status"] == "completed" else "yellow"
        table.add_row(
            e["id"],
            Text(e["status"], style=status_style),
            e["created_at"][:19],
            e["command"],
        )
    console.print(table)


def _show_protocols(config: str):
    from src.data import Database
    import yaml

    cfg = yaml.safe_load(Path(config).read_text())
    db = Database(cfg["data"]["database_url"])
    protocols = db.list_protocols()

    if not protocols:
        console.print("[dim]No protocols yet. Use plan_experiment() to create one.[/]")
        return

    table = Table(title="Protocols", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Steps", justify="right")
    table.add_column("Est. (h)", justify="right")
    table.add_column("Validated")
    table.add_column("Approval")
    table.add_column("Execution")
    table.add_column("Goal", max_width=50)

    APPROVAL_STYLE = {
        "pending_approval": "yellow",
        "approved": "green",
        "rejected": "red",
    }
    for p in protocols:
        table.add_row(
            p["id"],
            str(p["steps"]),
            str(round(p["estimated_duration_h"] or 0, 1)),
            Text(p["validation_status"],
                 style="green" if p["validation_status"] == "valid" else "red"),
            Text(p["approval_status"],
                 style=APPROVAL_STYLE.get(p["approval_status"], "white")),
            p["execution_status"],
            p["goal"],
        )
    console.print(table)
    console.print(
        "\n[dim]To approve: [bold]python run.py --approve PROTOCOL_ID[/bold]"
        "  |  To reject: [bold]python run.py --reject PROTOCOL_ID[/bold][/dim]"
    )


def _approve_protocol(protocol_id: str, config: str):
    from orchestrator import LabOrchestrator
    agent = LabOrchestrator(config_path=config)
    result = agent.approve_protocol(protocol_id)
    console.print(
        Panel(
            f"[green]Protocol [bold]{result['protocol_id']}[/bold] approved.[/green]\n"
            f"Note: {result['note']}\n\n"
            f"Run it with:\n  python run.py \"execute protocol {protocol_id}\"",
            title="Protocol Approved",
            border_style="green",
        )
    )


def _reject_protocol(protocol_id: str, config: str):
    from orchestrator import LabOrchestrator
    agent = LabOrchestrator(config_path=config)
    result = agent.reject_protocol(protocol_id)
    console.print(
        Panel(
            f"[red]Protocol [bold]{result['protocol_id']}[/bold] rejected.[/red]",
            title="Protocol Rejected",
            border_style="red",
        )
    )


if __name__ == "__main__":
    cli()
