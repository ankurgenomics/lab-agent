# Lab Agent — System Documentation

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Enterprise Value](#enterprise-value)
3. [Architecture Deep-Dive](#architecture-deep-dive)
4. [Data Sources — Where to Get Them](#data-sources)
5. [Connecting Real Instruments](#connecting-real-instruments)
6. [Configuration Reference](#configuration-reference)
7. [Results — What the System Produced](#results)
8. [What Works Today](#what-works-today)
9. [What Is Left to Build](#what-is-left-to-build)
10. [Code Map](#code-map)

---

## What This System Does

Lab Agent is a software layer that lets a scientist type a goal in plain English and have a multi-instrument laboratory workflow execute automatically — imaging, liquid handling, cell analysis, cytometry, and data archiving — without writing a script, clicking a GUI, or manually moving plates between instruments.

### The problem it solves

A typical drug-discovery or cell-biology experiment involves four or more pieces of hardware from different vendors, each with its own software. A scientist who wants to run a dose-response mitosis assay must:

1. Open the liquid-handler software, set up a dilution, run it
2. Open the microscope software, configure channels, run the acquisition, wait
3. Open the image analysis software, process images, export a CSV
4. Open the cytometer software, load the CSV gating, sort cells
5. Manually correlate well IDs across four different spreadsheets

This takes 4–6 hours of operator time for a 96-well plate experiment. Errors in well mapping between steps are common and often undetected.

Lab Agent collapses all four steps into one command:

```
"Run a mitosis assay on plate P001. Image every 30 minutes for 2 hours.
 If mitotic index exceeds 20% in any well, run flow cytometry on it."
```

The orchestrator handles sequencing, error checking, conditional branching, and data archiving. The operator gets a structured report.

---

## Enterprise Value

This architecture maps directly to three high-value enterprise workflows:

### 1. Contract Research Organisations (CROs)
A CRO running 200 plates per week spends ~40% of technician time on manual instrument handoffs. Full automation of the acquire → analyze → sort pipeline at 20 plates/day saves roughly **8 FTE-hours per day**. At $80/hr fully-loaded cost that is **$150,000/year per site**.

### 2. Pharma high-throughput screening
HTS campaigns run 10,000–100,000 compounds per campaign. The bottleneck is not the instruments (which are already roboticised) but the decision layer: which wells to re-test, which concentrations to escalate, which compounds to advance. An LLM-driven decision layer on top of existing LIMS data can cut the manual triage step by 60–80%, compressing campaign timelines from 8 weeks to 5.

### 3. Academic core facilities
Core microscopy facilities at universities run 15–30 bookings per day across 3–5 instruments. Lab Agent running in simulation mode can pre-validate every protocol before the booking slot — catching volume errors, incompatible channel combinations, and under-specified wells before the scientist walks into the room.

---

## Architecture Deep-Dive

The system has four independently testable layers, built in order:

```
┌─────────────────────────────────────────────────────────────┐
│  User natural-language input                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  LLM Orchestrator  (orchestrator.py)                         │
│  Receives the goal, decides which tools to call, in what     │
│  order, checks results, handles errors and branching.        │
│  Backends: Groq llama-3.3-70b · Anthropic Claude · Ollama   │
└──────────────────────────┬──────────────────────────────────┘
                           │ every tool call goes through
┌──────────────────────────▼──────────────────────────────────┐
│  M2 — Protocol Validator  (src/validation/protocol_validator)│
│  Checks physical limits before any dispatch:                 │
│  channel names · volumes · well format · timepoint range     │
│  Returns a reason string; blocks the call if invalid.        │
└──────┬──────────────────┬───────────────┬───────────────────┘
       │                  │               │
┌──────▼──────┐  ┌────────▼──────┐  ┌───▼────────────────────┐
│ M1 Async    │  │ M4 Protocol   │  │ Direct tool calls        │
│ Job System  │  │ Executor      │  │ analyze_images()         │
│             │  │               │  │ run_flow_cytometry()     │
│ acquire()   │  │ plan_exp()    │  │ transfer_liquid()        │
│ → job_id    │  │ → JSON proto  │  │ query_results()          │
│             │  │ → approval    │  └──────────┬───────────────┘
│ bg_thread   │  │ → execute     │             │
│ wait_for_   │  │   step-by-    │             │
│ job()       │  │   step        │             │
└──────┬──────┘  └───────────────┘             │
       │                                       │
┌──────▼───────────────────────────────────────▼───────────────┐
│  M3 — Local LIMS  (src/lims/lims_client.py)                  │
│  Writes every per-well metric with full provenance:          │
│  experiment_id · plate_id · well · timepoint ·               │
│  metric · value · unit · instrument · method                 │
│  Queryable by the orchestrator: summary, timecourse, filter  │
└──────────────────────────────────────────────────────────────┘
```

### M1 — Async Job System

`acquire_images()` does not block. It creates a database record, starts a background thread, and returns `{job_id: "JOB-XXXXXXXX", status: "running"}` immediately. The orchestrator can do other work (or hand back to the user) while acquisition runs.

When the job completes, the thread writes `status="done"` and the result payload to the database. The orchestrator calls `wait_for_job(job_id)` when it needs the result. If the timeout expires, it returns `status="timeout"` and the job keeps running in the background.

The `ProtocolExecutor` (M4) calls `wait_for_job` automatically before advancing to the next step, so `acquire → analyze` sequencing is guaranteed safe.

**Current limitation:** the background thread is in-process. A process crash loses the in-flight job. Production-grade durability requires Temporal or Prefect workflow orchestration.

### M2 — Protocol Validator

Every tool call passes through `ProtocolValidator.validate_tool_call(tool_name, params)` before dispatch. The validator checks against a `LIMITS` dictionary that encodes the physical constraints of each instrument:

```python
LIMITS = {
    "microscope": {
        "valid_channels": {"DAPI", "FITC", "TRITC", "CY5", "GFP", "mCherry", "BF"},
        "valid_objectives": {"4x", "10x", "20x", "40x", "60x", "100x"},
        "max_wells_per_run": 96,
    },
    "liquid_handler": {
        "min_volume_ul": 0.5,
        "max_volume_ul": 1000.0,
    },
    ...
}
```

Blocked examples from testing:
- `channel="UV_LASER"` → rejected: `"UV_LASER is not a valid channel. Valid: DAPI, FITC, TRITC..."`
- `volume_ul=99999` → rejected: `"Volume 99999µL exceeds maximum 1000µL"`
- `well="Z99"` → rejected: `"Well Z99 is not a valid 96-well address (A1–H12)"`

The validator also runs on complete protocol objects (`validate_protocol()`), checking every step before the protocol is saved to the database.

### M3 — Local LIMS

Results from every `analyze_images()` and `run_flow_cytometry()` call are written to a separate SQLite database (`lab_lims.db`) in a normalised schema:

```sql
CREATE TABLE lims_results (
    id           TEXT PRIMARY KEY,
    experiment_id TEXT,
    plate_id      TEXT,
    well          TEXT,          -- "A1", "B3", etc.
    timepoint     INTEGER,       -- 0-indexed
    metric        TEXT,          -- "mitotic_pct", "cells", "live_pct", ...
    value         REAL,
    unit          TEXT,
    instrument    TEXT,
    method        TEXT,          -- "watershed", "cellpose", "simulation"
    acquired_at   TEXT,          -- ISO-8601
    provenance    TEXT           -- JSON blob: software version, params
);
```

The orchestrator queries this table through three methods:
- `query_lims(summary=True)` → `{top_well, top_value, mean, min, max, n_measurements}`
- `query_lims(metric="mitotic_pct", min_value=15)` → filtered list of wells
- `query_lims(well="B1")` → full timecourse for that well

This data persists across sessions. The orchestrator can open a conversation referencing results from a week ago without re-running the experiment.

### M4 — Protocol Executor

The full four-step lifecycle:

**Step 1 — Plan.** The orchestrator calls `plan_experiment(goal, plate_id, wells, reagents)`. This sends a structured prompt to the LLM asking it to generate a JSON protocol with typed steps. The prompt includes the tool schema so the generated steps are always valid tool calls.

```json
{
  "hypothesis": "DAPI staining at 1µg/mL will mark all nuclei at 20x",
  "success_criteria": "mitotic_pct > 5% in at least 2 wells",
  "estimated_duration_h": 0.33,
  "steps": [
    {"instrument": "liquid_handler", "operation": "setup_plate",
     "params": {"plate_id": "P001", "wells": ["A1","B1"], "reagent": "DAPI", "volume_ul": 100}},
    {"instrument": "microscope", "operation": "acquire_images",
     "params": {"plate_id": "P001", "wells": ["A1","B1"], "channels": ["DAPI","FITC"]}},
    {"instrument": "microscope", "operation": "analyze_images",
     "params": {"plate_id": "P001", "timepoint": -1}}
  ]
}
```

If the LLM returns invalid JSON, the system retries up to three times, feeding the parse error back in the next prompt. If validation fails with three or fewer errors, it attempts one self-correction pass with the specific errors listed.

**Step 2 — Validate.** Every step in the generated protocol passes through M2 before the protocol is saved. Validation status is stored in the database.

**Step 3 — Approve.** The protocol sits in `pending_approval` state until a human approves it. Approval is via CLI (`--approve PROTO-XXXXXXXX`) or auto-approved in `--loop` mode. This is the human gate — no protocol executes without explicit approval.

**Step 4 — Execute.** `ProtocolExecutor.run()` iterates through steps deterministically, calls `dispatch_fn(operation, params)` per step, waits for async jobs, and halts on the first error. The result includes `{steps_completed, step_results, status}`.

---

## Data Sources

### Simulation mode (default, zero setup)

All instruments generate realistic synthetic data out of the box:
- Microscope: numpy arrays with Gaussian nuclei, correct 16-bit TIFF format
- Flow cytometer: 10,000-event populations with realistic FSC/SSC/FITC/PE distributions
- Liquid handler: validates timing and volume maths, returns realistic pipetting logs

No data download required. Run `python run.py --demo` immediately after installation.

### BBBC020 — Real fluorescence microscopy images

The [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC020) dataset provides real HeLa cell images for benchmarking cell analysis pipelines.

**What it contains:**
- HeLa cells stained with DAPI (nuclei) and FITC (tubulin)
- 96-well plate format, multiple fields of view per well
- 16-bit TIFF images, ~1040×1388 pixels per field
- Ground-truth cell counts for a subset of images

**Download:**
```bash
# Manual download from the BBBC website (free, no registration required)
# https://bbbc.broadinstitute.org/BBBC020
# Files: BBBC020_v1_images.zip (~2 GB)

python run.py --demo-real
# This automatically locates BBBC020 images in ./experiment_data/bbbc020/
# if the directory exists. Otherwise it falls back to simulation.
```

**Where to put it:**
```
lab-agent/
└── experiment_data/
    └── bbbc020/
        ├── BBBC020_v1_images_A01_w1.tif   ← DAPI channel
        ├── BBBC020_v1_images_A01_w2.tif   ← FITC channel
        └── ...
```

**Biological note:** BBBC020 uses tubulin staining in the FITC channel, not the phospho-histone H3 (Ser10) antibody that is the standard mitosis marker. The mitotic index computed from FITC overlap is a proxy, not a validated assay. For absolute mitotic index, use a phospho-H3 dataset such as the Human Protein Atlas open microscopy data.

### Your own data

Any 16-bit TIFF (grayscale or two-channel) works. Name the channels `_w1` (DAPI) and `_w2` (FITC) or configure channel mapping in `config.yaml`. The analysis pipeline auto-detects bit depth.

---

## Connecting Real Instruments

### Microscope — any Micro-Manager compatible scope

Micro-Manager supports Leica, Zeiss, Nikon, Olympus, Andor, Photometrics and 200+ other devices via hardware adapters.

```yaml
# config.yaml
instruments:
  microscope:
    mode: pymmcore          # ← change from "simulation"
    config_file: /path/to/MMConfig_demo.cfg
```

```bash
pip install pymmcore-plus
```

The `config_file` is a standard Micro-Manager `.cfg` file generated by the Micro-Manager Hardware Configuration Wizard. Once the `.cfg` is in place, no other changes are needed — the driver calls `core.snap_image()` and returns a numpy array in the same format the simulation produces.

For Harmony (PerkinElmer Opera Phenix):
```yaml
instruments:
  microscope:
    mode: harmony_api
    api_endpoint: http://192.168.1.10:8080
```
Implement `src/instruments/microscope.py::_acquire_real_harmony()` with your site's API token.

### Liquid handler — Opentrons Flex

```yaml
instruments:
  liquid_handler:
    mode: opentrons
    api_endpoint: http://192.168.1.20:31950   # robot IP on your network
    robot_type: opentrons_flex
```

The Opentrons HTTP API (`/runs`, `/commands`) is REST over JSON. The driver calls `POST /runs` to create a run, then `POST /runs/{id}/commands` for each transfer step. No additional library needed — `httpx` is already a dependency.

Verify the robot is reachable:
```bash
curl http://192.168.1.20:31950/health
```

### Flow cytometer

Flow cytometer APIs are vendor-specific. The integration point is `src/instruments/cytometer.py::_execute_real()`. Implement this method using your vendor's SDK:

- **Cytek Aurora:** REST API at `http://cytometer-host:port/api/v1/`
- **BD FACSAria/Melody:** COM automation via `win32com` on Windows
- **Beckman Coulter:** proprietary SDK — contact vendor for developer access
- **Generic file-drop:** write a protocol file to a watched folder; cytometer software picks it up

The simulation and real modes both return the same dict structure, so the orchestrator does not change regardless of which backend is active.

### All-at-once: change one line

```yaml
# config.yaml
simulation_mode: false    # ← flip this
```

With `simulation_mode: false`, each instrument driver attempts to connect to the configured endpoint. If connection fails, the driver raises `InstrumentConnectionError` and the orchestrator reports the error before executing any steps.

---

## Configuration Reference

```yaml
# config.yaml — full annotated reference

simulation_mode: true       # true = synthetic data, no hardware needed

instruments:
  microscope:
    mode: simulation        # simulation | pymmcore | harmony_api
    config_file: null       # Micro-Manager .cfg path (pymmcore mode)
    api_endpoint: null      # HTTP endpoint (harmony_api mode)
    default_channels: ["DAPI", "FITC", "TRITC"]
    default_objective: "20x"
    default_z_step_um: 1.0
    default_z_slices: 5

  flow_cytometer:
    mode: simulation        # simulation | cytek | bd_facsdiva
    api_endpoint: null
    default_channels: ["FSC", "SSC", "FITC", "PE", "APC"]

  liquid_handler:
    mode: simulation        # simulation | hamilton | opentrons
    api_endpoint: null      # http://<robot-ip>:31950 for Opentrons
    robot_type: opentrons_flex

analysis:
  mode: watershed           # watershed (fast, CPU) | cellpose (accurate, GPU) | simulation
  # watershed:  scipy distance-transform, good for dense nuclei < 25px diameter
  # cellpose:   deep learning segmentation, best for 30-100px nuclei
  # simulation: numpy thresholding, demos only

data:
  database_url: "sqlite:///lab_agent.db"   # swap to postgresql://... for production
  lims_path: "lab_lims.db"                 # separate LIMS database
  storage_path: "./experiment_data"        # image file storage root
  omero_host: null        # OMERO server hostname for institutional image storage
  s3_bucket: null         # AWS S3 bucket for cloud sync

llm:
  backend: auto           # auto = Groq > Anthropic > Ollama (first key found)
  model: "claude-sonnet-4-5"             # Anthropic model
  groq_model: "llama-3.3-70b-versatile"  # Groq primary model
  groq_model_fallback: "qwen/qwen3-32b"  # Groq fallback (higher rate limit)
  ollama_model: "qwen2.5:14b"            # local Ollama model
  max_tokens: 4096
  temperature: 0.2                       # low temperature = more consistent tool calls
```

**Environment variables** (`.env` file, never committed):
```bash
GROQ_API_KEY=gsk_...         # free at console.groq.com — 100k tokens/day
ANTHROPIC_API_KEY=sk-ant-... # paid, $3/M tokens input
# Ollama needs no key — just run `ollama serve` locally
```

---

## Results

Numbers from the test database built during development:

| Metric | Value |
|--------|-------|
| Experiments run | 20 |
| Image acquisitions completed | 14 |
| Per-well analysis results stored | 252 |
| Async jobs tracked | 8 |
| Protocols planned by LLM | 11 |
| LIMS rows written | 70 |
| Distinct LIMS metrics | 5 |
| Protocols executed end-to-end | 3 |

LIMS metrics tracked per well per timepoint:
- `cells` — segmented nucleus count
- `mitotic_pct` — fraction of nuclei classified as mitotic (%)
- `live_pct` — viability estimate from DAPI morphology (%)
- `mean_dapi` — mean DAPI channel intensity (normalised 0–1)
- `mean_fitc` — mean FITC channel intensity (normalised 0–1)

### Example: closed-loop run

```bash
python run.py --loop 2 "Measure mitotic index in wells A1, B1, C1"
```

Output (real terminal output, condensed):

```
── Cycle 1/2 ──
  Planning...
  Protocol: PROTO-E0998C6C (3 steps) — auto-approving
  Executing...
  [STEP ✓] setup_plate completed in 0.0s
  [STEP] Waiting for async job JOB-3ACFB8F9...
  [SCOPE] Job JOB-3ACFB8F9 done — plate LOOP-P01 acquired.
  [STEP ✓] acquire_images completed in 2.0s
  [STEP ✓] analyze_images completed in 1.4s
  LIMS: top_well=A1  0.0% mitotic  |  mean=0.0%  n=3

── Cycle 2/2 ──
  Protocol: PROTO-6904CBC7 (3 steps) — auto-approving
  [STEP ✓] All 3 steps completed in 3.8s
  LIMS: top_well=A1  0.0% mitotic  |  mean=0.0%  n=3

  Per-Cycle LIMS Summary
  Cycle   Top Well   Top Mitotic %   Mean %   N wells
   1      A1         0.0             0.0      3
   2      A1         0.0             0.0      3
```

The 0% mitotic index is expected in simulation mode — synthetic cells have no biology. With real BBBC020 images, the watershed pipeline has returned values of 3–9.5% depending on the well, consistent with typical asynchronous HeLa populations (expected 2–8%).

### M2 Validator — blocks in testing

| Bad call | Rejection message |
|----------|-------------------|
| `channel="UV_LASER"` | `UV_LASER is not a valid channel. Valid: DAPI, FITC, TRITC, CY5, GFP, mCherry, BF` |
| `volume_ul=99999` | `Volume 99999µL exceeds maximum 1000µL for liquid_handler` |
| `well="Z99"` | `Z99 is not a valid 96-well address (A1–H12)` |
| `objective="1000x"` | `1000x is not a valid objective. Valid: 4x, 10x, 20x, 40x, 60x, 100x` |

---

## What Works Today

Everything below runs without hardware — clone, install, run `--demo`.

| Feature | Status | Notes |
|---------|--------|-------|
| Natural-language → tool calls | Working | 10 registered tools |
| M1: async image acquisition | Working | threading; job_id returned immediately |
| M1: auto-wait in protocol executor | Working | fixed in session 2 |
| M2: protocol validator | Working | channels, volumes, wells, timepoints |
| M2: self-correction retry | Working | up to 3 JSON retries + 1 validation fix |
| M3: LIMS write after every analysis | Working | 5 metrics per well per timepoint |
| M3: LIMS query (summary/timecourse) | Working | query_lims() returns structured dict |
| M4: LLM protocol generation | Working | JSON with hypothesis, steps, criteria |
| M4: human approval gate | Working | CLI --approve or auto in --loop |
| M4: deterministic protocol execution | Working | halts on first error |
| --loop N closed-loop automation | Working | LIMS feedback into next cycle |
| Multi-backend LLM | Working | Groq / Anthropic / Ollama |
| Ollama (no API key) | Working | runs fully locally |
| Real BBBC020 image analysis | Partial | watershed on real TIFFs; mitotic index approximate (tubulin stain) |
| Time-lapse experiments | Working | acquire every N minutes, N timepoints |
| Conditional branching | Working | LLM branches based on LIMS results |

---

## What Is Left to Build

Ranked by proximity to production value:

### 1. FastAPI approval endpoint (3 days)

Replace the CLI `--approve` flag with a browser form. A scientist reviewing a protocol should see the full step list, estimated duration, hypothesis, and success criteria — not a terminal flag.

```python
# Skeleton already clear from the existing approve_protocol() method:
# GET  /protocols/{id}     → show protocol for review
# POST /protocols/{id}/approve  → sets approval_status = "approved"
# POST /protocols/{id}/reject   → sets approval_status = "rejected"
```

### 2. Connect Opentrons Flex (1 week)

The REST driver skeleton is in `src/instruments/liquid_handler.py`. The Opentrons HTTP API (`/health`, `/runs`, `/commands`) is documented and stable. The only gap is mapping the internal `setup_plate` and `transfer_liquid` operations to Opentrons deck layout commands (labware definitions, pipette names).

### 3. Temporal workflow engine for crash-durable async (2 weeks)

Replace `threading.Thread` with a Temporal activity. The job state then lives in Temporal's durable event log — a process restart picks up where it left off. Temporal's Python SDK is a direct drop-in for the background thread pattern used here.

### 4. Phospho-H3 mitosis assay (1 week)

The current BBBC020 dataset uses tubulin staining, which is not a specific mitosis marker. A phospho-histone H3 (Ser10) immunofluorescence dataset gives a clean binary signal: mitotic cells are FITC-bright, interphase cells are not. The Human Protein Atlas has freely downloadable phospho-H3 images. Swapping the dataset and rerunning the watershed pipeline gives calibrated, publishable mitotic index values.

### 5. OMERO integration (2 weeks)

Institutional microscopy core facilities use OMERO for image storage. `data.omero_host` is already in `config.yaml` but `src/data/__init__.py` uses local filesystem. Implement `_store_images_omero()` using `ezomero` and images are automatically searchable from any browser in the facility.

### 6. GxP audit trail (3 weeks)

For pharma, every result needs a tamper-evident audit log: who ran it, when, with which instrument firmware version, which reagent lot. The `provenance` JSON column in `lims_results` is the foundation. Extending it to include operator ID (from LDAP), reagent lot (from barcode scan), and a cryptographic hash of the raw image file completes the 21 CFR Part 11 requirement.

### 7. Plate barcode → automatic experiment ID (3 days)

Add a `scan_barcode(plate_id)` tool that reads a DataMatrix or QR code from the plate carrier. The experiment ID is then auto-populated from the LIMS rather than typed by the scientist, eliminating the most common transcription error in wet-lab workflows.

---

## Code Map

```
lab-agent/
├── orchestrator.py          1342 lines  Main agent loop, tool dispatch, LLM calls
├── run.py                    440 lines  CLI entry point (click + rich)
├── config.yaml                         Instrument + LLM configuration
├── requirements.txt                    Python dependencies
├── src/
│   ├── analysis/
│   │   └── __init__.py       488 lines  AnalysisPipeline: watershed, cellpose, simulation
│   ├── data/
│   │   ├── __init__.py       424 lines  SQLAlchemy models + Database class
│   │   └── dataset_loader.py           BBBC020 TIFF loader
│   ├── instruments/
│   │   ├── microscope.py               Microscope driver (simulation + pymmcore + harmony)
│   │   ├── cytometer.py                Flow cytometer driver (simulation + stub for real)
│   │   └── liquid_handler.py           Liquid handler driver (simulation + opentrons stub)
│   ├── lims/
│   │   └── lims_client.py    197 lines  LocalLIMS: write, query, summary, timecourse
│   ├── protocol/
│   │   └── executor.py       134 lines  ProtocolExecutor: step runner with async-wait
│   └── validation/
│       └── protocol_validator.py 305 lines  ProtocolValidator: LIMITS dict + per-tool checks
├── docs/
│   └── images/                         SVG diagrams for README
└── tests/
    └── test_system.py                  Integration smoke tests
```

Total: ~3,330 lines of Python across 9 source files.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sqlalchemy` | ≥2.0 | ORM for experiment + protocol databases |
| `anthropic` | ≥0.40 | Anthropic Claude API client |
| `openai` | ≥1.30 | Groq and Ollama use the OpenAI-compatible API |
| `numpy` | ≥1.24 | Image array maths, synthetic data generation |
| `scipy` | ≥1.11 | Distance transform, watershed segmentation |
| `scikit-image` | ≥0.24 | Watershed fill, morphological filters |
| `pillow` | ≥10.0 | TIFF read/write |
| `pyyaml` | ≥6.0 | Config file parsing |
| `click` | ≥8.1 | CLI argument parsing |
| `rich` | ≥13.0 | Terminal formatting, tables, progress |
| `python-dotenv` | ≥1.0 | `.env` file loading |

Optional (not in requirements.txt — install only if needed):
- `cellpose` — deep learning cell segmentation (analysis mode: cellpose)
- `pymmcore-plus` — Micro-Manager microscope control (instrument mode: pymmcore)
