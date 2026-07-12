# lab-agent

Running a mitosis assay used to mean opening four different vendor GUIs in sequence, copying well IDs between spreadsheets, and hoping nothing got misaligned between the liquid handler software and the microscope software and the image analysis software and the cytometer software. Every instrument is from a different vendor, none of them talk to each other, and the scientist in the middle becomes the glue.

This project is my attempt to remove that glue layer.

You describe the experiment in plain English. The system works out the order of operations, dispatches each instrument, waits for it to finish, checks the result, and branches if something interesting happens. Everything lands in a local database with the full record of what ran, when, on which instrument, and with which parameters.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![SQLite](https://img.shields.io/badge/storage-SQLite%20%2B%20LIMS-green.svg)](https://sqlite.org)
[![License MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

```bash
python run.py "Run a mitosis assay on plate P001. Image every 30 minutes.
               If mitotic index exceeds 20%, sort positive cells by flow cytometry."
```

That single command runs: liquid handler staining → confocal acquisition (async, non-blocking) → watershed nucleus segmentation → LIMS write → conditional cytometry sort. The operator does nothing between steps.

---

## How it works

![Architecture overview](docs/images/architecture-overview.png)

The system has four layers, each independent and testable without hardware.

**The orchestrator** receives your command and decides which instruments to call, in what order, and what to do with the results. It uses a language model as the reasoning engine — describe a dose-response experiment, a time-lapse assay, or a conditional sort, and it figures out the tool sequence.

**The safety validator (M2)** sits between the orchestrator and every instrument. Before any tool call goes out, it checks the parameters against known physical limits — valid channel names, volume ranges, well formats. If the orchestrator asks for `channel="UV_LASER"` or `volume_ul=99999`, the call is blocked with a specific reason and the orchestrator corrects and retries. Nothing physically impossible reaches an instrument.

**Async job tracking (M1)** handles the fact that image acquisition takes time. `acquire_images()` returns a `job_id` immediately and runs in a background thread. The orchestrator can keep working or wait — the protocol executor waits automatically before moving to the next step. A process crash loses in-flight jobs (threading, not Temporal), which is the main production gap.

**The LIMS (M3)** writes every result — cell count, mitotic index, viability, mean channel intensities — to a local SQLite database keyed by `experiment_id · plate_id · well · timepoint`. The orchestrator can query it: ask for the top well by mitotic index, pull a timecourse for well B1, or filter everything above 15%. Results persist across sessions so you can reference an experiment from last week without re-running it.

**The protocol executor (M4)** handles structured multi-step experiments. Ask it to plan an experiment and it generates a typed JSON protocol — hypothesis, success criteria, ordered steps with parameters. The validator checks every step. The protocol then sits waiting for a human to approve it before anything runs. Once approved, the executor runs each step deterministically and halts on the first error.

---

## Milestones

![M1–M4 milestones](docs/images/milestones.png)

---

## Automated closed-loop

![Closed-loop sequence](docs/images/closed-loop-sequence.png)

`--loop N` runs the full cycle N times without stopping. After each cycle, the LIMS results from that run are fed into the context for the next plan. In a real experiment, this would tighten the focus over time — if cycle 1 finds well B1 has 9% mitotic index, cycle 2 might concentrate on that condition.

```bash
python run.py --loop 3 "Measure mitotic index in wells A1, B1, C1"
```

The human approval gate is skipped in loop mode. That's intentional for demos and closed-loop testing, but a production deployment would pause here and wait for sign-off before executing each cycle.

---

## LIMS — what gets recorded

![LIMS chart](docs/images/lims-chart.png)

After every analysis step, five metrics are written per well per timepoint: `cells` (segmented nucleus count), `mitotic_pct` (fraction of nuclei classified as mitotic), `live_pct` (viability estimate from DAPI morphology), `mean_dapi`, and `mean_fitc`. The chart above shows mitotic index across three experiments — values of 3–9.5% are consistent with what you'd expect from an asynchronous HeLa population.

---

## What the validator blocks

![Validator](docs/images/validator-blocks.png)

```
channel="UV_LASER"   →  "UV_LASER is not a valid channel. Valid: DAPI, FITC, TRITC, CY5..."
volume_ul=99999      →  "Volume 99999µL exceeds maximum 1000µL for liquid_handler"
well="Z99"           →  "Z99 is not a valid 96-well address (A1–H12)"
objective="1000x"    →  "1000x is not a valid objective. Valid: 4x, 10x, 20x, 40x, 60x, 100x"
```

Each rejection includes a typed reason string. The orchestrator feeds it back and retries with corrected parameters. In testing, every one of these cases resolved on the first retry.

---

## Numbers

| | |
|---|---|
| Experiments run in testing | 20 |
| Image acquisitions | 14 |
| Per-well analysis results stored | 252 |
| Async jobs tracked | 8 |
| Protocols planned end-to-end | 11 |
| LIMS rows written | 70 |
| Metrics per well | 5 |
| Registered tool calls | 10 |
| Source lines of Python | ~3,300 |
| LLM backends | 3 — Groq · Anthropic · Ollama |

---

## Quick start

No hardware needed. Simulation mode generates synthetic images and cytometry populations so you can run the full pipeline on any laptop.

```bash
git clone https://github.com/ankurgenomics/lab-agent
cd lab-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Any one of these. Ollama needs no key at all.
cp .env.example .env
# add GROQ_API_KEY or ANTHROPIC_API_KEY to .env

python run.py --demo
```

For your own command:
```bash
python run.py "Image wells A1-A6 on plate P001 with DAPI and FITC. Analyze mitotic index."
```

---

## Example commands

```bash
# Basic assay
python run.py "Run the mitosis assay on plate P001, wells A1-A4"

# Time-lapse with a conditional branch
python run.py "Image plate P002 every 30 minutes for 3 hours. After each timepoint,
               check mitotic index. If any well exceeds 25%, run flow cytometry on it."

# Dose-response
python run.py "Set up a 1:2 serial dilution of compound X in wells A1-H1 of plate P003,
               image at 20x with DAPI and FITC, report dose-response."

# Query the LIMS
python run.py "Show all wells from plate P001 where mitotic index exceeded 15%"

# Closed-loop
python run.py --loop 3 "Measure mitotic index in A1, B1, C1 and propose next experiment"

# Review pending protocols before they run
python run.py --list-protocols
python run.py --approve PROTO-XXXXXXXX
```

---

## How M4 (protocol planning) works step by step

```
1. You call:  plan_experiment(goal, plate_id, wells, reagents)

2. The language model generates a JSON protocol:
      {
        "hypothesis": "...",
        "success_criteria": "mitotic_pct > 5% in at least 2 wells",
        "steps": [
          { "instrument": "liquid_handler", "operation": "setup_plate", ... },
          { "instrument": "microscope",     "operation": "acquire_images", ... },
          { "instrument": "microscope",     "operation": "analyze_images", ... }
        ]
      }
   If the JSON is malformed, it retries up to 3 times with the parse error fed back.

3. The validator checks every step. If there are ≤3 errors, it makes one
   self-correction attempt with the specific problems listed.

4. The protocol is saved as  pending_approval  — nothing runs yet.

5. You approve it:
      python run.py --approve PROTO-XXXXXXXX
   (or it auto-approves in --loop mode)

6. The executor runs each step in order, waits for async jobs between steps,
   and stops on the first error. Results go to LIMS.
```

---

## Connecting real hardware

The change is one or two lines in `config.yaml`. The rest of the code doesn't move.

### Any Micro-Manager microscope (Leica, Zeiss, Nikon, Olympus, 200+ others)

```yaml
instruments:
  microscope:
    mode: pymmcore
    config_file: /path/to/MMConfig.cfg
```
```bash
pip install pymmcore-plus
```

The `.cfg` file comes from Micro-Manager's Hardware Configuration Wizard — it describes your specific hardware. Once it's in place, the driver calls `core.snap_image()` and returns a numpy array in the same format the simulation produces.

### Opentrons Flex

```yaml
instruments:
  liquid_handler:
    mode: opentrons
    api_endpoint: http://192.168.1.20:31950
```

The Opentrons robot exposes a REST API. The driver calls `/runs` and `/commands` — no extra library needed.

### Flow cytometer

Fill in `_execute_real()` in `src/instruments/cytometer.py` with your vendor's SDK:
- Cytek Aurora: REST at `http://cytometer-host/api/v1/`
- BD FACSAria: COM automation via `win32com` on Windows
- Generic: drop a protocol file into a watched folder

### Turn simulation off

```yaml
simulation_mode: false    # config.yaml
```

With this set, each driver tries to connect to its configured endpoint on startup. If it can't connect, you get `InstrumentConnectionError` before any steps run.

---

## Data sources

The demo works immediately with synthetic data — no download needed.

For real microscopy images, the [BBBC020 dataset](https://bbbc.broadinstitute.org/BBBC020) from the Broad Institute has HeLa cells stained with DAPI and FITC, in 96-well plate format, free to download. Put the TIFFs here:

```
experiment_data/
└── bbbc020/
    ├── BBBC020_v1_images_A01_w1.tif   ← DAPI channel
    ├── BBBC020_v1_images_A01_w2.tif   ← FITC channel (tubulin — see note below)
    └── ...
```

One caveat: BBBC020 uses tubulin staining in the FITC channel, not phospho-histone H3, which is the standard specific marker for mitotic cells. The mitotic index numbers this produces are an approximation. For a properly validated assay you need a phospho-H3 dataset — the Human Protein Atlas has freely available images with that staining.

---

## What still needs building

| | effort | why it matters |
|---|---|---|
| FastAPI approval UI | 3 days | right now approval is a CLI flag; a browser form is what scientists actually need |
| Opentrons deck layout + labware mapping | 1 week | the REST driver is there, the labware definitions aren't |
| Temporal for crash-durable async | 2 weeks | a process restart currently loses in-flight jobs |
| Phospho-H3 dataset | 1 week | gives accurate mitotic index rather than the tubulin proxy |
| OMERO integration | 2 weeks | most academic facilities use OMERO for image storage |
| GxP audit trail | 3 weeks | needed for any pharma or CRO deployment |

Full details in [docs/SYSTEM.md](docs/SYSTEM.md).

---

## Where this makes financial sense

**CROs:** Around 40% of a technician's time on a 96-well plate campaign is instrument handoffs — moving plates, re-entering well IDs, waiting for one piece of software before opening the next. At 20 plates/day, automating the acquire → analyse → sort chain saves roughly 8 hours of operator time daily. At UK CRO fully-loaded rates, that is about £120,000/year per site.

**Pharma high-throughput screening:** A campaign runs 10,000–100,000 compounds. The instruments handle the volume fine — the bottleneck is the decision layer afterwards, working out which wells to escalate and which concentrations to re-test. Putting a reasoning layer on top of live LIMS data cuts that triage step and can compress an 8-week campaign to 5 weeks.

**Academic core facilities:** Pre-validating protocols before the booking slot catches bad experiments before the scientist walks in the door. Wrong channel combinations, volumes out of range, wells specified incorrectly — these are common. At 20 bookings a day, catching even half of them saves an hour or two of corrective time every morning.

---

## Stack

| | |
|---|---|
| Orchestration | LLM function calling — Groq llama-3.3-70b, Anthropic Claude, or Ollama locally |
| Cell segmentation | scipy watershed (fast, CPU-only) or Cellpose (more accurate, optional GPU) |
| Database | SQLite via SQLAlchemy — swap the connection string for PostgreSQL in production |
| Image I/O | Pillow + numpy — handles 8-bit, 16-bit, grayscale, and two-channel TIFFs |
| Microscope driver | pymmcore-plus → Micro-Manager hardware abstraction layer |
| Liquid handler | Opentrons HTTP REST API |
| CLI | Click + Rich |

---

## Files

```
lab-agent/
├── orchestrator.py        1342 lines — the main loop: receives commands, calls tools, handles results
├── run.py                  440 lines — CLI entry point
├── config.yaml                        instrument settings, LLM backend, storage paths
├── requirements.txt                   all dependencies with minimum versions
├── src/
│   ├── analysis/           488 lines — three analysis modes: watershed, cellpose, simulation
│   ├── data/               424 lines — SQLAlchemy models for experiments, jobs, protocols
│   ├── instruments/                   one driver file per instrument, simulation + real
│   ├── lims/               197 lines — write results, query by well/metric/timepoint
│   ├── protocol/           134 lines — step-by-step executor with async-wait
│   └── validation/         305 lines — physical limits dictionary, per-tool checks
└── tests/
    └── test_system.py                 integration smoke tests
```

---

## License

MIT. See [LICENSE](LICENSE).
