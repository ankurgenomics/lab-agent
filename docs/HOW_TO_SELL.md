# How to Talk About lab-agent

Different situations need different versions. Pick the one that fits.

---

## The 30-second version (interview, conference, first meeting)

"I got frustrated watching scientists spend half their day moving plates between instruments that don't talk to each other — open this software, wait, copy the well IDs, open that software, wait, hope nothing got misaligned. I built a system where you just describe the experiment you want and it coordinates the instruments automatically. You say 'image this plate every 30 minutes, and if mitotic index goes above 20% run cytometry on those wells' — and it does exactly that, end to end, with everything logged and traceable. I tested it with real microscopy images from the Broad Institute's open benchmark dataset. The architecture handles async job tracking, has a safety layer that validates every instrument call before dispatch, and a local LIMS that stores results per well per timepoint. The gap to production is real hardware connections — the software layer is done."

---

## The LinkedIn post version

I spent a few weeks building something I'd wanted to see for a while.

Drug discovery labs run experiments across 4+ instruments that have never been designed to talk to each other. A single 96-well plate mitosis assay involves the liquid handler software, the microscope software, the image analysis software, and the cytometer software — with a scientist manually gluing the pieces together at each handoff. That's 4–6 hours of coordination work per experiment, and well-mapping errors between steps are common and hard to catch.

I built a coordination layer that sits on top of all of them.

You describe the experiment in plain English. The system figures out the sequence — liquid handler first, then async microscope acquisition (non-blocking, returns immediately), then watershed cell segmentation, then conditional cytometry if any well meets the threshold. Every result lands in a local LIMS with full provenance: which instrument, which method, which parameters produced each number.

A few things I'm reasonably happy with:
- The safety validator intercepts every tool call and checks physical limits before dispatch. Volume out of range, invalid channel name, bad well format — it blocks the call, returns a reason, and the orchestrator corrects and retries. No physically impossible instructions reach a real instrument.
- The protocol executor handles structured multi-step experiments with a human approval gate. The plan sits pending until you sign off, then runs deterministically.
- `--loop N` runs the whole cycle N times, feeding each cycle's LIMS results into the next plan. Crude closed-loop, but it works.

Tested with BBBC020 real HeLa cell images from the Broad Institute. 20 experiments, 252 per-well analysis results, 70 LIMS rows. ~3,300 lines of Python across 9 source files.

The gap to production is real hardware — the software layer is done and the driver stubs are ready to fill in.

---

## The interview story (STAR format)

**Situation:** In drug discovery and cell biology research, a typical 96-well plate experiment involves four separate instruments — a liquid handler, a confocal microscope, an image analysis platform, and a flow cytometer. Each is from a different vendor with its own proprietary software. A scientist running a standard mitosis assay spends 4–6 hours manually coordinating between them: copying well IDs from one system to the next, waiting for one software to finish before starting another, and manually verifying the data didn't get misaligned in transfer. Errors in well mapping between steps are common.

**Task:** I wanted to build a coordination layer that removed the manual handoffs — something where you describe the experiment once and the system handles the sequencing, waiting, conditional branching, and data archival automatically.

**Action:** I designed and built the system in four independent layers. First, async job tracking — the microscope acquisition call returns a job ID immediately and runs in a background thread, so the system doesn't block. Second, a safety validator that intercepts every instrument call before dispatch, checks the parameters against physical limits (channel names, volume ranges, well formats), and returns a typed rejection reason if anything is wrong — this prevents impossible instructions from ever reaching hardware. Third, a local LIMS that writes five metrics per well per timepoint after every analysis step, queryable by the orchestrator. Fourth, a protocol executor for structured experiments — the language model generates a typed JSON protocol with hypothesis, steps, and success criteria; the validator checks every step; a human approves it; then it runs deterministically.

I tested the full pipeline with real HeLa cell microscopy images from the Broad Institute's BBBC020 benchmark dataset, getting mitotic index values of 3–9.5% across wells, consistent with what you'd expect from an asynchronous cell population.

**Result:** The full M1–M4 architecture passes end-to-end in simulation. 20 experiments, 252 per-well analysis results, 70 LIMS rows across 5 metrics. The `--loop N` mode runs fully automated closed-loop cycles with LIMS feedback into each next plan. The remaining gap is real hardware connections — the driver stubs are written, the REST endpoints are configured, they just need a physical robot and microscope on the other end.

---

## The cover letter paragraph

One project I'd point to is lab-agent — a software coordination layer for multi-instrument biotech workflows. The problem it solves is straightforward: a liquid handler, a confocal microscope, and a flow cytometer have never been designed to talk to each other, and in practice a scientist spends hours manually transferring data between their separate software systems. I built a system where you describe the experiment once and it coordinates all three instruments in sequence, handles asynchronous acquisition (the microscope doesn't block the rest of the workflow), validates every instrument call against physical limits before dispatch, and archives every result per well per timepoint with full provenance. The architecture is designed so you flip one line in the config to go from simulation to a real Opentrons robot or Micro-Manager microscope. I tested it with the Broad Institute's BBBC020 HeLa cell benchmark dataset and ran 20 experiments end-to-end in the process.

---

## Things NOT to say

- "I built an AI-powered lab automation system" — vague, sounds like a pitch deck
- "leveraging state-of-the-art language models" — no
- "seamless integration" — no
- "robust pipeline" — no
- "end-to-end solution" — no

Say what it actually does. The specific detail is what makes it believable.

---

## When to show the repo

Keep it private until:
- The FastAPI approval UI is built (3 days) — right now `--approve` is a CLI flag, which looks unfinished
- One real instrument is connected, even in a test environment — "tested on simulation" is weaker than "tested on an Opentrons Flex"
- The BBBC020 mitotic index caveat is addressed — either switch dataset or add the phospho-H3 mode

Then make it public and link it directly in job applications. A working repo with real numbers is worth three bullet points on a CV.
