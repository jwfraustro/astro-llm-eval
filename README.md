# astro-llm-eval

Benchmarking LLM agents on astronomy data-access tasks: can a model, given a
shell, Python, and (optionally) web search, correctly fetch catalog data via
TAP / astroquery / pyvo?

The first task is deliberately minimal: **fetch 10 Gaia DR3 sources inside a
random cone on the sky**. It is cheap to verify against a live catalog and
essentially impossible to fake. The repo is structured so that more tasks drop
in beside it.

## How it works

```
control box (CPU)                                    GPU / API
┌──────────────────────────────────────────────┐
│ inspect eval                                 │
│   ├─ per-sample docker sandbox ───────────── │ ──► model API or Runpod vLLM
│   │    (pinned astropy/astroquery/pyvo,      │
│   │     network on, no credentials)          │
│   └─ scorer (trusted, outside the sandbox)   │
│        read /workspace/result.csv            │
│        └─► MAST Gaia DR3 TAP: ids exist?     │
│            inside the cone? coords match?    │
│ logs/*.eval  (full transcripts)              │
└──────────────────────────────────────────────┘
```

Built on [Inspect AI](https://inspect.aisi.org.uk/), which supplies the
tool-calling agent loop, docker sandboxing, model providers, logging and a
transcript viewer. This repo adds the astronomy: tasks, prompts, variant
generation, the verifier, and a failure taxonomy.

### Why a random cone?

A bare "fetch 10 rows of Gaia DR3" can in principle be answered from memory
(real `source_id`s appear in papers). Pinning the request to a seeded random
cone removes that: the verifier checks that every id exists in Gaia DR3 *and*
that its catalog position lies inside the cone *and* that the coordinates the
model wrote match the catalog. One template → unlimited trials.

### Tool tiers (the main ablation)

| tier     | tools                 | measures                                |
|----------|-----------------------|-----------------------------------------|
| `none`   | bash, python          | recall of astro infrastructure          |
| `search` | bash, python, web     | ability to look things up               |
| `docs`   | bash, python + notes  | ability to apply good documentation     |

The gap between `none` and `search` is probably the number people actually
want: does the model *know* the archive landscape, or can it only find it?

### What gets recorded

- `tap_verify` scorer: `CORRECT` / `INCORRECT` / `NOANSWER` (verifier
  infrastructure error, never blamed on the model), with a mechanical failure
  reason: `no_output_file`, `missing_columns`, `too_few_rows`,
  `source_id_not_integer`, `duplicate_source_id`, `unknown_source_id`
  (fabricated), `outside_cone`, `coordinates_mismatch`.
- Self-report calibration: the prompt asks the model to prefix its submit
  message with `SUCCESS:` or `FAIL:`. Crossed with the verifier this gives
  `true_positive` / `overconfident` / `underconfident` / `honest_failure`.
  *Overconfident* (claimed success, verifiably wrong) is the cell to watch.
- `failure_taxonomy` (optional, LLM-judged): *how* it failed —
  hallucinated endpoint, wrong table/column, ADQL mistakes, library misuse,
  gave up, fabricated output, ...
- Per-sample message count and token usage from the Inspect log.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/pytest -m "not live"       # offline verifier tests
```

```bash
.venv/bin/pytest -m live             # three small queries against MAST
```

Run one model, five trials per variant, search tier (needs Docker and
`TAVILY_API_KEY` or another provider via `ASTRO_EVAL_SEARCH_PROVIDER`):

```bash
scripts/run.sh anthropic/claude-sonnet-5-5 search 5 --limit 5
```

Then:

```bash
.venv/bin/inspect view                       # browse transcripts
```

```bash
.venv/bin/python scripts/summarize.py logs/  # markdown results table
```

Open-weight models: see [`infra/runpod/README.md`](infra/runpod/README.md).

## Repo layout

```
sandbox/            Dockerfile + compose.yaml for the agent container (pinned)
harness/            trusted-side code: MAST TAP verifier client, geometry, LLM judge
tasks/gaia_cone/    task.py (Inspect task), prompts.py, variants.py/.jsonl,
                    verify.py (pure logic), scorer.py, docs/ (for the docs tier)
configs/models.yaml model registry; edit ids before a run
infra/runpod/       vLLM launch script and tool-parser notes
scripts/            run.sh, summarize.py, rescore.py
tests/              offline verifier tests + opt-in live MAST tests
```

## Verification details

Verifier endpoint: `https://mast.stsci.edu/vo-tap/api/v0.1/gaiadr3`, table
`dbo.gaia_source`. Ids are looked up with `WHERE source_id IN (...)` in chunks
of 100 with ≥0.5 s spacing and retries on 5xx; a 10-row result is a single
request. `source_id` is parsed as a literal integer — a float-formatted id
(`3.7e18`) fails before any network call, because an id that went through a
double is not the id the model fetched.

If MAST is unreachable mid-run, affected samples score `NOANSWER` with the CSV
text stashed in the score metadata; `scripts/rescore.py` re-verifies them later
without re-running the model. `scripts/run.sh` refuses to start if the
healthcheck fails.

## Reproducibility notes

- Pin everything in the sandbox image, and bump pins in their own commit.
- Record model snapshot ids, and for vLLM the version, parser, and quantisation.
- Live services change. Keep the `.eval` logs and the run date with results.
- Use ≥5–10 epochs per variant; `summarize.py` reports Wilson 95% intervals
  because the counts will be small.
- With `--epochs > 1`, Inspect reduces epochs *before* computing metrics and
  keeps only the first epoch's score metadata, so the histogram metrics shown
  at the end of `inspect eval` describe epoch 0. `accuracy` is exact;
  `summarize.py` reads every epoch and is the source of truth for the rest.

## Roadmap

Next task rungs, in rough order of difficulty: Gaia × 2MASS crossmatch, "list
HST observations of target X" via MAST, SIMBAD name resolution → cone search,
service discovery through the pyvo registry.
