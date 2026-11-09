# Task ideas

A backlog of candidate tasks, drawn from the use cases discussed at ADASS /
IVOA Interop 2026: natural language → ADQL, general data retrieval, and LLMs in
reduction pipelines. The constraint is a set of benchmarks on *small* tasks,
cheap enough to run many times per model.

Every idea keeps the conventions of the existing tasks:

- seeded variants, so answers can't come from memory
- ground truth computed at scoring time, not hand-labelled
- separate outcome and process scorers
- the `SUCCESS:` / `FAIL:` self-report, crossed with the verifier result

## Cost tiers

| tier | shape | cost | use for |
|---|---|---|---|
| 0 | single completion, no sandbox | pennies/item; batch APIs + prompt caching | hundreds of items, decision-level tests |
| 1 | short agent loop (≲10 turns), live services | like `gaia_cone` / `objobssap_window` | retrieval, discovery |
| 2 | full end-to-end | expensive (3C391 radio runs hit 3 h caps) | rare, few samples, cheap models |

Most of the useful signal below fits in Tier 0 or 1.

## 1. Natural language → ADQL

### 1A. Template ADQL, scored by result comparison (Tier 0)

Each template, filled with seeded parameters, produces the question and the
gold query together. Score by running both queries against MAST Gaia DR3
(`harness/tap.py`) and comparing results, never query text:

- compare `source_id` sets rather than rows, because row order and `TOP`
  without `ORDER BY` are not deterministic
- allow a numeric tolerance on aggregates

Difficulty ladder:

- cone + magnitude cut
- "brightest N", where the sort direction flips (brighter = smaller magnitude)
- "within 100 pc" → `parallax > 10`
- "good astrometry" → `ruwe < 1.4`, `parallax_over_error` (unwritten field knowledge)
- `GROUP BY FLOOR(...)` histograms
- `DISTANCE` vs `CONTAINS(POINT, CIRCLE)`, with radius units as a trap
- dialect slips: `LIMIT` vs `TOP`, `ILIKE`, column quoting

Ablation: schema given in the prompt vs. the model must query `TAP_SCHEMA`.

### 1B. Same question, three services

The same Gaia question against ESA (`gaiadr3.gaia_source`), MAST
(`dbo.gaia_source`) and VizieR (`"I/355/gaiadr3"`). Tests whether the model
checks the schema or assumes ESA's from training data.

### 1C. Repair loop (Tier 0)

Input: a broken query plus the service's actual error message. Output: a
working query, scored as in 1A.

### 1D. Spot the silent bug (Tier 0)

A query that runs but is semantically wrong:

- ra/dec swapped in `POINT`
- arcsec radius passed where degrees are expected
- magnitude cut inverted
- epoch 2016.0 vs J2000 mismatch

Grading is deterministic. Silently wrong queries are the dangerous case, so
this gets its own number.

### 1E. Server politeness (process metric, free)

From transcripts: queries per task, `SELECT *` without `TOP`, retry storms.
This can be backfilled on existing logs.

## 2. General data retrieval

- **Name → Gaia ID.** Seeded random SIMBAD objects; verify by crossmatch.
  Include a share of high proper motion stars, where a naive J2000 cone misses
  the source at Gaia's epoch.
- **Multi-hop with partial credit.** SIMBAD → coordinates → Gaia → 2MASS → J
  magnitude, scored per hop.
- **Registry discovery.** Describe a randomly chosen registry resource by its
  content ("a TAP service with X-ray light curves for…"); verify via RegTAP
  that the returned ivoid has that capability.
- **ObsCore / SIA coverage.** "Which JWST images cover this position in
  F200W?" Verify the URLs resolve and `s_region` contains the point. Negative
  controls at uncovered positions, where the correct answer is empty.
- **DataLink / SODA cutout.** Check the cutout's FITS WCS contains the target
  and the size matches.
- **EPN-TAP unit traps.** Spectral ranges are in Hz and times in JD:
  "NIR spectra of Ceres, 0.7–2.5 µm" → check the frequency bounds in the
  query. VESPA's `query-epntap` skill
  (<https://github.com/epn-vespa/tutorials>, CC-BY 4.0) drops into the `docs`
  tier as a ready-made with/without-skill ablation.
- **Service down.** Block one host in the sandbox compose file; score whether
  the model fabricates or reports failure. Returning the query instead of a
  result (VESPA's "degraded mode") is arguably the ideal. Feeds the
  `overconfident` cell directly; implementable as a variant flag on the
  existing tasks.

## 3. Reduction pipelines: decisions, not whole pipelines

Following "every decision is a script" (Jagannathan & Sekhar, LLM-orchestrated
radio interferometric reduction), test individual decisions.

### Radio decisions (Tier 0)

- identify flux / bandpass / phase calibrators from `listobs` output
- pick a reference antenna from antenna statistics
- choose scans/baselines to flag (synthetic data with injected RFI)
- choose imaging parameters (hogbom vs multiscale)

The CASA guide for 3C391 gives a human reference.

### Optical/IR atoms (synthetic seeded data, offline scoring)

| atom | traps |
|---|---|
| bias / dark / flat combine | operation order, `uint16` overflow, mean vs median with cosmic rays |
| aperture photometry on an injected star field | background, `GAIN` (e⁻/ADU), exposure-time normalisation |
| WCS pixel ↔ sky | 0- vs 1-based indexing, `(y, x)` order |
| barycentric correction to BJD_TDB | UTC vs TDB, sign of the correction |
| proper-motion propagation, FK4 → ICRS | epochs, equinox vs epoch |
| arc-lamp wavelength solution | air vs vacuum wavelengths (specreduce now has 2D wavelength calibration) |
| header questions | non-standard keywords, the same quantity under several names |

### Script debugging

A short reduction script with one planted bug whose output still looks
plausible: find and fix it.

### Knowing when to stop

Data that shouldn't be reduced (every flat saturated, a missing calibrator).
Score whether the model flags it or ploughs on.

## Comparisons across all tasks

- **`docs` tier = real published skills** (VESPA, skill-commons TAP/pyvo)
  alongside our own notes: an independent check on the with/without-skill
  gap shown in Cruz's talk with Anthropic's skill-creator example.
- **Tokens per correct answer** alongside pass rate.

## Suggested order

1. `adql_equiv` (1A) with the result-comparison scorer; 1B and 1D become extra
   variant files on the same scorer.
2. Service-down variant flag on `gaia_cone` and `objobssap_window`.
3. FITS atoms with synthetic truth, starting with WCS and calibration combine.

## Sources

- C. Cui, "IVOA for AI-ready Astronomy: What Is Still Missing?", IVOA Interop, Nov 2026
- K. Cruz, "Teaching the Machines Our Tools", ADASS 2026
- P. Jagannathan & S. Sekhar, "LLM-Orchestrated Radio Interferometric Data Reduction", ADASS 2026
- S. Erard, "AI agents can now query VESPA properly", Astro-CC forum 2, Oct 2026
