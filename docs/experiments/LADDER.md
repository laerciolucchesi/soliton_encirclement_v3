# The integration ladder, regenerated on committed code

**Why.** The ladder (baseline → A → B → B2) is the thesis's central table. Its published
numbers come from `optionB_results.csv` and `gain_results.csv`, produced 2026-05-30 in an
**uncommitted tree**. `DUAL_PULSE_DELTA_SCALE` and `DUAL_PULSE_INTEGRATION` only entered the
versioned config in **`ff54ade` (2026-06-07)** — the same commit that first committed those
CSVs. There is no versioned config state corresponding to the runs that produced them.

**What was done.** A new runner (`run_ladder.py`; the old ones untouched) re-runs the whole
ladder with every relevant parameter fixed in the child environment and provenance on every
row: 6 variants × N ∈ {24, 40, 50} × 3 seeds × 2 control periods = **108 runs, zero failures,
every row `dirty=False`**.

```powershell
python experiments/scaling_law/run_ladder.py      # CONTROL_PERIOD / LADDER_NS / LADDER_SEEDS / LADDER_VARIANTS / LADDER_TAG
python experiments/scaling_law/analyze_ladder.py  # tables, criterion histogram, dt-invariance, figure
```

Outputs: `ladder_results_*.csv` (12 files), `ladder_summary.csv`, `figures/fig_ladder.png`.
Provenance: 47 rows at `f71600a`, 61 at `5701c18`; the two commits differ only by an analysis
script and result CSVs — the simulation code is byte-identical between them.

## 0. The historical table was run at dt = 0.01 — so the *replica* is the reproduction test

Neither `run_optionB_test.py` nor `run_gain_sweep.py` pins `CONTROL_PERIOD`; both inherited the
default, which was **0.01** until the Ciclo 1 campaign changed it to 0.05 (`config_param.py:22-26`).
`figure_data.csv` confirms it (`dt_telem ≈ 0.01`). So the dt = 0.01 grid is the reproduction
attempt and the dt = 0.05 grid is the dt-invariance test — not the other way round.

**On `METRICS_T0`:** the historical τ values were **not** measured at t0 = 0. `METRICS_T0` feeds
exactly one place in the codebase (`plot_telemetry.py:802` → M1..M7 in `runs_summary.csv`), which
the ladder never used. Every τ in the campaign comes from `event_metrics(df, T0)` — or each old
runner's local copy of the same algebra — with `T0 = 5.0` passed explicitly. It is pinned here
for hygiene; it changes nothing.

## 1. The ladder — dt = 0.01 (the reproduction test)

median [min–max] over 3 seeds, τ in s.

| variant | N=24 | N=40 | N=50 |
|---|---|---|---|
| baseline, fixed gain 25 | 7.08 [7.1–7.1] | 12.26 [12.2–12.3] | **1949** [201–6862] |
| baseline, stable gain 250/N | 19.49 [19.5–19.5] | 54.73 [54.7–54.8] | 85.41 [85.3–85.5] |
| Option A, scale 0.5 | 11.67 [11.6–11.7] | 41.86 [41.8–42.0] | 74.84 [74.7–75.1] |
| **Option B-min, scale 0.5** | **2.30** [2.3–2.3] | **2.62** [2.6–2.6] | **2.75** [2.7–2.8] |
| Option B-min, scale 1.0 | 20.04 [20.0–20.1] | 53.71 [53.7–53.8] | 81.57 [81.6–81.9] |
| Option B2, scale 1.0 | 2.22 [2.2–2.2] | 3.07 [3.0–3.1] | 4.06 [4.0–4.1] |

Against the published values (`tese_estrutura.md:55-60`; `*` = deviation > 10%):

| variant | N=24 | N=40 | N=50 |
|---|---|---|---|
| baseline, fixed gain | 7.08 → 7.08 (+0%) | 12.26 → 12.26 (−0%) | 140.1 → 1949 (**+1291%**)\* |
| baseline, stable gain | 19.48 → 19.49 (+0%) | 54.79 → 54.73 (−0%) | 85.35 → 85.41 (+0%) |
| Option A | 11.63 → 11.67 (+0%) | 42.02 → 41.86 (−0%) | 74.71 → 74.84 (+0%) |
| Option B-min @0.5 | 3.27 → 2.30 (**−30%**)\* | 7.78 → 2.62 (**−66%**)\* | 12.20 → 2.75 (**−77%**)\* |
| Option B-min @1.0 | 16.51 → 20.04 (**+21%**)\* | 43.00 → 53.71 (**+25%**)\* | 62.59 → 81.57 (**+30%**)\* |
| Option B2 @1.0 | 2.17 → 2.22 (+2%) | 2.13 → 3.07 (**+44%**)\* | 2.12 → 4.06 (**+92%**)\* |

**Where the parameters were knowable, reproduction is exact.** The two baselines and Option A
land within 0.4% of the published values at all three N — nine cells, three digits. That
validates the runner, the metric path and the dt attribution simultaneously.

**Where the parameters were unknowable, nothing reproduces.** The two rows that fail are exactly
the two Option-B rows, whose `DELTA_SCALE` could not be recovered. And the failure is not a
matter of picking the other scale: the published 3.27 / 7.78 / 12.20 sits between the two scales
at N=24 and outside both by N=50, and its *shape* matches neither — it grows by 3.7× from N=24
to N=50 while the measured @0.5 grows by 1.20× and @1.0 by 4.07×.

## 2. The ladder — dt = 0.05 (the dt-invariance test)

| variant | N=24 | N=40 | N=50 |
|---|---|---|---|
| baseline, fixed gain 25 | 7.83 [7.8–7.9] | 12.33 [12.3–12.4] | **17.19** [17.1–17.4] |
| baseline, stable gain 250/N | 20.29 [20.2–20.3] | 55.55 [55.5–55.6] | 88.26 [88.0–88.4] |
| Option A, scale 0.5 | 13.50 [13.5–13.6] | 42.32 [42.3–42.4] | 77.71 [77.5–78.7] |
| **Option B-min, scale 0.5** | **2.37** [2.3–2.4] | **2.69** [2.7–2.7] | **2.91** [2.9–2.9] |
| Option B-min, scale 1.0 | 19.54 [19.5–19.5] | 53.54 [53.4–53.7] | 79.13 [78.4–79.2] |
| Option B2, scale 1.0 | 2.14 [2.1–2.2] | 2.68 [2.7–2.8] | 3.20 [3.1–3.3] |

Seed spread is ≤ 1% everywhere. With an equidistant ring and a deterministic fault the seed only
selects which node dies, and the ring is symmetric — **the relevant uncertainty in this grid is
dt, not seed.**

## 3. The stability criterion does not separate cleanly

As specified: `settled = egap_final < 1e-2 AND egap_late_std < 1e-3 AND tau_fit_r2 > 0.80`.
**Applied unchanged**; the diagnosis below is reported rather than acted on.

| variant | settled | fails at |
|---|---:|---|
| baseline, fixed gain | 15/18 | N50/dt0.01 |
| baseline, stable gain | 12/18 | N24, both dt |
| **Option A** | **0/18** | every N, both dt |
| Option B-min @0.5 | 18/18 | — |
| Option B-min @1.0 | 12/18 | N24, both dt |
| Option B2 @1.0 | 18/18 | — |

Two distinct pathologies, neither of which is instability:

* **Option A fails all 18 on `tau_fit_r2` (0.737–0.778), while its `egap_late_std` is 0.00065 —
  below the threshold.** It settles; what it does not do is decay as a clean exponential. R²
  measures the *shape* of the decay, not stability, so including it in a stability criterion
  makes "not settled" mean "not exponential" for this variant.
* **The N=24 failures of the two slow variants are a budget artefact.** With τ ≈ 20 s and
  budget = 67 s, the run ends at 3.3 τ, so the last-20 s `egap_late_std` (0.0015) still reflects
  ongoing decay, not oscillation. `egap_final` is 0.0044 — comfortably inside its own threshold.

Boundary cases are dense: 9 runs within a factor 3 of the `egap_final` threshold and 18 within a
factor 3 of the `egap_late_std` threshold. The criterion is not sharp here.

**Recommendation, not applied:** move `tau_fit_r2` out of the stability criterion into a separate
fit-quality column, and either scale the budget to ≥ 6 τ or measure `late_std` over a window
defined relative to τ rather than a fixed 20 s. Both changes move rows between "settled" and
"not settled", so they are the author's call.

## 4. The three closing questions

### (i) Which scale does the proposal's "Option B" actually use?

**Unknowable from the artifacts, and not recoverable by trying both.**

`run_optionB_test.py` never sets `DUAL_PULSE_DELTA_SCALE`; `optionB_results.csv` has no such
column; the constant was not versioned when the run happened. Under *today's* config, re-running
that script unchanged would produce **scale 1.0**, not 0.5 — the default is mode-dependent
(`_DPS_DEFAULT = 1.0` for B/B2, `config_param.py:553`), so the script and the published label
now disagree.

The empirical fingerprint does not settle it either:

| N | published "B" | measured @0.5 | measured @1.0 |
|---:|---:|---:|---:|
| 24 | 3.27 | 2.30 | 20.04 |
| 40 | 7.78 | 2.62 | 53.71 |
| 50 | 12.20 | 2.75 | 81.57 |

At N=24 the published value is nearest @0.5 (42% above it, 6× below @1.0), but by N=50 it is
4.4× above @0.5 and 6.7× below @1.0 — outside both. **The published Option-B row is not
reproducible at either scale.**

Note also that **the published table omits one of the six rows**: `5-preliminary-results.tex:23-33`
lists five, dropping "Option B-min, scale 1.0" (16.51 / 43.00 / 62.59) — precisely the row that
makes the double-drive narrative falsifiable.

### (ii) Does the double-drive narrative hold?

**Its structure holds, far more strongly than the published numbers show — but its conclusion
does not.**

At dt = 0.01:

| N | B-min@0.5 | B-min@1.0 | B2@1.0 | B@1.0 / B@0.5 | B@1.0 / B2 |
|---:|---:|---:|---:|---:|---:|
| 24 | 2.30 | 20.04 | 2.22 | 8.7× | 9.0× |
| 40 | 2.62 | 53.71 | 3.07 | 20.5× | 17.5× |
| 50 | 2.75 | 81.57 | 4.06 | 29.7× | 20.1× |

Raising the scale to 1.0 under the *minimal* cancelling bias is catastrophic and gets worse with
N — 8.7× → 29.7×, against 5.0× in the published table. The mechanism the thesis proposes
(minimal bias + full scale = double-drive over-drive) is strongly confirmed.

**But the conclusion drawn from it is not.** The claim is that only the *complete* cancelling
bias escapes. Measured, **B-min@0.5 is both flatter and faster than B2 at N ≥ 40**:

| | N=24 → N=50 growth | τ at N=50 |
|---|---:|---:|
| B-min @0.5 | **1.20×** | **2.75 s** |
| B2 @1.0 | 1.83× | 4.06 s |

Same ordering at dt = 0.05 (1.23× vs 1.50×). So the halved-scale minimal bias is not a stepping
stone that B2 supersedes — on this grid it is the better of the two at large N. The published
row for it (3.27 / 7.78 / 12.20, growing 3.7×) is what made it look like a dead end, and that row
does not reproduce.

### (iii) Is τ invariant in dt?

**Yes for 16 of 18 cells; the two exceptions are the interesting ones.** CV between dt = 0.01
and 0.05:

| variant | N=24 | N=40 | N=50 |
|---|---:|---:|---:|
| baseline, fixed gain | 5.0% | 0.3% | **98.3%** ← |
| baseline, stable gain | 2.0% | 0.7% | 1.6% |
| Option A | 7.3% | 0.5% | 1.9% |
| Option B-min @0.5 | 1.6% | 1.2% | 2.8% |
| Option B-min @1.0 | 1.3% | 0.2% | 1.5% |
| Option B2 @1.0 | 1.7% | 6.8% | **11.9%** ← |

* **The fixed-gain instability at N=50 is a dt = 0.01 phenomenon.** At dt = 0.01 the fit is
  divergent — 201 / 1949 / 6862 across three seeds, R² near zero, 0/3 settled — which is what an
  unstable signal does to an exponential fit, and confirms the *qualitative* claim while showing
  the published number 140.1 is an artifact of fitting one seed of a non-decaying signal. At
  dt = 0.05 the same configuration is **stable**: 17.19 s [17.1–17.4], 3/3 settled. The
  campaign moved the default to dt = 0.05 on the strength of dt-invariance; the Cap. 3
  instability claim does not survive that move.
* **B2 drifts with dt at large N** (11.9% at N=50, 6.8% at N=40), and the drift grows with N —
  the only variant whose dt-invariance degrades systematically.

## 5. The B2 discrepancy, bounded

The regenerated B2 gives 4.06 s at N=50 / dt=0.01. `largeN_results.csv` — itself committed, and
nominally the same configuration — reports **2.115** (two seeds, 2.115 and 2.116);
`figure_data.csv` agrees. At N=24 the two agree (2.22 vs 2.17); the divergence grows with N.

Ruled out by direct test:

| hypothesis | test | result |
|---|---|---|
| fit window (largeN used a 90 s budget, this grid 289 s) | recompute τ on truncated windows of the same telemetry | τ identical for budgets 30–289 s — the 5%-of-peak floor makes it window-independent |
| M8 (`DUAL_PULSE_CONSUME_FF_ONLY`, default-ON since Phase 3) | ablate at N=50 | **τ = 285.6, R² = 0.14** with M8 off — M8 is *essential* to B2 at N=50, not the cause |
| M-mult (`DUAL_PULSE_MULTIPLICITY`, default-ON since Ciclo 2) | ablate | 3.10 vs 3.10 — no effect, as documented for k=1 |
| hop attenuation (`ALPHA_CLOSE_RATIO`) | set to 1.0 | 3.07 vs 3.10 — no effect |
| ramp (`RAMP_TICKS`) | set to 1 | 3.17 vs 3.10 — no effect |
| dual_pulse knobs differing between runners | read `run_largeN_confirm.py:105-111` | it sets the **same four** (B2, 1.0, 1.0, 3N) |
| `AGENT_STATE_TIMEOUT` (the one knob this runner pins and the old one inherits) | 5·dt = 0.25 vs today's default 1.0 vs 0.2 | **3.10 / 3.13 / 3.11 — no effect** |

Every parameter difference that can still be tested is eliminated. What is left is **drift in the
simulation code itself** between May 2026 and today — the `dual_pulse` layer has since gained the
premise-clean trigger refactor, M8 and M-mult, and the B/B2 bias path was reworked. The May run
cannot be re-executed to check, because the tree that produced it was never committed.

**That is the P2 premise demonstrating itself.** An unpinned runner's numbers depend on code and
defaults that have since moved; once they move, the numbers cannot be reproduced *or* diagnosed.
The regenerated ladder is reproducible by construction — every row carries its commit and its
pinned parameters — which is why the same question will be answerable next time.

## 6. What this changes in the thesis

**Draft v1, `5-preliminary-results.tex:23-33` (Table `tab:scaling`).** Four of its five rows are
confirmed to three digits. The "Feedforward (B), scale 0,5" row is not reproducible and its
provenance is unrecoverable; the sixth row (B-min @1.0) is missing and should be restored,
because it is the evidence for the double-drive mechanism the surrounding text argues.

**The caption's claim "only B2 is flat" does not survive.** B2 is not flat on this grid
(2.22 → 4.06, dt = 0.01) and B-min @0.5 is flatter (2.30 → 2.75). Either the caption changes, or
the B2 discrepancy of §5 is resolved first — it is the same question.

**`4-proposal.tex:189` ("The complete cancelling bias zeroes the double-drive") keeps its
mechanism and loses its exclusivity.** The 8.7× → 29.7× penalty of the minimal bias at full
scale is confirmed and strengthened; what is not supported is that the complete bias is the
only way out.

**Cap. 3 / `1-introduction.tex` instability claim needs a dt qualifier.** "unstable beyond
N ≈ 40" holds at dt = 0.01 and fails at dt = 0.05, which is the current default. Either state
the dt, or re-examine whether the dt = 0.05 default is safe for the fixed-gain regime.

## 7. Caveats

* Three N points and three seeds; the seed spread is ≤ 1% because the scenario is symmetric, so
  the error bars here are *not* an estimate of scenario variability.
* Single permanent fault, ideal comms, static target, uniform ring — the clean regime only.
* τ is an exponential fit; for Option A (R² ≈ 0.75) and the unstable fixed-gain N=50 cell
  (R² ≈ 0) it should not be read as a time constant at all.
* §5 bounds the B2 discrepancy but does not close it. Until it is closed, the B2 row of this
  table and the B2 row of `largeN_results.csv` disagree, and this document does not assert which
  is right — only that they were produced by runners that differ in what they pin.

## Related

* [PROVENANCE.md](PROVENANCE.md) — the schema every row here carries.
* [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) — the dated entry.
* [README.md](README.md) — metric definitions and the reporting rule.
