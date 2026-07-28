# τ_B2 vs N and vs dt: arbitrating the headline

**Why.** The thesis headline is "reconfiguration time **flat in N** (~2.1 s up to N=100)", from
`largeN_results.csv`. P2 left two committed and incompatible measurements of the same cell —
B2 at N=50/dt=0.01: **4.06 s** (ladder, committed and tested tree, everything pinned) versus
**2.115 s** (largeN, May tree never committed). That cell decides whether Lei 1 is ~N^1.9 or
~N^1.1, and whether the N=100 advantage is ~148× or ~43×.

**Answer, in one line.** The ladder value is confirmed exactly (4.060 s). The published metric
`tau_fit` then grows as N^1.25, which would refute the flat-τ claim — **except that the growth is
an artefact of the exponential fit**, and the campaign's own *primary* metric `t_settle` is flat
(6.97 → 8.07 s from N=24 to N=100 at dt=0.01). Dissemination is never the bottleneck: it is
5–11 % of τ and its share *falls* with N.

Pre-registration: [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md), 2026-07-27, written and committed
(`83a53ca`) before the grid ran. Divergences in §7.

```powershell
python experiments/scaling_law/run_dt_scaling.py        # DTS_GRIDS / DTS_NS / DTS_DTS / DTS_SEEDS
python experiments/scaling_law/analyze_dt_crossover.py  # tables + figure
```
46 runs, zero failures, every row `dirty=False`. Outputs: `dt_scaling_results_*.csv`,
`dt_crossover_summary.csv`, `figures/fig_dt_crossover.png`.

**Victim rule** (written here as required): `victim = 2 + ((N//2 + seed) % N)` — agents occupy
ids 2..N+1, so this is the node diametrically opposite id 2, rotated by the seed. With an
equidistant ring the seed is a symmetry check, not a source of scenario variability.

---

## 0. Arbitration — the N=50 / dt=0.01 cell

| source | τ | seeds | tree |
|---|---:|---:|---|
| **this grid** | **4.060 s** [4.049, 4.069] | 3 | committed, all pinned |
| P2 ladder | 4.060 s | 3 | committed, all pinned |
| `largeN_results.csv` | 2.115 s | 2 | May 2026, **never committed** |

**Deviation from the ladder: −0.0 %. From largeN: +92.0 %.**

Two independent runners, written weeks apart, with independently specified environments, land on
the same three decimals. `largeN_results.csv` does not reproduce.

**Decision: the ladder/P3 value stands as the record; `largeN_results.csv`'s B2 rows are
superseded.** P2 §5 had already eliminated every testable cause of the discrepancy (fit window,
M8, M-mult, hop-alpha, ramp, `AGENT_STATE_TIMEOUT`, and the four dual_pulse knobs, which the old
runner sets identically); what remains is code drift in a tree that cannot be checked because it
was never committed. Per campaign rule 2 the old file is not deleted — it is superseded, and this
is the log entry saying so.

---

## 1. The exponent — and why the published metric misleads at large N

### 1a. On `tau_fit`, the metric the thesis publishes

| N | dt=0.01 | R² | dt=0.05 | R² |
|---:|---:|---:|---:|---:|
| 24 | 2.218 | 0.969 | 2.143 | 0.976 |
| 32 | 2.579 | 0.955 | 2.336 | 0.975 |
| 40 | 3.074 | 0.930 | 2.680 | 0.970 |
| 50 | 4.060 | 0.871 | 3.199 | 0.950 |
| 75 | 7.273 | **0.737** | 4.644 | 0.907 |
| 100 | 12.895 | **0.631** | 9.966 | **0.753** |

Fits: `τ ~ 0.035·N^1.246` (dt=0.01, R²=0.958) and `τ ~ 0.069·N^1.019` (dt=0.05, R²=0.888).

Taken at face value this refutes flatness and rewrites Lei 1 as `A ~ N^(1.94−1.25) = N^0.69`,
giving A(100) = 29× instead of 148×.

**But the R² column disqualifies it.** The campaign's own rule
([README.md](README.md):63) is *"tau_fit — secondary; only trust R² ≥ 0.9"*. At N=75 and N=100
the fit fails that bar in the dt=0.01 arm, and at N=100 in both.

### 1b. Why the fit fails — mechanically, not mysteriously

| N (dt=0.01) | `egap_peak` | 5 %·peak (fit floor) | `egap_final` | final / floor | R² |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.1956 | 0.00978 | 0.00019 | **0.02** | 0.969 |
| 40 | 0.1544 | 0.00772 | 0.00148 | 0.19 | 0.930 |
| 50 | 0.1385 | 0.00693 | 0.00213 | 0.31 | 0.871 |
| 75 | 0.1139 | 0.00569 | 0.00251 | 0.44 | 0.737 |
| 100 | 0.0993 | 0.00497 | 0.00281 | **0.57** | 0.631 |

Two things move in opposite directions as N grows. The **peak falls** — `E_gap` is an RMS across
the ring, so one death perturbs a larger ring less. The **residual floor rises**. `exp_tau` fits
only where `e > 0.05·peak`, so the window's lower edge descends toward a residual that is
climbing to meet it: at N=100 the flat tail sits at 57 % of the fit floor. The exponential is
then fitted largely to a plateau, which inflates τ and collapses R² — in lockstep, as the table
shows.

**`tau_fit`'s growth with N is a property of the metric, not of the algorithm.**

### 1c. On `t_settle`, the campaign's primary metric

`t_settle` is defined ([metrics_util.py](../../experiments/scaling_law/metrics_util.py):43-67) as
enter-and-**stay** settling time and is explicitly *"robust to the shape of the decay; does not
require an exponential"*.

| N | dt=0.01 | dt=0.05 |
|---:|---:|---:|
| 24 | 6.97 | 7.30 |
| 32 | 7.75 | 7.70 |
| 40 | 8.06 | 8.20 |
| 50 | 8.00 | 8.25 |
| 75 | 8.01 | 9.50 |
| 100 | 8.07 | 10.40 |

Fits: `N^{+0.079}` (dt=0.01, R²=0.55 — i.e. **no trend**) and `N^{+0.246}` (dt=0.05, R²=0.97).

**On the robust metric the flat-τ claim survives at dt=0.01 and degrades mildly at dt=0.05
(+42 % from N=24 to N=100).**

### 1d. Lei 1, computed both ways, on the same code

Grid B measured the baseline at N=50 in this same tree: `tau_fit` 85.38 s (dt=0.01) / 88.11 s
(dt=0.05); `t_settle` 128.06 / 130.60 s. The dt=0.01 value reproduces the historical 85.35 s
exactly, as the baselines did in P2.

| metric | advantage at N=50, dt=0.01 | dt=0.05 |
|---|---:|---:|
| `tau_fit` | 21.0× | 27.5× |
| `t_settle` | 16.0× | 15.8× |

Note the two metrics agree far better for `t_settle` across dt (16.0 vs 15.8) than `tau_fit`
does (21.0 vs 27.5) — another symptom of the fit instability.

Extrapolating to N=100 with the baseline's own law (`τ_base ≈ 0.0417·N^1.94` → 316 s):

* on `tau_fit`: A(100) ≈ 316 / 11.0 ≈ **29×** (dt=0.01), 316 / 7.6 ≈ **42×** (dt=0.05)
* on `t_settle`: τ_B2 is flat at ~8 s, so A(100) grows as N^1.94 — the published shape survives,
  with a different constant.

---

## 2. dt-invariance, and grid C

| N | `tau_fit` dt=0.01 | dt=0.05 | CV |
|---:|---:|---:|---:|
| 24 | 2.218 | 2.143 | 1.7 % |
| 32 | 2.579 | 2.336 | 5.0 % |
| 40 | 3.074 | 2.680 | 6.8 % |
| 50 | 4.060 | 3.199 | **11.9 %** |
| 75 | 7.273 | 4.644 | **22.1 %** |
| 100 | 12.895 | 9.966 | **12.8 %** |

Invariant to N=40, then not. On `t_settle` the picture is better: 8.00 vs 8.25 at N=50 (1.5 %),
diverging only at N=75–100 (8.01 vs 9.50, 8.07 vs 10.40).

**Grid C settles what the non-invariance is not.** Fixing `AGENT_STATE_TIMEOUT` at 0.25 s in
seconds — instead of 5·dt, which is 0.05 s at dt=0.01 and 0.25 s at dt=0.05 — moves τ at N=50 by:

| dt | τ with 5·dt | τ with 0.25 s fixed | Δ |
|---:|---:|---:|---:|
| 0.01 | 4.060 | 4.016 | **−0.044** |
| 0.05 | 3.199 | 3.199 | **+0.000** |

A five-fold change in detection latency moves τ by 1 %. **The dt-dependence is not detection
latency.** For Cap. 7 this is a one-sentence result in its own right: `tau_fit` is immune to the
failure detector's timeout in the clean regime.

---

## 3. c — ticks per hop

| N | dt | ms/hop | c | R² | `t_dissem` | c·(N−1)·dt | (N/2)·dt | coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.01 | 5.06 | 0.51 | 0.64 | 0.210 | 0.116 | 0.120 | 1.000 |
| 50 | 0.01 | 6.05 | 0.61 | 0.95 | 0.370 | 0.299 | 0.250 | 1.000 |
| 100 | 0.01 | 5.83 | 0.58 | 0.86 | 0.660 | 0.573 | 0.500 | 1.000 |
| 24 | 0.05 | 27.06 | 0.54 | 0.81 | 1.100 | 0.622 | 0.600 | 1.000 |
| 50 | 0.05 | 29.00 | 0.58 | 0.92 | 1.700 | 1.421 | 1.250 | 1.000 |
| 100 | 0.05 | 29.98 | 0.60 | 0.97 | 3.250 | 2.970 | 2.500 | 1.000 |

**c = 0.581, range [0.477, 0.653], N-independent.** Not 1, and not in [1, 4].

**Why c < 1.** Within one control tick the agents fire in some order. A pulse whose receiver
fires *after* its sender advances a hop in the *same* tick; one whose receiver fires first waits
for the next. Averaged over the ring that is roughly half the hops — hence ~0.5 ticks per hop.
This is the same intra-tick ordering that `DUAL_PULSE_BROADCAST_REPEATS=2` exists to defend
against.

**Correction to the assumed geometry.** The prompt's reference line was `c·(N/2)·dt`, on the
reasoning that each of the two counter-propagating pulses travels ~N/2 hops. Each pulse does —
but a node applies its shift only once **both** have arrived, so the controlling variable is
`max(h_CCW, h_CW)` and the *last* node to complete waits **N−1** hops, not N/2. Regressing
`t_apply` on `h_CCW` alone gives R² ≈ 0.03; on `max(h_CCW, h_CW)` it gives 0.56–0.98.

With that correction the measurement closes: `t_dissem ≈ t_detect + c·(N−1)·dt`. At N=100 /
dt=0.05: 0.25 + 0.60·99·0.05 = 3.22 s against 3.25 s measured.

**Coverage is 1.000 in every cell** — better than the 0.95–0.99 the prompt expected from
`ttl_coverage_results.csv`. With `TTL = 3N` no node is missed.

---

## 4. The crossover — it is not there

| N | dt=0.01: `t_dissem`/τ | dt=0.05 |
|---:|---:|---:|
| 24 | 0.095 | 0.513 |
| 50 | 0.091 | 0.531 |
| 100 | **0.051** | **0.326** |

The ratio never approaches 1, and it **falls** with N in both arms, because τ grows faster than
dissemination does. On `t_settle` the ratio is smaller still (0.66 / 8.07 = 0.08 at N=100,
dt=0.01).

**Dissemination is not what limits B2 in N ≤ 100, and there is no crossover to locate in this
range.** Pre-registration P3.3 confirmed. Extrapolating the two fitted lines, `t_dissem` would
only reach `t_settle` at dt=0.05 around N ≈ 300 — a regime the hysteresis limit of §6 forbids
long before.

---

## 5. Does "flat to N=100" survive? Both sentences, as requested

**If the thesis reports the robust metric (`t_settle`) — the version the data supports:**

> The 2-DOF overlay reconfigures in a time that is **flat in N**: 7.0 s at N=24 and 8.1 s at
> N=100 (dt=0.01, three seeds, ≤ 1 % spread), against a baseline that grows as Θ(N^1.94). The
> flatness is not an artefact of the exponential fit — it is measured with the enter-and-stay
> settling time, which does not assume a decay shape. At the coarser control period the
> flatness degrades mildly but does not break (7.3 → 10.4 s, +42 % over a 4× range of N).

**If the thesis keeps reporting the exponential fit (`tau_fit`) — the version the data forces:**

> The exponential-fit time constant of the overlay is **not** flat: it grows as N^1.25 (dt=0.01)
> and N^1.02 (dt=0.05), from 2.2 s at N=24 to 12.9 s at N=100. The advantage over the baseline
> therefore grows as N^0.69 rather than N^1.94, reaching ≈ 29× at N=100 rather than 148×.
> Caveat: at N ≥ 75 the fit's R² falls to 0.63–0.74, below the 0.9 the campaign requires of this
> metric, because the residual floor rises to 57 % of the fit's 5 %-of-peak window.

**Recommendation.** The first sentence is the defensible one, and it requires switching the
headline metric to `t_settle` — which the campaign's own README already designates as primary.
The cost is that the headline number changes from 2.1 s to ~8 s, and the N=50 advantage from
~21× to ~16×. That is a smaller, better-supported claim.

**The crossover is not the new positive result the prompt hoped for** — there is no crossover in
range. The new positive result is §1b: *the published metric stops measuring what it claims to
measure above N ≈ 50, and the campaign has a robust metric that does not.*

---

## 6. What is deliberately not here

**Grid B at N=100 — cut for cost, no scientific loss.** Budget alone is
`3.5·0.0417·100^1.94 ≈ 1087 s` of simulated time per run, i.e. hours. It would confirm an
exponent that is not in dispute: `baseline_long_results.csv` fixes 1.94 and
`largeN_results.csv` gives τ_base(100) = 311 s, which Lei 1 uses only through the exponent.
Gated behind `DTS_B_NS` in the runner rather than left to an operator decision.

**Grid A2 (N=150, 200) — cut for contamination, not only cost.** `HYSTERESIS_RAD` is **0.05 rad
absolute** and `HYSTERESIS_FRAC` is 0 (disabled), while the ideal gap at N=200 is
`2π/200 = 0.031 rad`. Above **N ≈ 126** the neighbour-switching hysteresis exceeds the ideal gap
— [config_param.py](../../config_param.py):263 documents exactly this limit. Measuring "the
algorithm's asymptote" there would measure the hysteresis artefact instead.

> **This is an independent finding.** The current code has a structural N ceiling at ≈ 126 that
> the campaign has never measured, and any scaling claim beyond it is unsupported by
> construction. Reaching the true asymptote requires a separate grid with `HYSTERESIS_FRAC` set
> (≈ 0.08 to match the legacy ratio at N=10) — a different configuration, hence a different
> experiment.

**The spacing caveat, declared as required.** At R = 20 m, N = 200 gives 0.63 m between agents
and there is no collision model. By the R-invariance of the angular loop (ω = K_TAU·u,
independent of R) the dynamics are identical at any radius, so such points would characterise
the **asymptote of the algorithm**, not a realisable formation. This stands as a caveat for any
large-N number the thesis cites, whether or not that grid is ever run.

---

## 7. Divergences from the pre-registration

1. **No instrumentation was needed.** The pre-registration already disclosed this:
   `events.csv` logs `dual_pulse_event_completed_*` with `(timestamp, node_id, h_CCW, h_CW,
   N_new)`, which is the (hop, t_apply) pair. Since there is **no code change**, the required
   "run one cell before and after and confirm τ is identical" verification is **vacuous** and was
   not performed — there is nothing that could have moved τ.
2. **`FAILURE_ENABLE=False`, as specified in the prompt, is wrong and silently destroys the
   experiment.** [protocol_agent.py](../../protocol_agent.py):266 only schedules the failure-check
   timer when `FAILURE_ENABLE` is set, and the deterministic-fault branch lives inside that
   timer's handler (`:866`). With it False the run completes with no event at all: measured
   `tau_fit` = 22.93 at N=24/dt=0.01 (an exponential fitted to noise) with zero dual_pulse events,
   against 2.22 expected. Caught after one cell; grid aborted, fixed to `True`, relaunched.
   The Poisson stream is bypassed for every agent whenever `DETERMINISTIC_FAILURE_ENABLE` is True,
   so True is both necessary and safe.
3. **`ADVERSARY_ROAM_SPEED_XY` cannot be pinned.** It is a literal at `config_param.py:157`,
   already 0.0. Recorded as observed, not fixed. Same for `TARGET_SWARM_SPIN_ENABLE` (`:746`).
4. **P3.0 — confirmed.** Predicted the ladder value (~4.0–4.1); measured 4.060.
5. **P3.1 — the prediction was directionally right and materially wrong.** Predicted p ≈ 0.8 on
   `tau_fit`; measured 1.25. More importantly the prediction was made on the wrong metric: the
   pre-registration did not anticipate that `tau_fit` would stop being a time constant, which is
   the actual result.
6. **P3.2 — confirmed.** Predicted c ≈ 0.5, in [0.45, 0.65], N-independent; measured 0.581 in
   [0.477, 0.653], N-independent. The mechanism predicted (intra-tick firing order) is consistent
   with the value. The reference line was corrected from `c·(N/2)·dt` to `c·(N−1)·dt`.
7. **P3.3 — confirmed.** Predicted no crossover in the dt=0.01 arm within N ≤ 100 and that
   dissemination does not drive the growth. Measured `t_dissem`/τ ≤ 0.11 at dt=0.01, falling
   with N.

---

## 8. What changes in the thesis

**`5-preliminary-results.tex` §The scaling law.** The claim "τ_B2 ≈ 2.1 s at N=50/75/100, flat"
rests on `largeN_results.csv`, whose N=50 value does not reproduce (§0), and on a metric that
stops being valid above N ≈ 50 (§1b). Replace with the `t_settle` version of the sentence in §5,
or with the `tau_fit` version plus its R² caveat. The "~149× at N=100" figure becomes ~29–42× on
`tau_fit`, or keeps its N^1.94 shape with a different constant on `t_settle`.

**`2-related-work.tex:177`** (the Ω(N) diameter-bound positioning). Strengthened, not weakened:
dissemination is measured at c ≈ 0.58 ticks/hop with coverage 1.000, and it is 5–11 % of the
reconfiguration time — the O(N) round complexity is real and is *not* the binding constraint,
which is what the positioning argues.

**Draft v2 `cap6_caracterizacao_adimensional.md`.** Lei 1 needs the metric named. On `t_settle`
the dimensionless collapse keeps its form; on `tau_fit` the exponent changes and the fit quality
must be reported alongside.

**Draft v2 `cap7_robustez.md`.** Two additions: grid C's result (τ is immune to a 5× change in
detector timeout in the clean regime) and the N ≈ 126 hysteresis ceiling of §6, which belongs
with the other characterised limits.

---

## Related

* [LADDER.md](LADDER.md) — P2, which raised the N=50 discrepancy this report arbitrates.
* [PROVENANCE.md](PROVENANCE.md) — the schema every row here carries.
* [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) — the pre-registration and the dated entry.
