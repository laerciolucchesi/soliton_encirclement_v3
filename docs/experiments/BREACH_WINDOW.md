# The breach window: kinematic or coordination-limited?

**Question.** The churn re-analysis ([CHURN_PAIRED.md](CHURN_PAIRED.md)) found that the overlay
improves the mean spacing error unanimously but shows no reliable effect on the tail — and that
the campaign had **never measured the mission-critical quantity at all**: the maximum angular
gap. This experiment measures it directly and tests the conjecture that the breach window is
limited by actuation (`Vmax`, `tau_a`), not by coordination.

**Answer, in one line.** The peak is the exact **expected value** `2(M−1)/M` of a distribution
no protocol can shift — not a floor (individual events land both below and above it; see §1).
The *duration* is actuation-limited — it scales with `tau_a`, is
completely flat in `Vmax` — and the overlay buys a real but modest **1.1–1.5×**. And the two
claims the thesis's motivation rests on are **both refuted by the simulator's own data**: the
reconfiguration time is *not* how long the gap stays open (it is 11× longer), and the breach
window does *not* grow as N² (it grows as N^0.3–0.6, and the breach *width* actually **shrinks**
with N).

## The experiment

Single **deterministic permanent** failure — deliberately not churn, where concurrent deaths
contaminate the peak (measured: peak `G_max` climbs 2.11 → 3.49 from rate 6 to 48).

| | |
|---|---|
| fixed | N = 24 (plus an N axis), one permanent victim at t = 5 s, uniform initial ring, loss = 0, delay = 0, dt = 0.05, `K_E_TAU` = 250/N, B2 knobs = validated config, TTL = 3N |
| swept | `VM_MAX_SPEED_XY` ∈ {2.5, 5, 10, 20} × `VM_TAU_XY` ∈ {0.5, 1, 2} × {baseline, B2} × 5 seeds, plus N ∈ {12, 24, 48} at `Vmax` = 10, `tau_a` = 1 |
| runs | **140, zero failures** |
| measured | peak `G_max`, peak gap in **degrees**, `t_close` at two thresholds, breach **area** ∫max(0, `G_max` − 1.25)dt — all anchored on the real `failure_start` timestamp from `events.csv` |

Requires the `alive_count` / `gap_max_rad` telemetry columns added in `e062eed`; without them
the absolute breach is not recoverable. Reproduce:

```powershell
python experiments/scaling_law/run_breach_window.py       # BREACH_VMAX / BREACH_TAUS / BREACH_N / BREACH_SEEDS
python experiments/scaling_law/analyze_breach_window.py   # tables + figure
```

Outputs: `breach_window_results_*.csv`, `breach_window_summary.csv`,
`breach_a1_reconfig_vs_breach.csv`, `figures/fig_breach_window.png`.

## 1. The peak is a geometric *expectation* — exactly

> **Corrected 2026-08-02 — it is a mean, not a bound.** The wording below used to say
> "floor" / "no protocol can beat this". That is false: it is an **exact expectation**, and
> under churn individual events fall **below** it about 30 % of the time. See §1.1.

Prediction **on a uniform ring**: one death merges two gaps of 2π/N while the ideal becomes
2π/(N−1), so the instantaneous peak is `2(N−1)/N` = **1.9167** at N = 24. The runs in this
table start uniform by construction (`INIT_ANGLES_EQUIDISTANT=True`, single deterministic
fault), so the uniform case is the one being measured here.

### 1.1 The general statement — a theorem, with no hypothesis on the configuration

Take a ring of `M` alive agents with gaps `g_1..g_M`, `Σ g_k = 2π`, **no assumption on how
the gaps are distributed**. Agent `i` is flanked by `g_{i-1}` and `g_i`; when it dies those
merge into `g_{i-1} + g_i`. Summing over every agent:

```
Σ_i (g_{i-1} + g_i) = 2 · Σ_k g_k = 4π        (each gap is counted twice)
```

So for a victim drawn uniformly at random among the alive — which is what the simulator does
(`protocol_agent.py:918-920`, one independent draw per agent at the same rate):

```
E[merged gap] = 4π/M    and, normalised by the new ideal 2π/(M−1):
E[peak G_max] = 2(M−1)/M      EXACTLY, FOR ANY CONFIGURATION
```

This is **not a bound**. It is a mean, and it is exact. The observed spread under churn —
30 % of events below `2(M−1)/M`, 24 % at it, 46 % above — is the variance around an exact
mean, not a violated floor. A uniform ring is simply the configuration with zero variance.

Two consequences for how this must be written:
* "no protocol can beat this" is **false** as stated. What no protocol can move is the
  *expectation*; single events are movable in both directions, and are moved.
* the churn peak climbing to 2.11 → 3.49 with rate (§ below) was already known to exceed
  1.92; what was never considered is that it also falls **below**. Exceeded from above and
  breached from below — it is a floor in neither direction.

| method | median peak | [min, max] | n | ratio to prediction |
|---|---:|---|---:|---:|
| baseline | **1.9174** | [1.9167, 1.9246] | 60 | 1.0004 |
| B2 | **1.9174** | [1.9167, 1.9246] | 60 | 1.0004 |

Identical between methods at **every** `Vmax` (ratio 1.00, 0/60 losing pairs) and every
`tau_a`. This half of the conjecture is confirmed **for the uniform ring these runs start
from** — and there it needed no experiment: the peak happens at the instant of the failure,
before any protocol can act. It should be stated in the thesis as an **exact expectation**
(§1.1), not as a bound and not as an optimisation target. Stating it as a bound is wrong:
under churn the ring is not uniform and ~30 % of events land below it.

## 2. The duration is actuation-limited, and the overlay buys 1.1–1.5×

`t_close` = the time after the failure at which `G_max` stops exceeding the threshold for good
(not "first crosses" — a transient dip on the way down would report an open breach as closed).

**By `tau_a`** (N = 24, pooled over `Vmax`, 20 pairs each):

| `tau_a` | baseline | B2 | ratio | n_lose | p |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 2.30 s | 1.80 s | 1.28 | 0/20 | 0.0001 |
| 1.0 | 3.15 s | 2.35 s | 1.35 | 0/20 | 0.0001 |
| 2.0 | 4.75 s | 3.35 s | 1.42 | 0/20 | 0.0001 |

**By `Vmax`** (N = 24, pooled over `tau_a`, 15 pairs each): baseline 3.15 s and B2 2.35 s at
`Vmax` = 2.5, 5, 10 **and** 20 — the medians are *identical to three digits across an 8× range
of speed*.

So the breach window scales with `tau_a` and is completely insensitive to `Vmax`. It is limited
by the **first-order actuation lag**, not by top speed and not by coordination. The conjecture's
kinematic half is confirmed — with the qualification that coordination is not irrelevant: the
overlay wins **60/60 pairs**, p = 1e-4, by 1.28–1.42×, and the advantage *grows* with `tau_a`.

At the stricter threshold the overlay's edge is larger — 3.08× / 2.22× / 2.00× at
`tau_a` = 0.5/1/2 for `t_close(1.10)` — which is consistent with the mechanism: getting under
1.25 only needs the two neighbours of the hole to move in (fast for both methods), while getting
under 1.10 needs the ring-wide redistribution that is exactly what the overlay accelerates.

## 3. Refuted: "the reconfiguration time is how long the gap stays open"

`draft v1 / 1-introduction.tex:26` asserts this as an identity. Both quantities come from the
same `target_telemetry.csv`, so it is directly checkable (140 runs, `breach_a1_reconfig_vs_breach.csv`):

| method | `t_close` (gap closes) | `t_settle` on `E_gap` (ring redistributes) | `tau_fit` | ratio |
|---|---:|---:|---:|---:|
| baseline | 3.15 s | **35.15 s** | 20.29 s | **11.2×** [5.3, 17.4] |
| B2 | 2.35 s | 6.95 s | 2.14 s | 3.00× [2.6, 3.4] |

They are not the same quantity and they are not close. The gap closes **11× before** the
baseline ring finishes redistributing. The consequence is uncomfortable and must be stated
plainly:

> The overlay's advantage on **reconfiguration time** is 35.15/6.95 ≈ **5×** (and 9–149× in the
> published large-N campaign). Its advantage on the **breach window** is 3.15/2.35 = **1.34×**.
> These are different results about different quantities, and only the second one is about the
> thing the introduction calls mission-critical.

## 4. Refuted: "the breach window grows as N², at N=100 it stretches to minutes"

`draft v1 / 1-introduction.tex:36-37` and `6-conclusion.tex:65-68`. Measured at
`Vmax` = 10, `tau_a` = 1, 5 seeds per cell:

| N | baseline `t_close(1.25)` | B2 | baseline `t_close(1.10)` | B2 |
|---:|---:|---:|---:|---:|
| 12 | 2.40 s | 2.15 s | 4.65 s | 2.65 s |
| 24 | 3.15 s | 2.35 s | 7.65 s | 3.45 s |
| 48 | 3.70 s | 2.55 s | 11.35 s | 4.60 s |

Fitted exponent in `t_close ~ N^p` — **the claim under test is p ≈ 2**:

| metric | method | p | extrapolated to N = 100 |
|---|---|---:|---:|
| `t_close(1.25)` | baseline | **0.31** | **4.7 s** |
| `t_close(1.25)` | B2 | 0.12 | 2.8 s |
| `t_close(1.10)` | baseline | **0.64** | **18.2 s** |
| `t_close(1.10)` | B2 | 0.40 | 6.2 s |

Not N², not minutes. The Θ(N²) is real but it governs the **E_gap relaxation** (§3), not the
breach.

### And the breach *width* shrinks with N

The absolute peak gap, which is what the perimeter-defense condition is stated on (a maximum
admissible spacing set by the target's speed — `1-introduction.tex:31-33`):

| N | peak gap | ideal gap after the death | final gap |
|---:|---:|---:|---:|
| 12 | **60.1°** | 32.7° | 32.7° |
| 24 | **30.0°** | 15.7° | 15.8° |
| 48 | **15.1°** | 7.7° | 7.8° |

A bigger swarm has a *narrower* absolute breach for the same single failure, because each gap is
smaller to begin with. So on **both** axes — duration and width — the single-failure risk
**decreases** with swarm size. That is the opposite of the motivation's "this is also why scale
matters".

## 5. Where the overlay's value actually is, and what the thesis should argue instead

Nothing above damages the Θ(N²) result or the flat-in-N reconfiguration result: both are
reproduced here (baseline `tau_fit` 20.29 s vs B2 2.14 s at N = 24). What is damaged is the
*motivational bridge* from those results to mission relevance, which ran through a single-failure
breach window that turns out to be short, weakly N-dependent, and narrowing with N.

The reframing the data does support:

**Slow reconfiguration matters because the ring is not ready for the *next* event.** At N = 100
the baseline's relaxation is `0.033·N²` ≈ 330 s. Any realistic failure rate then finds the ring
permanently non-uniform, and a failure landing on a non-uniform ring opens a *worse* gap than one
landing on a uniform ring. The existing churn data already shows this compounding: peak `G_max`
climbs from 2.11 at 6 failures/min to **3.49** at 48/min — well above the single-failure floor of
1.92 — and the overlay's unanimous win on `egap_avg` (32/32 pairs, p < 0.001) is precisely a
measurement of how much more often the ring is uniform when the next event arrives.

That argument is mission-relevant, N²-driven, and supported by data already in hand. It is also
**not yet the argument the thesis makes**, and making it requires one more experiment
(§6) because the existing churn campaign never measured `G_max`.

## 6. The one experiment that would close this

Repeat the churn sweep with the new telemetry, and report the breach metrics instead of (or
alongside) `egap_avg`:

```powershell
$env:CHURN_TAG="gmax"; python experiments/scaling_law/run_churn_sweep.py
```
plus adding `gap_max_rad` / `alive_count` aggregation to `metrics_from_tgt`. Cost: the same
64 runs as `c3_churn8_dt05`, now with the mission-critical metric recorded.

**Prediction, stated before running** (so it can fail): under churn the overlay's advantage on
peak `G_max` and on time-above-threshold should be **larger** than the 1.34× measured here for a
single failure, and should **grow with the churn rate** — because that is where incomplete
redistribution compounds. If instead the advantage stays ~1.3× and flat in rate, then the overlay
does not buy breach safety at all, and C5 must say so.

## Verdict against the decision rule

The rule was set in [CHURN_PAIRED.md](CHURN_PAIRED.md) §5.3 *before* the runs:

| pre-registered observation | outcome |
|---|---|
| peak `G_max` ≈ 1.92 for both, flat in `Vmax`/`tau_a` | ✅ **confirmed** (1.9174 vs 1.9167, identical, flat) |
| `t_close` ~equal for both, scaling with `tau_a` | ⚠️ **half-confirmed**: scales with `tau_a`, flat in `Vmax` (actuation-limited) — but B2 wins 60/60 by 1.28–1.42× |
| `t_close` flat for B2 while baseline grows with N | ❌ **refuted**: baseline grows as N^0.31, B2 as N^0.12; neither is Θ(N²) |
| `t_close` worse for B2 | ❌ except one corner: breach **area** at `tau_a` = 2 is a tie with 16/20 pairs slightly worse for B2 |

**Conjecture: confirmed in substance.** The breach window is actuation-limited. Coordination
buys a consistent but modest factor, and the platform's `tau_a` — not the protocol, not the swarm
size — sets the floor.

## Caveats

* Single **permanent** failure. Recovery (ENTRADA) is untested here; a rejoin re-narrows the gap
  and its transient may behave differently.
* `t_close` uses "last excursion above threshold", which is budget-dependent in principle. The
  budget (60 s, 120 s for the N axis) is 15–25× the measured `t_close`, and no run showed a late
  re-excursion, so the measure is stable here — it would not be under churn.
* Thresholds 1.25 and 1.10 are stand-ins for the literature's "maximum admissible spacing", which
  is set by the target's speed and is **absolute**. With `gap_max_rad` now logged, a real
  threshold in degrees can replace them.
* The N axis has 3 points (12/24/48) and 5 seeds; the exponent is a fit through 3 medians. It
  separates p ≈ 0.3 from p ≈ 2 decisively, but it does not pin p to two digits.

## Related

* [CHURN_PAIRED.md](CHURN_PAIRED.md) — the re-analysis that raised the question, and the metric
  semantics (`E_gap` is a ring-wide RMS; `egap_max` is not the maximum gap).
* [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) — 2026-07-26 entry.
* [PROVENANCE.md](PROVENANCE.md) — every row here carries its commit and pinned parameters.
