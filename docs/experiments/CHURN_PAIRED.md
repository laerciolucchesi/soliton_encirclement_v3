# Churn campaign, re-analysed pairwise across every collected metric

**Question (E4).** The thesis argues the **maximum angular gap** is the mission-critical
quantity — the breach window a target escapes through. Robustness under churn was
reported on the **mean** spacing error (`egap_avg`). What do the other already-collected
metrics say?

**Answer, in one line.** The overlay's advantage is real, large and unanimous on the
**mean** (32/32 pairs, p < 0.001), shrinks to a small-but-consistent edge on the **P90**,
and **disappears on the maximum** (14/32 pairs worse) and on **fairness** (15/32 worse,
p = 0.73). And the metric the thesis actually needs — the maximum angular gap — **was never
measured by this campaign at all**, although the simulator has been logging it in every run.

Reproduce with (no simulation; reads only existing CSVs):

```powershell
python experiments/scaling_law/analyze_churn_paired.py     # tables + churn_paired_results.csv + figure
python experiments/scaling_law/probe_gmax_floor.py         # G_max probe on surviving telemetry
```

Source: `churn_sweep_results_c3_churn8_dt05.csv` (64 rows = 2 methods × 4 rates × 8 seeds;
N = 24, `tau_xy` = 1.0, dt = 0.05, `off` = 8 s). Outputs: `churn_paired_results.csv`,
`figures/fig_churn_paired.png`, `gmax_probe_results.csv`.

---

## 0. What the metrics actually mean

This section must be read before any number below. Three of the six metrics are not
computed where you would expect, and one of them does not measure what its name suggests.

### 0.1 The two quantities the simulator computes per tick

Both come from `protocol_target.py`, in the same loop, from the same sorted list of angles
(`protocol_target.py:655-686`):

```python
ideal_gap = two_pi / float(M)          # M = len(angles) = ALIVE agents
    ratio = gap / ideal_gap
    if ratio > max_ratio: max_ratio = ratio
    e_gap = ratio - 1.0
    sum_sq_gap += e_gap * e_gap

G_max = float(max_ratio)                                  # protocol_target.py:685
E_gap = float(math.sqrt(sum_sq_gap / count_gap))          # protocol_target.py:686
```

* **`G_max`** = `max_k (gap_k / ideal_gap)` — **this is the maximum angular gap**, normalised
  by the ideal one. The mission-critical quantity.
* **`E_gap`** = **RMS across the ring** of the relative gap error. A *spatial aggregate*. It
  is not a maximum of anything.

Both are written to `target_telemetry.csv` on every run
(`protocol_target.py:706`). **`G_max` has been recorded all along and no churn analysis has
ever aggregated it.**

### 0.2 Where each campaign metric comes from

> **`egap_avg` / `egap_p90` / `egap_max` do NOT come from `metrics_util.py`.** For the churn
> campaign they are computed by a local helper inside the runner,
> `run_churn_sweep.metrics_from_tgt` (`run_churn_sweep.py:47-56`):

```python
steady = df[df["timestamp"] >= T0 + WARMUP_AVG]["E_gap"].to_numpy(float)   # T0=5, WARMUP_AVG=15
return {"egap_avg": float(np.mean(steady)),
        "egap_p90": float(np.percentile(steady, 90)),
        "egap_max": float(np.max(steady))}
```

Only `effort_mean_v2`, `sat_frac` and `fairness_p95` come from `metrics_util.effort_metrics`
(`metrics_util.py:130-173`), called at `run_churn_sweep.py:91-92` with the same `t0 = 20 s`.

> **Naming collision worth knowing about:** `metrics_util.event_metrics` defines its *own*
> `egap_avg` over a *different* window — `sub[sub.timestamp >= t0 + 10.0]`
> (`metrics_util.py:119`), i.e. t ≥ 15 s with the usual `t0 = 5`. So the column `egap_avg`
> means one thing in `churn_sweep_results*.csv` (t ≥ 20 s) and another in
> `collapse_results*.csv` / `trackC_results*.csv` (t ≥ 15 s). They are not comparable
> across campaigns without saying which runner wrote the file.

### 0.3 The four questions, answered per metric

| metric | (a) window | (b) initial transient included? | (c) denominator | (d) direction |
|---|---|---|---|---|
| `egap_avg` | **t ∈ [20 s, 155 s]**, whole run — **not** post-event | **No** | alive at the instant | lower is better |
| `egap_p90` | same | **No** | alive at the instant | lower is better |
| `egap_max` | same | **No** | alive at the instant | lower is better |
| `effort_mean_v2` | same (t ≥ 20 s), pooled over all nodes and samples | **No** | — (`v/Vmax`, `Vmax` fixed) | lower = cheaper |
| `sat_frac` | same | **No** | — | lower is better |
| `fairness_p95` | same | **No** | — (`\|e_tau_real\|` per node) | lower is better |

**(a) Window.** All six share `t ≥ T0 + WARMUP_AVG = 20 s` and run to the end
(`SIM_DURATION = T0 + BUDGET = 155 s`). None is post-event: churn has no single event to
anchor to, so these are whole-regime statistics over ~2 700 telemetry samples at dt = 0.05.

**(b) Initial transient.** No. The 15 s warm-up excludes it, and in any case the churn runs
start already equidistant at exactly R (`INIT_ANGLES_EQUIDISTANT=True`,
`INIT_RADIUS_RANGE=0.0`, `run_churn_sweep.py:73`), so there is almost no formation
transient to exclude. Outages that *begin* before t = 20 s and are still open after it do
enter — which is intended.

**(c) Denominator — alive, not nominal.** `M = len(angles)` iterates `self.agent_states`
(`protocol_target.py:606`), which `_prune_expired_states(now)` has just emptied of anything
older than `AGENT_STATE_TIMEOUT` (`protocol_target.py:549-562`, called at `:571`). So M is
"alive as the target currently detects it", lagging a real death by at most one timeout
(5·dt = 0.25 s in these runs).

**(d)** All six are lower-is-better. `effort_mean_v2` is reported below as a **cost ratio**
B2/baseline (>1 ⇒ the overlay spends more), the others as an **advantage ratio**
baseline/B2 (>1 ⇒ the overlay is better).

### 0.4 Two consequences that change the interpretation

> ### ⚠️ `egap_max` includes the whole run, and it is **not** the maximum angular gap.
>
> It is the **maximum over time** of a **spatial RMS**. Two distinct aggregations are stacked:
> first RMS across the ring (which *averages away* a single wide gap among 23 narrow ones),
> then max over ~2 700 samples (an extreme-value statistic, driven by one instant). Calling
> it "the worst gap" is wrong on both counts: it is the worst *instant* of a *ring-average*.
> The actual worst gap is `G_max`, and it is not in any churn CSV.

> ### ⚠️ Both `E_gap` and `G_max` are normalised by the **current alive count**.
>
> A ring of 12 survivors spread perfectly scores `E_gap = 0` and `G_max = 1` — identical to
> a full ring of 24 — even though every physical gap is twice as wide. These metrics measure
> **redistribution quality**, not **absolute coverage**. The absolute breach in radians is
> `G_max · 2π/M`, and **M is not logged** (`target_telemetry.csv` carries only
> `timestamp, E_r, E_vr, rho, G_max, E_gap`), so the absolute gap **cannot be recovered from
> existing telemetry**. If the thesis wants to argue about a physical escape window, this is
> a one-line instrumentation gap that must be closed first (§4.4).

---

## 1. Paired results

Pairing is by `(rate_total, seed)`: baseline and B2 share `EXPERIMENT_SEED`, hence the same
Poisson failure stream, so the comparison is within-stream. 8 pairs per rate, 32 aggregated.
Wilcoxon signed-rank on the **raw paired values** (not on the ratios); `r = |Z|/√n_eff`.

| metric | scope | n | median | min | max | n_lose | p | r |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **egap_avg** (base/B2) | rate 6 | 8 | 1.31 | 1.24 | 1.34 | **0** | 0.0078 | 0.89 |
| | rate 12 | 8 | 1.23 | 1.14 | 1.30 | **0** | 0.0078 | 0.89 |
| | rate 24 | 8 | 1.15 | 1.11 | 1.20 | **0** | 0.0078 | 0.89 |
| | rate 48 | 8 | 1.14 | 1.11 | 1.18 | **0** | 0.0078 | 0.89 |
| | **aggregate** | 32 | **1.19** | 1.11 | 1.34 | **0** | **<0.001** | 0.87 |
| **egap_p90** (base/B2) | rate 6 | 8 | 1.04 | 1.00 | 1.08 | 1 | 0.0156 | 0.84 |
| | rate 12 | 8 | 1.05 | 1.03 | 1.10 | 0 | 0.0078 | 0.89 |
| | rate 24 | 8 | 1.09 | 1.03 | 1.12 | 0 | 0.0078 | 0.89 |
| | rate 48 | 8 | 1.13 | 1.06 | 1.18 | 0 | 0.0078 | 0.89 |
| | **aggregate** | 32 | **1.07** | 1.00 | 1.18 | 1 | **<0.001** | 0.87 |
| **egap_max** (base/B2) | rate 6 | 8 | 0.99 | 0.92 | 1.19 | **5** | 0.945 | 0.05 |
| | rate 12 | 8 | 1.06 | 0.85 | 1.40 | **3** | 0.250 | 0.45 |
| | rate 24 | 8 | 1.15 | 1.00 | 1.35 | 1 | 0.0156 | 0.84 |
| | rate 48 | 8 | 0.98 | 0.91 | 1.46 | **5** | 0.742 | 0.15 |
| | **aggregate** | 32 | **1.05** | 0.85 | 1.46 | **14** | 0.0165 | 0.42 |
| **fairness_p95** (base/B2) | rate 6 | 8 | 1.02 | 0.89 | 1.08 | 1 | 0.195 | 0.50 |
| | rate 12 | 8 | 0.98 | 0.85 | 1.13 | **6** | 0.461 | 0.30 |
| | rate 24 | 8 | 0.97 | 0.60 | 1.21 | **4** | 0.547 | 0.25 |
| | rate 48 | 8 | 0.99 | 0.57 | 1.63 | **4** | 0.844 | 0.10 |
| | **aggregate** | 32 | **1.00** | 0.57 | 1.63 | **15** | 0.733 | 0.06 |
| **sat_frac** | all | 32 | — | — | — | 0 | n/a | n/a |
| **effort_mean_v2** (B2/base) | rate 6 | 8 | 2.54 | 2.37 | 3.23 | **8** | 0.0078 | 0.89 |
| | rate 12 | 8 | 2.38 | 2.22 | 2.72 | **8** | 0.0078 | 0.89 |
| | rate 24 | 8 | 2.35 | 1.97 | 2.66 | **8** | 0.0078 | 0.89 |
| | rate 48 | 8 | 2.37 | 1.74 | 2.65 | **8** | 0.0078 | 0.89 |
| | **aggregate** | 32 | **2.41** | 1.74 | 3.23 | **32** | **<0.001** | 0.87 |

`sat_frac` is **identically 0.0** in all 64 cells: every paired difference is exactly zero,
so the test is undefined. `n_lose` counts pairs where B2 is worse (advantage < 1, or cost
> 1 for effort); `churn_paired_results.csv` also carries `n_lose_5pct`, the ±5 % margin
`run_churn_sweep.py:151` uses.

**Caveat on the aggregate rows.** Pooling 4 rates × 8 seeds into one n = 32 Wilcoxon treats
rate as noise when it is a designed factor. For `egap_avg` this is harmless (every per-rate
test is already significant in the same direction). For `egap_max` it is **not**: the pooled
p = 0.0165 is produced almost entirely by the rate-24 block, while rates 6 and 48 sit at
p = 0.945 and p = 0.742 with 5/8 losses each. Read the per-rate rows, not the pooled one.

### One sentence per metric

* **`egap_avg`** — *supports*: under continuous churn the overlay reduces the time-averaged
  spacing error on **every** seed at **every** rate (32/32, p < 0.001, large effect); this
  reproduces the §7.2.7 claim exactly (1.31/1.23/1.15/1.14, min ≥ 1.11, zero losses).
  *Does not support*: any claim about worst-case behaviour, since a mean over 135 s is
  insensitive to brief excursions.
* **`egap_p90`** — *supports*: the improvement is not confined to the bulk; the upper decile
  improves too, and the edge **grows with churn rate** (1.04 → 1.13). *Does not support*:
  calling it a tail guarantee — 1.07 median is a 7 % edge, an order of magnitude smaller than
  the mean effect.
* **`egap_max`** — *supports*: essentially nothing. Median 1.05 with range [0.85, 1.46] and
  **14/32 pairs where the overlay is worse**; two of four rates are indistinguishable from
  chance. *Does not support*: "the overlay improves the worst case". And even a positive
  result here would not have been about the maximum angular gap (§0.4).
* **`fairness_p95`** — *supports*: the overlay does not systematically *harm* the worst-served
  node (median 1.00). *Does not support*: any fairness benefit — 15/32 losses, p = 0.73,
  r = 0.06, and the spread [0.57, 1.63] is wider than for any other metric.
* **`sat_frac`** — *supports*: no actuator saturation anywhere, for either method, at any
  rate; consistent with the anti-windup diagnosis of 2026-06-13. *Does not support*: any
  comparison, the metric is constant.
* **`effort_mean_v2`** — *supports*: the cost is real, uniform and unambiguous — 2.41× the
  baseline, 32/32 pairs, p < 0.001, roughly flat in rate. *Does not support*: calling it
  dangerous; absolute velocities remain at 3–9 % of `Vmax` with `sat_frac = 0`.

**Figure** — `experiments/scaling_law/figures/fig_churn_paired.png` (regenerated by the
command at the top of this page; `*.png` is gitignored, so it is not in the repository).
One panel per metric; within each, 4 rate groups with the 8 pairs joined by a grey line
(baseline → B2) and the median in black; the narrow companion panel shows the per-pair ratio
against the ratio = 1 line, annotated with the per-rate Wilcoxon p.

---

## 2. Does the `egap_max` pattern hold across campaigns?

Yes. It is not specific to `c3`. Aggregate paired comparison, all rates pooled:

| campaign | seeds, dt | `egap_avg` median [min, max], n_lose, p | `egap_max` median [min, max], n_lose, p |
|---|---|---|---|
| `c3_churn8_dt05` | 8, 0.05 | **1.19** [1.11, 1.34], **0**/32, <0.001 | 1.05 [0.85, 1.46], **14**/32, 0.017 |
| `m8off_ablation8seed` | 8, 0.01 | 1.15 [0.71, 1.49], **11**/32, 0.172 | 1.04 [0.70, 1.38], **10**/32, 0.027 |
| `c1B_m8on_dt01` | 3, 0.01 | **1.28** [1.17, 1.40], **0**/12, <0.001 | 1.13 [0.94, 1.34], 3/12, 0.009 |
| `c1C_dt05` | 3, 0.05 | **1.16** [1.10, 1.30], **0**/6, 0.031 | **0.98** [0.97, 1.37], **5**/6, 0.438 |

The pattern is consistent in all four: **the `egap_max` advantage is always far smaller than
the `egap_avg` advantage, and always has losing pairs where `egap_avg` has none.** In
`c1C_dt05` the median advantage is below 1 outright. The only campaign where `egap_max`
looks respectable (`c1B_m8on_dt01`, 1.13, p = 0.009) has 12 pairs, and its `egap_avg`
advantage is correspondingly the highest of the set (1.28) — the ordering never inverts.

The `m8off` row is a useful control: with M8 disabled the mean advantage itself becomes
unreliable (11/32 losses, p = 0.17), confirming M8 is what makes the mean result hold, and
that it does **not** rescue the maximum.

---

## 3. Probe: is there a floor on the maximum gap? (`probe_gmax_floor.py`)

`churn_sweep_runs_stamp/` is the only churn run directory whose `target_telemetry.csv`
survived — 24 runs, dt = 0.01, N = 24, `tau` = 1.0, `off` = 8 s, rates 6/12/24/48, seeds 0–2.
It therefore still contains `G_max`.

> **Attribution warning.** The **baseline** half of these runs matches, to floating-point
> exactness, the baseline of the dt = 0.01 ablation family
> (`churn_sweep_results_m8off_ablation8seed.csv`, `_add_clean`, `_gate_clean`, and
> `c1B_m8on_dt01`). The **B2** half matches **no committed result CSV** (closest median
> |Δ`egap_avg`| = 4.3 × 10⁻², against values of order 0.1–0.5) and it *loses* on `egap_avg`
> (ratio 0.72), unlike every committed B2. Which overlay configuration produced it is
> **unrecoverable** — there was no provenance before P0. So: the baseline numbers below are
> evidence; the B2 numbers are from an unidentified overlay variant and are **not** evidence
> about the validated B2.

Geometric prediction for one death in a nominally-N ring: the two adjacent gaps merge into
`2·(2π/N)` while the ideal becomes `2π/(N−1)`, so the instantaneous peak is

  `G_max_peak = 2(N−1)/N = 1.917` for N = 24.

Observed peak `G_max` (median over seeds):

| rate/min | baseline | B2 (unattributed) | baseline / predicted |
|---:|---:|---:|---:|
| 6 | **2.113** | 2.034 | **1.10** |
| 12 | 2.266 | 2.242 | 1.18 |
| 24 | 3.058 | 2.684 | 1.60 |
| 48 | 3.485 | 3.469 | 1.82 |

At the sparse rate — where events are mostly isolated single deaths — the observed peak sits
**within 10 % of the protocol-independent geometric value**, and the two methods are within
4 % of each other. As the rate rises the peak climbs well above the single-death prediction,
which is what concurrent deaths must do (k adjacent deaths merge k+1 gaps).

Paired over all 12 pairs, no `G_max` statistic separates the two methods:

| statistic | baseline median | B2 median | ratio [min, max] | n_lose | p |
|---|---:|---:|---|---:|---:|
| `G_max` mean | 1.425 | 1.497 | 0.98 [0.77, 1.05] | 7/12 | 0.129 |
| `G_max` P90 | 1.893 | 1.882 | 1.00 [0.90, 1.06] | 7/12 | 0.424 |
| `G_max` P99 | 2.357 | 2.349 | 1.01 [0.79, 1.11] | 5/12 | 0.970 |
| `G_max` max | 2.644 | 2.644 | 1.02 [0.79, 1.20] | 5/12 | 0.910 |
| fraction of time > 1.5× ideal | 0.313 | 0.405 | 0.92 [0.21, 1.20] | 9/12 | 0.052 |
| fraction of time > 2× ideal | 0.064 | 0.066 | 0.89 [0.18, 4.09] | 7/12 | 0.129 |

Given the attribution warning this **cannot decide** whether the validated overlay improves
`G_max`. What it does establish: `G_max` is extractable from existing telemetry at zero
simulation cost, and the peak's magnitude at sparse churn matches the geometric prediction
closely enough that a protocol-independent floor is a live hypothesis rather than a guess.

---

## 4. What this forces to change

### 4.1 Draft v2 — `docs/thesis/draft/cap7_robustez.md`

* **§7.2.7, "Confirmação 8-seed (vantagem PAREADA por seed)"** (around lines 330-340). The
  claim *"o overlay **ajuda em 8/8 seeds, em todas as taxas** — adv_med 1,31/1,23/1,15/1,14 e
  adv_min ≥ 1,11 … zero seeds onde perde"* is **verified exactly** by this re-analysis. It is
  also, as written, **silent about which metric it is** — it holds for `egap_avg` only.
  Required edit: state the metric explicitly, add the P90 result as the tail check, and add
  the negative results for max and fairness. Suggested replacement sentence: *"…ajuda em 8/8
  seeds em todas as taxas **no erro médio de espaçamento** (`egap_avg`, p < 0,001, r = 0,87);
  no P90 a vantagem cai para 1,07 (mas cresce com a taxa, 1,04→1,13); **no máximo e na
  fairness não há efeito** (14/32 e 15/32 pares em que o overlay é pior)."*
* **§7.2.7, the `[decidir depois]` note** (lines 317-321) asks whether to keep the unpaired
  table. Decision now has evidence: keep only the paired one, and add the p-values — the
  repository had **no statistical test at all** before this re-analysis.
* **New subsection needed** — the metric-semantics box of §0.4 belongs in Cap. 7 as a
  definition, because the chapter uses `E_gap` throughout without saying it is a spatial RMS
  normalised by the *alive* count.
* **The cost line** *"esforço de controle ~2,2–2,7× o baseline"* (line 339) should become
  2.41× median [1.74, 3.23], 32/32 pairs, p < 0.001 — the interval is what makes it a
  characterised trade-off rather than an anecdote.

### 4.2 Draft v2 — `docs/thesis/draft/cap9_conclusao.md`

* **§9.1, RQ4** (lines 23-27): *"o overlay ajuda ou degrada graciosamente sob perda, atraso,
  fora-de-ordem, **churn (vantagem pareada, 8/8 seeds)** e movimento"*. The parenthetical
  must name the metric. As it stands a reader will carry it to the max-gap claim of the
  introduction, which the data does not support.
* **§9.2, C5 — "o mapa de robustez"** (lines 53-57): the claim *"ele degrada graciosamente
  para o baseline auto-estabilizante"* survives, and is in fact **strengthened** by the
  `sat_frac = 0` and `fairness ≈ 1.00` results (no harm done). But C5 should say what the map
  covers: mean spacing error, not worst-case gap.
* **§9.3, Limitações honestas**: add one item — *the mission-critical quantity (maximum
  angular gap) was not measured; the campaign's `egap_max` is the maximum over time of a
  spatial RMS, not a maximum gap, and shows no overlay advantage.* This is exactly the kind
  of limitation §9.3 already handles well.

### 4.3 Draft v1 (`5-preliminary-results.tex`, `6-conclusion.tex`) — not found

There is **no `.tex` file anywhere** in this repository, nor in the sibling projects
(`../soliton_encirclement`, `../soliton_encirclement_v2`, `../SolitonSwarm`). Draft v1 lives
somewhere outside this workspace, so the corresponding edits could not be located. The v2
edits above are the same in substance; apply them to the v1 §robustez and §C5 by hand.

### 4.4 Instrumentation (blocking, one line)

`target_telemetry.csv` carries `G_max` but **not the alive count**, so the absolute maximum
gap in radians (`G_max · 2π/M`) is not recoverable — including from the 765 result rows
already collected. `protocol_target.py` already computes `alive_count`
(`protocol_target.py:313`); adding it (and, better, `gap_max_rad` directly) to the telemetry
row is a one-line change that unblocks every future breach-window claim. **Do this before
running §4.5.** Note this changes the `target_telemetry.csv` schema — readers must tolerate
the added column (they all select by name, so they will).

---

## 5. Mechanistic hypothesis and the experiment that decides it

### 5.1 The conjecture, restated

The a-priori conjecture — *the max gap jumps to ~2× ideal after a failure and closing it is
limited by `Vmax` and `tau_a`, not by coordination, so there is a protocol-independent floor*
— must be split in two, because half of it is true by construction and half is the real
question.

* **The peak is trivially protocol-independent.** At the instant of the death the two
  adjacent gaps merge before any protocol can act: `G_max` jumps to `2(N−1)/N = 1.92`
  regardless of the algorithm. No coordination scheme can beat this; it is geometry, not
  control. The probe (§3) measures 2.03–2.11 at sparse churn, within 10 % of it. **This part
  of the conjecture needs no experiment — it needs to be stated as a bound in the thesis.**
* **The *dwell* above threshold is the open question.** How long the breach stays open is a
  kinematic race, and that is where a protocol can win or lose.

### 5.2 Why the mean improves and the maximum does not

The two aggregations weight different parts of the same trajectory:

* `egap_max` is set by the **instant of the event** — geometric, identical for both methods.
* `egap_avg` is set by the **recovery** that follows — `Θ(N²) ≈ 20 s` for the baseline versus
  ≈ 2 s for B2. That is where the entire advantage lives.

This predicts that a metric weighting recovery more heavily should show more advantage, and
that the advantage should shift between metrics as churn density changes. **The data already
confirms this**: as the rate rises from 6 to 48, `egap_avg`'s advantage *falls* (1.31 → 1.14,
because the baseline never settles and its whole trace is elevated) while `egap_p90`'s
advantage *rises* (1.04 → 1.13, because at high density even the upper decile becomes
recovery-dominated rather than peak-dominated). The two converge to ≈ 1.14 at rate 48. A
peak-dominated metric (`egap_max`) sits outside this convergence and shows no trend — which
is what the mechanism predicts.

There is a second, sharper implication. B2's reconfiguration time is ≈ 2.1 s ≈ 2·`T_FF` with
`T_FF = tau_a = 1 s`, and the largest displacement any node must make is about
`r·gap_new/2 ≈ 20 · (2π/23)/2 ≈ 2.7 m` — which at `Vmax = 10 m/s` takes ≈ 0.27 s of travel
but ≈ 2–3·`tau_a` of first-order lag. **B2 is already at the actuation-limited floor**; the
remaining lever on the breach window is platform agility, not coordination. If that is right,
no further protocol work can shorten the breach, and the thesis should say so as a positive
characterisation rather than leave it as an unexplored gap.

### 5.3 Minimal deciding experiment

Deliberately *not* a churn sweep: under continuous churn, events overlap and the peak is
contaminated by concurrent deaths (§3 shows the peak climbing from 2.11 to 3.49 with rate).
The clean design is the **single deterministic failure** the scaling-law campaign already uses.

**Fixed:** `DETERMINISTIC_FAILURE_ENABLE=True`, one victim, `DETERMINISTIC_FAILURE_OFF_TIME=-1`
(permanent), `DETERMINISTIC_FAILURE_TIME_T0=5`, `N=24`, `INIT_ANGLES_EQUIDISTANT=True`,
`COMMUNICATION_*=0`, `CONTROL_PERIOD=0.05`, `K_E_TAU=250/N`, `DUAL_PULSE_INTEGRATION=B2`,
`DUAL_PULSE_DELTA_SCALE=1.0`, `DUAL_PULSE_TTL_HOPS=3N`, 5 seeds.

**Swept — the kinematic axes, which is the whole point:**
`VM_MAX_SPEED_XY ∈ {2.5, 5, 10, 20}` × `VM_TAU_XY ∈ {0.5, 1.0, 2.0}` (with
`DUAL_PULSE_T_FF = VM_TAU_XY`, the approved rule), × {baseline, B2}.
= 4 × 3 × 2 × 5 = **120 runs of ~35 s** — cheap, and it reuses `run_scaling_sweep.py`'s
structure.

**Measured** (needs §4.4 first): peak `G_max`; `t_close` = time from the failure until
`G_max` first returns below 1.25 and stays; the integral of `max(0, G_max − 1.25)·dt` (the
breach *area*, the closest scalar to "escape opportunity"); and `gap_max_rad` for the
absolute reading.

**Decision rule.**

| observation | conclusion |
|---|---|
| peak `G_max` ≈ 1.92 for both methods, flat in `Vmax` and `tau_a` | the peak floor is confirmed as geometric; report it as a bound, stop trying to improve it |
| `t_close` and breach area scale with `tau_a` and are ~equal for both methods | **conjecture confirmed**: the breach window is actuation-limited, coordination cannot shorten it |
| `t_close` for B2 flat while the baseline's grows with `N` (add an N axis) | **conjecture refuted**: coordination does control the breach, and the campaign was measuring the wrong metric — `G_max` becomes the headline result |
| B2's `t_close` *worse* than the baseline's | the feedforward overshoots the tail; a real defect, and the priority changes |

The third row is the outcome that would most change the thesis, and it is not implausible:
the baseline's `Θ(N²)` relaxation should leave the gap open far longer at large N, in which
case the overlay's advantage on the mission-critical metric could be *larger* than on
`egap_avg`, not absent. The churn data cannot see this because concurrent events blur the
per-event recovery — which is precisely why the deciding experiment must be single-fault.

---

## Related

* [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) — dated record; this re-analysis is the 2026-07-26 entry.
* [PROVENANCE.md](PROVENANCE.md) — why `churn_sweep_runs_stamp/`'s B2 half is unattributable.
* [README.md](README.md) — metric and scenario definitions.
