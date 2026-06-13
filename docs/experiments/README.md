# Experimental campaign — structured test documentation

This is the reproducibility contract for the `dual_pulse` overlay research
campaign: the locked configuration, the scenario definitions, the metric
definitions, the acceptance criteria, and the evidence index mapping each
scientific claim to the CSV that supports it.

The hypothesis-by-hypothesis history (including negative results) is in
[CAMPAIGN_LOG.md](CAMPAIGN_LOG.md). Portuguese thesis prose lives in
`docs/thesis/` (gitignored — local/IDE only, not part of the public repo).

---

## 1. The three systems under comparison

| Name | What it is | How to run it |
|---|---|---|
| **baseline** | Local two-channel tangential controller only; self-stabilizing; re-stabilizes a single fault in Θ(N²) (measured N^1.97, N=24..100) | `PROPAGATION_METHOD=baseline` |
| **overlay (current)** | `dual_pulse` + **B2** 2-DOF feedforward (locked config below); flat reconfiguration ~2·T_FF up to N=100 | `PROPAGATION_METHOD=dual_pulse` (B2 is the default integration) |
| **overlay (modified)** | Any candidate change under test; always benchmarked against BOTH of the above in the same grid | per-experiment env flags |

## 2. Locked overlay configuration (the "thesis B2 config")

Since 2026-06 these are the **repository defaults** (`config_param.py`); the
campaign runners also pin them explicitly so results never depend on defaults:

```
DUAL_PULSE_INTEGRATION = B2          # 2-DOF feedforward, full cancelling bias
DUAL_PULSE_DELTA_SCALE = 1.0         # full analytical shift (mode-dependent default; A uses 0.5)
DUAL_PULSE_T_FF        = VM_TAU_XY   # T_FF = tau_a rule (c_FF = 1.0)
DUAL_PULSE_TTL_HOPS    = max(50, 3N) # must be >= N or coverage truncates
K_E_TAU                = 250 / N     # stable normalized gain (fixed gain destabilizes N >~ 40)
DUAL_PULSE_CONSUME_FF_ONLY = True    # M8: consume only the FF-commanded rotation
DUAL_PULSE_MULTIPLICITY    = True    # M-mult: k-aware δ for adjacent-block failures (k=1 ≡ legacy)
Trigger: neighbor-only (succ-freshness classification; no global alive_count)
```

Discarded-but-kept flags (all default off; see log entries): `DUAL_PULSE_GATE_*`,
`DUAL_PULSE_USE_STAMPED_N` (M2), `DUAL_PULSE_IDEMPOTENT` (M5),
`DUAL_PULSE_ADD_IF_SETTLED`.

## 3. Scenarios and their runners (`experiments/scaling_law/`)

| Scenario | Runner | Key env knobs | Notes |
|---|---|---|---|
| Single deterministic fault (scaling law) | `run_baseline_longbudget.py`, `run_largeN_confirm.py`, `run_collapse_sweep.py` | `COLLAPSE_N_VALUES`, `COLLAPSE_TAU_VALUES`, `CONTROL_PERIOD` | victim `2+((N//2+seed)%N)`, crash at t0=5 s, permanent |
| Dimensionless law (N × tau_a × dt) | `run_collapse_sweep.py` | as above | A ≈ 0.014·N²/tau_a; valid tau_a ≥ 0.5 |
| Churn (Poisson, temporary faults) | `run_churn_sweep.py` | `CHURN_RATES` (total/min), `CHURN_OFF`, `CHURN_SEEDS` | continuous churn → `egap_avg`, no settling |
| Packet loss / redundancy | `run_comm_sweep.py` | `COMM_LOSS_VALUES`, `COMM_REPEATS`, `COMM_FD_TIMEOUT` | FD-fix (timeout 20·dt) applied by default and recorded per-row |
| Delay | `run_comm_sweep.py` | `COMM_DELAY` (one value per invocation; use `COMM_TAG`) | overlay's open limit: breaks at 10·dt (dt=0.01) |
| Moving target (constant / maneuver) × stresses | `run_trackC.py` | `TRACKC_SCENARIO` ∈ none, fail, loss, delay, churn_sparse, churn_dense, recover, stress; `TRACKC_MOTION` | `recover` = controlled ENTRADA; `stress` = churn+loss+delay |
| Simultaneous multi-victim faults | `diag_churn.py` | `DETERMINISTIC_FAILURE_AGENT_ID` accepts a comma list | adj2/adj3 vs non2/non3 |

Gaps (no runner yet): stale-message injection per se; parametric target
trajectories (circular / zigzag / step); churn at N ≠ 24; delay grid in one
invocation.

## 4. Metrics (`experiments/scaling_law/metrics_util.py`)

| Metric | Meaning | Use |
|---|---|---|
| `t_settle` (+`egap_settle`) | enter-and-STAY settling time, band = max(5% peak, 3σ noise); pair tells "how fast" + "did it actually reconfigure" | events (fault/recovery) |
| `tau_fit` (+R²) | exp-fit time constant of the decay tail | secondary; only trust R² ≥ 0.9 |
| `egap_avg` | steady-state mean E_gap (t ≥ t0+10..15) | continuous churn (no settling exists) |
| `egap_peak`, `egap_final`, `egap_late_std` | disturbance peak / end value / late jitter | late_std is the dt-sensitivity sentinel |
| `overshoot_frac` | re-excursion beyond the settle band after first entry, normalized by event amplitude; 0 = monotone | ringing detector |
| `effort_mean_v2`, `sat_frac` | mean (v/Vmax)² and Pr(v ≥ Vmax) from agent telemetry (M5/M6 style) | control effort / saturation |
| `fairness_p95` | P95 over nodes of per-node P95\|e_tau_real\| (M2 style) | fairness across agents |
| `aggregate_seeds` | median + worst (max) + std across seeds | never report central tendency alone |
| `Er_avg`, `Evr_avg` | radial tracking error (target telemetry) | overlay must NEVER degrade these |
| message counts | `dual_pulse_messages.csv` payload counts | O(N) total, O(1)/agent |

Per-run M1..M7 (`plot_telemetry.py`) remain available in each run dir's
`runs_summary.csv`.

## 5. Acceptance criteria for any overlay change

1. ≥ 3 seeds (8 where variance is high); fixed seeds; deterministic harness.
2. Improves the TARGET cell on the primary metric for that hypothesis.
3. No regression > 5% (median) on sentinel cells: single-fault N=24,
   churn 12/min, fail+maneuver; AND no worsening of the across-seed worst case.
4. `Er_avg` (tracking) unchanged.
5. Effort/saturation not significantly worse unless explicitly traded off.
6. `pytest` green + a unit test for the change itself.
7. Result recorded in CAMPAIGN_LOG.md **even when rejected**.

## 6. Evidence index (which CSV answers which claim)

Status: **canonical** = current best evidence; **diagnostic** = kept because it
documents a diagnosis (superseded as a result); **archived** = local archive
(`_soliton_v3_local_archive/experiments_variants/`), superseded exploration.

| Claim | File (in `experiments/scaling_law/`) | Status |
|---|---|---|
| Baseline Θ(N²): tau = 19.5/54.8/85.4/183.6/311.4 s at N=24/40/50/75/100 (fit N^1.97) | `baseline_long_results.csv` | canonical |
| B2 flat tau ≈ 2.09–2.12 s to N=100 (TTL=3N, 2 seeds) | `largeN_results.csv` | canonical |
| Speedup 9→149× (N=24→100); figure data | `figure_data.csv` | canonical |
| Dimensionless law A ≈ 0.014·N²/tau_a; dt-invariance of tau (CV < 5%, dt 0.01–0.1) | `collapse_results.csv` | canonical |
| Churn (current config, M8 on): overlay helps at all rates, adv 1.40/1.30/1.24/1.21 @ 6/12/24/48 per min (3 seeds, dt=0.01) | `churn_sweep_results.csv` (← Ciclo 1 Block B) | canonical |
| Churn M8 ablation (M8 OFF, 8 seeds): adv 1.42/1.21/1.02/0.96 — shows M8 turns dense churn from slightly harmful (0.96) to helpful (1.21) | archived `churn_sweep_results_m8off_ablation8seed.csv` | ablation |
| **M8 also fixes comm DELAY** (control: delay 0.1/dt 0.01/B2 → M8 OFF egap 0.109 broken, M8 ON egap 4e-5 settled). M8 = general consume_motion correctness fix (maneuver+churn+delay) | `comm_results_c1Dctrl_m8off.csv` + `comm_results_c1D_*.csv` | canonical (Ciclo 1) |
| dt=0.05 validation: τ dt-invariant; regime jitter dt-invariant (late_std 2.5e-4 both dt, clean budget); churn still helps; delay graceful | `collapse_results_c1A_*.csv`, `collapse_results_c1Along_*.csv`, `churn_sweep_results_c1C_dt05.csv` | canonical (Ciclo 1) |
| dt=0.05 LOSS caveat: FD timeout is tick-denominated — 5·dt (0.25 s) fails under loss 0.2 for BOTH methods; 20·dt (1.0 s) fixes it | `comm_results_c1E_dt05_tmo5t.csv`, `comm_results_c1E_dt05_tmo20t.csv` | canonical (Ciclo 1) |
| Churn pre-trigger-fix (adv 0.48 disaster at 12/min) — the refuted result | archived `churn_sweep_results_pre_trigger_fix.csv` | diagnostic |
| Loss ≤ 0.4 settles with FD timeout 0.2 s (graceful fallback; speedup shrinks 0.1–0.2, inert at 0.4) | `comm_results_fix.csv`, `comm_results_loss_clean.csv` | canonical |
| Loss vulnerability WITHOUT FD fix (B2 breaks at 0.1) — diagnosis artifact | `comm_results.csv` | diagnostic (pre-fix; do not cite as current behavior) |
| repeats ≥ 2 amplifies FD false positives under loss | `comm_results_repeats.csv` | diagnostic |
| Delay (M8 OFF, historical): degrades at 5·dt, breaks at 10·dt (egap 0.109); mechanism = stale state, not FD timeout | `comm_results_delay*.csv`, `comm_results_delaytmo.csv` | **superseded** — the break was an M8-OFF artifact; see the M8-fixes-delay row above |
| Moving target: M8 fixes maneuver (0.0485 ≤ baseline 0.0499); benefit preserved under constant motion | `trackC_results_m8clean.csv` | canonical |
| Churn+motion (no gate): overlay helps (1.42/1.26 const; with M8 1.16–1.20 maneuver) | `trackC_results_churnclean.csv`, `trackC_results_churnm8.csv` | canonical |
| Controlled ENTRADA (recovery) works: 1.88× | `trackC_results_recover.csv` | canonical |
| Neighbor-only premise PROVEN at 25 m range (≡ global) | `trackC_results_srange.csv` | canonical |
| Combined stress (churn+loss+delay+maneuver): overlay still helps (1.10–1.15) | `trackC_results_stress.csv` | canonical |
| Track C pre-M8/pre-refactor grid | `trackC_results.csv` | diagnostic — single-event cells valid; churn & maneuver cells SUPERSEDED by the files above |
| Gate / stamped-N / idempotent / conditional explorations | archived `*_gate*`, `*_stamp*`, `*_idem*`, `*_cond*` | archived (discarded mechanisms) |
| Option A vs B vs B2 ablation | `optionB_results.csv` (+archived variants) | canonical (ablation) |
| Agility / gain / delta-scale sweeps (Option-A era) | `agility_results.csv`, `gain_results.csv`, `deltascale_results.csv` | diagnostic (pre-B2 tuning studies) |

## 7. Reproduction quickstart

```powershell
$env:PYTHONIOENCODING = "utf-8"
# Scaling law cell (B2, N=24):
python experiments/scaling_law/run_collapse_sweep.py   # see COLLAPSE_* env knobs
# Churn sweep (canonical grid):
python experiments/scaling_law/run_churn_sweep.py      # CHURN_RATES="6,12,24,48", 3 seeds
# Loss sweep with FD-fix (default):
python experiments/scaling_law/run_comm_sweep.py
# Moving-target scenarios:
$env:TRACKC_SCENARIO = "fail"; $env:TRACKC_MOTION = "both"
python experiments/scaling_law/run_trackC.py
```

All runners: deterministic seeds, incremental CSV merge (re-running skips
completed cells), per-run dirs under `*_runs*/` (gitignored), `*_TAG` env to
fork an output CSV without touching the canonical one.

Known reproduction caveats: run with the repo as cwd; simulations are
CPU-bound (large-N baseline runs take minutes-hours); run dirs can fill the
disk — the runners delete `agent_telemetry.csv` per run after computing
effort metrics.
