# Campaign log — hypothesis → evidence → decision

Scientific record of the overlay (dual_pulse) campaign. Each entry: the
hypothesis, the problem it addresses, the experiment, the verdict, and the
knowledge produced — **negative results included** (they are data). Newest
entries last. Metric/scenario definitions: [README.md](README.md).

Convention for "advantage": baseline_metric / overlay_metric on the scenario's
primary metric (>1 = overlay better).

---

## Consolidated history (2026-05 → 2026-06, pre-campaign)

### H1 — "The overlay cuts the rescaling exponent" → REFRAMED
- Initial finding (fixed gain): baseline O(N), overlay O(N^0.6). REFUTED as
  fragile: the fixed gain itself destabilizes at N ≥ ~50 (limit cycle).
- With the STABLE normalized gain (K_E_TAU = 250/N): baseline is Θ(N²)
  (N^1.97, N=24..100, `baseline_long_results.csv`). The honest framing became
  the **trilemma** (stability × speed × N) — Cap. 3.

### H2 — "Option A escapes the trilemma" → REFUTED
- Option A advantage SHRINKS with N (1.68 → 1.30 → 1.14 at N=24/40/50):
  gap-bias executes THROUGH the controller gain, so no decoupling.
- Led to Option B (direct feedforward). Negative result that produced the
  central contribution.

### H3 — "B-minimal + scale=1.0 flattens tau" → REFUTED, then FIXED as B2
- B-min scale=1.0 was WORSE (double-drive over-drive). Diagnosis decomposed
  the residual: feedforward phase flat (~1.5 s), slow tail = baseline feedback
  cleaning ~25% residue.
- **B2 (full cancelling bias) + scale=1.0: tau flat ≈ 2.1 s, advantage grows
  ~N² (9→149× at N=24→100).** The trilemma is broken with no scaling penalty.
  (`largeN_results.csv`, `figure_data.csv`; TTL must be ≥ N — TTL=50 truncated
  coverage to 1% at N=100 before the fix.)

### H4 — Dimensionless law (Cap. 6)
- tau_base = 0.033·N²·(s), tau_B2 = 2.3·tau_a; A ≈ 0.014·N²/tau_a; collapses
  vs N²/tau_a (CV ~20%), NOT vs Péclet N·dt/tau_a (CV ~64%) — Pe hypothesis
  refuted. tau is dt-INVARIANT in seconds (CV < 5%, dt 0.01–0.1).
  Validity: tau_a ≥ 0.5 (tau_a = 0.2 saturates the actuator → ringing).
  (`collapse_results.csv`)

### H5 — "Overlay is fragile to packet loss" → REFUTED (was a failure-detector artifact)
- Apparent break at loss ≥ 0.1 (pre-fix `comm_results.csv`, kept as
  diagnostic). Root cause: AGENT_STATE_TIMEOUT = 5·dt → 5 consecutive lost
  messages mark a LIVE neighbor dead → storm of false SAIDA/ENTRADA;
  repeats ≥ 2 amplified the garbage (`comm_results_repeats.csv`).
- FD timeout 20·dt (0.2 s): settles at ALL loss ≤ 0.4
  (`comm_results_fix.csv`). Honest caveat: speedup degrades gracefully toward
  baseline (tau 7.8–19.5 s at loss 0.1–0.2 vs 2.17 clean; inert at 0.4).
- Knowledge: classic FD false-positive-vs-liveness trade-off; the overlay's
  loss robustness is an O(1) failure-detector parameter, not a fundamental
  limit.

### H6 — "Overlay is fragile to delay" → CONFIRMED (the open limit)
- Degrades from 5·dt (residual egap 0.015), breaks at 10·dt (egap 0.109,
  3 seeds, does not settle). Raising the FD timeout changes nothing →
  mechanism is STALE STATE feeding the open-loop feedforward, distinct from
  loss. Baseline ~immune (+11% tau at 10·dt).
  (`comm_results_delay*.csv`, `comm_results_delaytmo.csv`)
- OPEN: no mitigation attempted yet (candidates: age-stamped pulses, M6/M7/M9).

### H7 — "Dense churn breaks the overlay" → REFUTED (was the trigger, not the overlay)
- Pre-refactor disaster (adv 0.48 at 12/min) was caused by the GLOBAL
  alive_count trigger violating the neighbor-only premise.
- Explored remedies — all DISCARDED after the root-cause fix: binary gate
  (now HURTS: 0.80–0.90), M2 stamped-N (violates premise; now dead code),
  M5 idempotent (loses simultaneous-drop accumulation), conditional
  accumulation (worst).
- **Premissa-limpo trigger (succ-freshness classification): the ORIGINAL
  additive overlay helps under churn — adv 1.42/1.21/1.02/0.96 at
  6/12/24/48 per min, 8 seeds, never harmful** (`churn_sweep_results.csv`,
  promoted from `_add_clean8`).
- Knowledge: with hop-propagated information, freshness discipline at the
  TRIGGER is what matters; suppressing the overlay (gate) throws away help.

### H8 — "consume_motion eats the shift under maneuver" → CONFIRMED + FIXED (M8)
- Under maneuver the measured Δθ includes tracking rotation, which consumed
  the redistribution shift → under-redistribution.
- M8 (DUAL_PULSE_CONSUME_FF_ONLY, now default): consume only the
  FF-commanded rotation. Maneuver+fail: B2 0.0546 → 0.0485 ≤ baseline 0.0499
  (3/3 seeds); constant motion: no regression; churn+maneuver rises to
  1.16–1.20. (`trackC_results_m8clean.csv`, `trackC_results_churnm8.csv`)
- Residual: maneuver benefit is DILUTED (pursuit error dominates) — a
  ceiling, not a failure.

### Robustness map closure (Track C)
- Tracking (E_r) NEVER degraded by the overlay in any cell (tangential-only).
- Controlled ENTRADA works (recover: 1.88×, `trackC_results_recover.csv`).
- Neighbor-only premise PROVEN with 25 m range ≡ global
  (`trackC_results_srange.csv`).
- Combined stress (churn 18/min + loss 0.1 + delay 0.02 + maneuver + M8):
  overlay helps 1.10–1.15 (`trackC_results_stress.csv`).
- Known residual weaknesses: ADJACENT simultaneous faults sub-correct
  (tau 10.5/14 s vs 2.2; second originator dead → event lost; baseline
  closes it slowly); dead-canonical ENTRADA missed (~3/24 dense).

---

## 2026-06-12 — Campaign kickoff: evidence audit + Ciclo 0 (consolidation)

**Problem.** (a) The repo-committed campaign CSVs were STALE — they still
carried the refuted pre-fix results (churn adv 0.48; trackC pre-M8; comm
pre-FD-fix) while the current evidence lived only in the local archive: a
reproducibility hazard. (b) The repo default ran legacy Option A, not the
thesis B2. (c) The B/B2 feedforward path had zero unit tests. (d) Campaign
metrics lacked effort/saturation/fairness/overshoot and worst-case
aggregation. (e) dt=0.05 (requested as default) had red flags nobody had
examined.

**Decisions (user-approved):** D1 B2-locked defaults in `config_param.py`;
D2a promote archived clean CSVs as canonical; D3 this docs tree; D4 cycle
order 0 → 1 (dt) → 2 (delay) → 3 (adjacent faults).

**Changes (Ciclo 0):**
- `config_param.py`: defaults now `INTEGRATION=B2`, `DELTA_SCALE`
  mode-dependent (1.0 B/B2, 0.5 A), `TTL=max(50, 3N)`, `K_E_TAU=250/N`,
  `T_FF=VM_TAU_XY`. At the default N=10/tau_a=1 every numeric value is
  unchanged; env overrides intact; verified by import tests on 5 paths.
- `protocol_agent.py`: pure-helper extractions `_compute_cancelling_bias`,
  `_compute_ff_command` (behavior-preserving) + 10 unit tests including the
  closed-loop exp-decay property (e^-1 after one T_FF through the real
  `consume_motion`).
- `run_comm_sweep.py`: FD-fix (timeout 20·dt) applied by default and recorded
  per-row (`agent_state_timeout` column — provenance was previously only in
  shell history).
- `metrics_util.py`: + `overshoot_frac` (band-relative; the naive
  asymptote-relative definition was wrong — it reported ~band_frac for
  perfectly monotone decays, caught by the new golden tests), +
  `effort_metrics` (M5/M6/M2 from agent telemetry, computed before the
  runners delete it), + `aggregate_seeds` (median/worst/std). 13 unit tests
  (module previously had zero pytest coverage). Churn/trackC/comm runners now
  record effort/saturation/fairness per row; churn summary adds worst-case
  advantage (baseline's best seed vs overlay's worst).
- Sample re-verification of the archived CSVs before promotion (1 cell per
  family re-run with current code, tagged) — see verdict below.
- Docs: this tree created; CLAUDE.md/README updated to B2 default; archive
  docs' stale sections fixed (tese_estrutura Fase 3 checklist;
  plano_overlay_robusto_v2 bottom STATUS contradicted its own resolution
  banner).

**dt=0.05 red flags registered for Ciclo 1 (from the audit, all verified
against data):**
1. tau is dt-invariant BUT B2 `egap_late_std` grows ~100× for dt ≥ 0.02
   (4.2e-4 → 5.2e-2): regime jitter is NOT dt-invariant.
2. FD timeout is tick-denominated (5·dt default): at dt=0.05 the default
   tolerates 5 lost messages — exactly the pre-fix vulnerable configuration
   in message counts.
3. The delay breakpoint (10·dt at dt=0.01) has unknown denomination
   (ticks vs seconds) — at dt=0.05 it is either 2 ticks (0.1 s) or 0.5 s.
   Scientifically interesting either way.
4. Loss/delay/churn/maneuver/recovery have NEVER run at dt=0.05 (only one
   collapse cell, 1 seed).
5. Tick-denominated knobs silently change meaning (RAMP_TICKS=4 → 0.2 s;
   BROADCAST_REPEATS=2 → 0.1 s exposure); `run_trackC.py` hardcodes
   timeout/delay values calibrated for dt=0.01.

**Sample re-verification verdict (1 cell per family, tagged runs, current
code vs archived CSVs):**

| Cell | Re-run | Archive | Verdict |
|---|---|---|---|
| trackC fail+maneuver seed0 B2 (egap_avg / Er_avg) | 0.047038 / 0.028914 | 0.047038 / 0.028914 | **bit-exact** |
| comm loss 0.2 seed0 B2 (egap_final / tau / settled) | 0.000119 / 19.4985 / True | 0.000119 / 19.4985 / True | **bit-exact** |
| churn rate12 seed0 B2 (egap_avg) | 0.0751 | 0.1442 | mismatch → diagnosed |
| churn rate12 seed0 B2 with `CONSUME_FF_ONLY=False` | **0.1442** | 0.1442 | **bit-exact** — mismatch explained |

Diagnosis: the archived churn evidence (`add_clean8`) predates the M8 default
(M8 was OFF in those runs); the churn runner does not pin the flag, so the
re-run used the current default (M8 ON). No code drift: with the flag matched,
reproduction is exact.

**NEW FINDING (serendipitous): M8 also helps stationary churn — B2 egap at
rate 12 nearly HALVES (0.1442 → 0.0751, seed 0).** M8 was designed for the
maneuvering target (don't consume tracking rotation), but under churn the
agents' own redistribution+controller rotation was likewise eating shifts.
Consequence: the canonical churn reference must be regenerated under current
defaults (M8 on) — folded into Ciclo 1 together with the dt axis. Hypothesis
to confirm there: the M8-on advantage curve dominates the M8-off one at all
rates and seeds.

**CSV consolidation executed (D2a):** repo `churn_sweep_results.csv` ←
`add_clean8` (M8-off provenance noted in the index); stale pre-trigger-fix
churn CSV archived as `churn_sweep_results_pre_trigger_fix.csv`; promoted 6
trackC files (m8clean/churnclean/churnm8/recover/srange/stress), 7 comm files
(fix/loss_clean/delay×4/repeats), 3 churn supporting files
(add_clean/over_clean/gate_clean). `comm_results.csv` and `trackC_results.csv`
remain as labeled diagnostics.

**Ciclo 0 status: COMPLETE.** Test suite: 141 passing (118 → +10 FF-path,
+13 metrics). Next: Ciclo 1 (dt=0.05 validation + fresh M8-on churn reference).

---

## 2026-06-13 — Ciclo 1: dt=0.05 validation + M8 generalizes to delay (control-confirmed)

Grid (all tagged `c1*`, canonical CSVs untouched): A lei-limpa dt {0.01,0.05};
D delay denomination; E loss×FD-timeout-in-ticks; C churn dt=0.05; B M8-on
churn reference dt=0.01; plus two control cells.

### H-M8-delay (NEW) — "M8 also fixes the delay degradation" → CONFIRMED by control
- Problem: the one ACTIVE limit of the robustness map was delay > 5·dt
  (H6: B2 breaks at 10·dt, egap 0.109). That was measured with M8 OFF.
- Control experiment (delay=0.1, dt=0.01, B2, only DUAL_PULSE_CONSUME_FF_ONLY
  differs): **M8 OFF → egap 0.10870, NOT settled (reproduces the archived
  break exactly); M8 ON → egap 0.00004, settled, tau 3.24.** Decisive.
- Mechanism: under pure delay the pulses (event-triggered) arrive late but
  intact → δ_D correct → the feedforward is correct; the ONLY victim was
  `consume_motion` deducting the stale-measured Δθ. M8 deducts only the
  FF-COMMANDED rotation (computed locally, delay-immune) → correct
  consumption → clean settle (slightly later start: tau 3.24 vs 2.22 clean).
- Knowledge: M8 is not a maneuver patch — it is the GENERAL consume_motion
  correctness fix, and it resolves three regimes at once (maneuver + dense
  churn + comm delay). This UNIFIES the robustness story and CLOSES the only
  open active limit. **Action: cap7 §7.2.3 (delay) must be rewritten — the
  delay "limit" was an M8-off artifact; with the shipped default (M8 on) B2
  degrades GRACEFULLY (seconds-denominated), it does not break.**
- Block D (M8 on, dt=0.05): delay {0.1, 0.25, 0.5} s → B2 tau 3.12 / 4.80 /
  8.87, all settled, egap ≤ 3e-4. So the old "10·dt breakpoint" is GONE; the
  residual is a smooth physical slowdown in SECONDS (more delay = later start),
  not a tick cliff → flag nº3 (ticks vs seconds) resolved: seconds, no cliff.

### H-dt05 — "dt=0.05 preserves the conclusions" → CONFIRMED, with ONE coupled caveat
- **τ dt-invariant (Block A):** baseline 19.48 → 20.28 s, B2 2.22 → 2.15 s
  (dt 0.01 → 0.05); advantage 8.79 → 9.42. ✓
- **Regime jitter dt-invariant (CTRL-2, clean 60 s budget):** B2 egap_late_std
  = 0.00025 at BOTH dt. The Block-A short-budget late_std (0.047 / 0.065) and
  the audit's "100×" were WINDOW ARTIFACTS — the 20 s late-window of a ~20 s
  run includes the post-fault transient. Flag nº1 DISMISSED. ✓
- **Churn survives dt=0.05 (Block C):** advantage 1.21 / 1.14 at rates 12/48
  vs 1.30 / 1.21 at dt=0.01 — modest degradation, NO qualitative inversion. ✓
- **Delay (Block D):** no break at dt=0.05 (see H-M8-delay). ✓
- **LOSS — flag nº2 CONFIRMED (Block E):** loss 0.2, dt=0.05: FD timeout
  5·dt=0.25 s → NEITHER baseline NOR B2 settles (egap 0.002–0.031, settled
  False ×3 each); FD timeout 20·dt=1.0 s → both settle (egap ~1e-4). The FD
  requirement is TICK-denominated (tolerate ~20 consecutive losses), NOT
  absolute-seconds: 0.25 s = only 5 ticks at dt=0.05 → ~1 false-positive
  window over a 150 s run at loss 0.2 (0.2^5·3000 ≈ 1). This also hits the
  BASELINE (neighbor liveness uses the same timeout), so it is not
  overlay-specific.
- **VERDICT:** dt=0.05 preserves every qualitative conclusion AND gives ~5×
  faster sims (5× fewer ticks/s) — **on the condition** that the
  `AGENT_STATE_TIMEOUT` default is decoupled from `5·dt`. Proposed coupled
  change: default `max(20·dt, 0.2)` (= 0.2 s at dt=0.01, the campaign's
  validated FD-fix; = 1.0 s at dt=0.05, what Block E showed is needed). This
  is a behavior change (a real dead node is declared dead ~0.15 s later at
  dt=0.01) → requires a clean-scaling-law re-check before adoption. DECISION
  PENDING (user).

### M8-on churn reference (Block B) — PROMOTED to canonical
- Advantage 1.40 / 1.30 / 1.24 / 1.21 at rates 6/12/24/48 (3 seeds, dt=0.01).
- vs M8-OFF (the previous canonical, add_clean8, 8 seeds): 1.42 / 1.21 /
  1.02 / 0.96. **M8 dominates at moderate/dense rates** — at 48/min the
  overlay goes from slightly harmful (0.96) to helpful (1.21). Confirms the
  Ciclo-0 serendipitous finding with a full rate sweep.
- `churn_sweep_results.csv` ← Block B (M8-on, current config). The M8-off
  8-seed run is archived as the M8 ablation
  (`churn_sweep_results_m8off_ablation8seed.csv`). Caveat: Block B is 3 seeds;
  an 8-seed M8-on confirmation is a cheap follow-up.
- Effort cost recorded: B2 control effort ~2× baseline under churn
  (eff_B2/bs ≈ 1.9–2.2 across rates) — the overlay's speed is bought with
  more actuation; logged as a trade-off, not hidden.

**Ciclo 1 status: COMPLETE.** Knowledge produced: (1) M8 is the general
consume_motion fix (closes the delay limit) — strongest result of the cycle;
(2) dt=0.05 is adoptable as default with a coupled FD-timeout change; (3) two
audit "flags" (late_std explosion, delay tick-cliff) were artifacts, now
dismissed with controls; (4) the FD-robustness requirement is fundamentally
tick-denominated. Test suite unchanged (141). Next (pending user): adopt
dt=0.05 default + FD-timeout change (with clean re-check), then Ciclo 2 was
"delay mitigation" — now largely SUBSUMED by H-M8-delay, so Ciclo 2 becomes
the ADJACENT-faults / successor-fallback front (the remaining sub-correction).

### D5 applied (2026-06-13) — dt=0.05 default + FD-timeout + runner consistency
User-approved decisions: dt=0.05 as the repo default; AGENT_STATE_TIMEOUT
default `max(20·dt, 0.2)`; with a clean re-check. Applied:
- `config_param.py`: CONTROL_PERIOD default 0.01→0.05; AGENT_STATE_TIMEOUT
  default `5·dt`→`max(20·dt, 0.2)` (= 0.2 s at dt=0.01, 1.0 s at dt=0.05);
  both documented as deliberate changes with rationale.
- **Re-check found a real issue I introduced:** the campaign runners read `DT`
  with a 0.01 fallback (for budget/Pe/label/FD-timeout) but did NOT pin
  CONTROL_PERIOD in the child, so with the new global default they ran at 0.05
  while labeling/budgeting for 0.01 (verified via timestamp deltas). Worse,
  `run_comm_sweep` pinned a 0.2 s FD timeout = only 4 ticks at dt=0.05 →
  loss-fragile. Also the collapse baseline budget was `14·N²·dt` (scales with
  dt) → over-budgeted 5× at dt=0.05, erasing the speedup.
- **Fix:** all 5 runners (collapse, churn, comm, trackC, baseline_long) now
  PIN CONTROL_PERIOD in the child (label == reality) and derive the FD timeout
  from the actual dt: `5·dt` for loss-free runners (fast clean detection, no
  false positives) and `max(20·dt, 0.2)` for loss-facing ones
  (comm + trackC loss/stress). Collapse baseline budget made dt-invariant
  (`0.14·N²` s).
- **Re-check v2 (corrected runners, dt=0.05 pinned):** baseline τ 20.28 / 88.11
  s (N=24/50), B2 τ 2.15 / 3.15 s, advantage 9.42 / 27.98 — matches the
  dt=0.01 reference (N=24 B2 2.15 ≡ 2.16). comm smoke loss 0.2 at default dt:
  runner auto-set FD=1.0 s → both methods settle. **dt=0.05 VALIDATED + runner
  campaign self-consistent.** Residual (minor, NOT a dt effect): collapse-runner
  B2 τ at N=50 ≈ 3.1 vs the largeN-runner's 2.1 — a budget/window difference
  between runners, present at both dt; confirm with a collapse N=50 dt=0.01 if
  it ever matters for a figure. Detection-latency note: with the loss-robust
  1.0 s timeout the absolute recovery gains ~1 s at dt=0.05 (the loss-free
  runners use 5·dt=0.25 s to avoid this in clean measurements).

---

## 2026-06-13 — Ciclo 2 (in progress): adjacent simultaneous faults / successor fallback
Target: the remaining ACTIVE weakness — when ADJACENT agents fail together the
overlay sub-corrects (historical τ 10.5/14 s vs 2.2 s non-adjacent), because the
coordination rule lets ONLY the canonical originator (dead drone's predecessor)
inject, and for the 2nd of two adjacent deaths that predecessor is itself dead →
the event is never injected (also the dead-canonical ENTRADA, ~3/24 dense).
Plan: (1) reproduce + instrument the failure mode with CURRENT code (the trigger
was refactored to neighbor-only since the historical measurement); (2) form the
hypothesis; (3) design a fix; (4) test baseline/B2/B2+fix on adj2/adj3 with
non-adjacent + single-fault + churn as sentinels; (5) accept only if
τ(adj)→~τ(non-adj) with no sentinel regression.

### Step 1 reproduction (current code, neighbor-only trigger, dt=0.05, 5·dt timeout)
diag_churn det (B2), N=24, single permanent simultaneous fault block:
| scenario | deaths | inj | N_new | t_settle | tau |
|---|---|---|---|---|---|
| k1   | 1 | 1 | 23 ✓ | 7.30  | 2.14 |
| adj2 | 2 | **1** | 22 ✓ | 29.30 | 13.57 |
| adj3 | 3 | **1** | 21 ✓ | 34.05 | 15.45 |
| non2 | 2 | 2 | 22 ✓ | 6.60  | 1.94 |
| non3 | 3 | 3 | 21 ✓ | 7.95  | 2.46 |

**Refined mechanism (sharper than the historical "2nd originator dead" note):**
the adjacent block fires only ONE pulse (the predecessor of the FIRST dead drone
survives and injects; the predecessors of the others are themselves dead), BUT
the firing pulse reads the CORRECT N_new (22/21) via hop-count traversal. The
slowness is a **MAGNITUDE under-correction**: the δ formula assumes a SINGLE
removal (`N_old = N_new + 1`), so `gap_old = 2π/(N_new+1)`, while k drones were
removed (true `N_old = N_new + k`). The feedforward therefore delivers ~1/k of
the needed shift; the slow O(N²) baseline cleans the residual → τ 13.6/15.5 s
(between the full-FF 2.1 s and full-baseline ~20 s; egap_final stays small
0.001–0.0015 = baseline closes it, slowly). NOT a wrong-N bug, NOT (only) a
missing-injection bug — a multiplicity-unaware MAGNITUDE bug in the surviving
originator's δ.

### Hypothesis H-mult and the proposed fix (DESIGN — pending user approval before code)
H-mult: the surviving canonical originator can infer the death multiplicity k
from its OWN post-event succ_gap (≈ (k+1)·ideal_gap, since k frozen dead drones
lie between it and its new successor) — neighbor-only — and stamp k on the pulse;
with the δ formula using `N_old = N_new + k` (gap_old = 2π/(N_new+k)) the single
firing pulse delivers the FULL k-removal redistribution. Reduces EXACTLY to
current behavior at k=1. Predicted: τ(adj2/adj3) → ~τ(non2/non3) ≈ 2 s.

### Implementation (M-mult, flag DUAL_PULSE_MULTIPLICITY, default off for the A/B)
- `config_param.py`: `DUAL_PULSE_MULTIPLICITY` (bool) + `DUAL_PULSE_MAX_MULTIPLICITY`
  (clamp, default 6).
- `dual_pulse_layer.py`: `inject_pulse(multiplicity=k)` stamps `k`; SAIDA receiver
  and originator self-shift use `n_old = n_new + k`; `k` propagated on forward.
- `protocol_agent.py`: at SAIDA injection, infer `k = round(succ_gap/desired_gap) - 1`
  (neighbor-only, clamped), pass as multiplicity.
- Tests (4 new, suite 141→145): k=1 byte-identical to legacy (regression);
  missing-`k`-field == k=1 (back-compat); k=2/3 match the analytic `n_old=n_new+k`
  delta and grow in magnitude; originator self-shift uses k. Algebra LOCKED before sims.

### Benchmark — adjacent SAIDA (diag_churn det, B2, dt=0.05, 5·dt timeout): M-mult OFF vs ON
| scenario | τ OFF | τ ON | t_settle OFF→ON | verdict |
|---|---|---|---|---|
| k1   | 2.14  | 2.14 | 7.30 → 7.30  | identical (k=1) |
| adj2 | 13.57 | **2.17** | 29.30 → **7.50** | **FIXED (6.3×)** |
| adj3 | 15.45 | **2.21** | 34.05 → **7.75** | **FIXED (7×)** |
| non2 | 1.94  | 1.94 | 6.60 → 6.60  | identical (k=1) |
| non3 | 2.46  | 2.46 | 7.95 → 7.95  | identical (k=1) |

H-mult CONFIRMED: adjacent-block sub-correction repaired (τ → ~τ(non-adjacent));
sentinels byte-identical (k=1 path untouched, as the regression test guarantees).
N_new stays correct; still one injected pulse, now with the right multiplicity →
full redistribution. The historical "adjacent failures sub-correct" weakness is
closed for the discrete-event regime.

### Churn safety sentinel (M-mult ON vs OFF, dt=0.05, rates 12/48, 3 seeds)
| rate | advantage OFF | advantage ON | B2 worst-case egap OFF→ON |
|---|---|---|---|
| 12 | 1.21 | 1.21 | 0.1214 → 0.1214 |
| 48 | 1.14 | 1.14 | 0.2832 → **0.2803** |

NO regression under churn: the k-inference does NOT misfire on gradual/noisy gaps
(churn deaths are mostly spread → inferred k≈1; the k>1 path only engages on a
genuine adjacent block). Advantage identical, worst-case equal-or-better.

**Ciclo 2 verdict: M-mult ACCEPTED.** Acceptance criteria all met: (1) fixes the
target (adj2/adj3 τ 13.6/15.5→2.2 s, ~6–7×); (2) zero regression on sentinels
(k1/non2/non3 byte-identical, churn identical/better); (3) holds across discrete
AND churn regimes; (4) worst-case across seeds not worsened; (5) neighbor-only,
parameter-light, k=1≡legacy; (6) algebra locked by unit tests, 145 pass.
Knowledge produced: the adjacent sub-correction was a MAGNITUDE bug (correct
N_new, wrong N_old=N_new+1), not a missing-injection or wrong-N bug — fixable
neighbor-only by inferring the death multiplicity from the originator's own gap;
no successor-fallback/timer needed. The dead-canonical ENTRADA (~3/24 dense)
remains out of scope (only a successor fallback would cover it) — deferred.
Default decision: **DEFAULT-ON** (user-approved 2026-06-13) —
`DUAL_PULSE_MULTIPLICITY=True`, env-disablable for the ablation. Committed with
Ciclos 1+2.

Evidence: `churn_sweep_results_c2mmult_churn.csv` (M-mult-on churn sentinel);
the adjacent-block benchmark is in diag_churn output (det c2repro/c2mmult, not
persisted as a CSV — diag_churn prints). Open item: dead-canonical ENTRADA
(~3/24 dense) still uncovered (would need a successor fallback) — deferred to a
future cycle (user 2026-06-13: not worth the cost/risk).

## 2026-06-13 — 8-seed confirmation of the M8-on churn reference (user-requested)
Goal: bump the M8-on churn reference from 3 seeds to 8 seeds and report the
WORST case, not just the median. Run at the new default config (dt=0.05, M8 on,
M-mult on), rates 6/12/24/48, seeds 0–7 (`churn_sweep_results_c3_churn8_dt05.csv`).

**Methodology fix (lesson):** the per-rate summary's old `adv_worst` column
(best-baseline-seed / worst-B2-seed) is MISLEADING for stochastic churn — it
pairs DIFFERENT failure streams. Since baseline and B2 share EXPERIMENT_SEED
(= same Poisson stream), the honest metric is the PAIRED per-seed advantage
egap_base(s)/egap_B2(s). Fixed `run_churn_sweep.py` to report paired
adv_med / adv_min / n_lose.

**Result (paired by seed, 8 seeds):**
| rate | adv_med | adv_min | adv_max | seeds lost | help |
|---|---|---|---|---|---|
| 6  | 1.31 | 1.24 | 1.34 | 0 | 8/8 |
| 12 | 1.23 | 1.14 | 1.30 | 0 | 8/8 |
| 24 | 1.15 | 1.11 | 1.20 | 0 | 8/8 |
| 48 | 1.14 | 1.11 | 1.18 | 0 | 8/8 |

**The overlay helps on EVERY one of the 8 seeds at EVERY rate** (adv_min ≥ 1.11;
zero seeds where it loses). The unpaired "worst-case < 1" that appeared first
(0.66–0.92) was purely a seed-mismatch artifact (some Poisson streams are
intrinsically harder — high CV 23%@rate6 — but the overlay beats baseline on
each given stream). Magnitudes are the dt=0.05 values (slightly below Block B's
dt=0.01 1.40/1.30/1.24/1.21 — the tick-denominated detection latency), and
robust across 8 seeds. **CONFIRMED.** Promote `churn_sweep_results.csv` to the
8-seed dt=0.05 reference; keep the 3-seed dt=0.01 Block B as historical.
Effort cost re-confirmed ~2.2–2.7× baseline actuation (logged trade-off).

Next research front options (none chosen yet): dead-canonical ENTRADA fallback
(deferred); hardware/SITL (Cap. 8); larger-N robustness.
