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

## 2026-06-13 — Stale / out-of-order messages (front 2): diagnosis = already handled
Question for Cap. 7: is the overlay robust to STALE messages — i.e., an
AgentState carrying state OLDER than one already processed from the same sender
(reordering / out-of-order delivery), as opposed to LOSS (never arrives) or
uniform DELAY (all arrive late, characterized in §7.2.3)?

Diagnosis (read of handle_packet, protocol_agent.py): the neighbor cache is
guarded by a PER-SENDER sequence number. For a LIVE sender, a message is
accepted only if `seq > last_seq`; an older/equal seq is dropped. So:
- out-of-order (old after new) → the old message is REJECTED, fresh state kept;
- duplicate → dropped;
- `rxtime` updates only on ACCEPTED messages → liveness tracks the freshest;
- this guards the OVERLAY too: pulses ride in `prop_state` and the cancelling-
  bias `dp_shift` is an AgentState field, so stale overlay inputs are dropped
  before reaching the dual_pulse layer.
The only "stale" that reaches the controller is therefore uniform DELAY (the
freshest-available state is `delay` seconds old) — the axis already studied.
Residual nuance: after a sender EXPIRES, an old message is accepted (rxtime=now)
to re-acquire a recovered neighbor whose seq may have reset — so the cache can
briefly hold ~one broadcast-period-old state; negligible and bounded.

Action: extracted the decision into the pure helper
`AgentProtocol._accept_neighbor_state(seq, last_seq, expired)` (behavior-
preserving, de Morgan of the original guard) and locked it with
`tests/test_stale_messages.py` (7 tests: fresh accepted; stale/reordered
rejected when live; duplicate rejected; accepted after expiry; first-from-unseen;
monotonicity property). Suite 145→152.

**Verdict:** robustness to out-of-order delivery is a property of the per-sender
sequence guard — now TESTED. No reordering-medium sim is needed (GrADyS delivers
in order anyway; a custom reordering medium is optional future work if a
referee wants an end-to-end demonstration). Cap. 7 can state: "the per-sender
sequence numbers make the neighbor cache (and thus the overlay inputs) robust to
out-of-order delivery; the residual staleness equals the communication delay,
characterized in §7.2.3." Front 2 CLOSED (positive, cheap).

## 2026-06-13 — Anti-windup / limiter (front 4): diagnosis = no pathology, no limiter
Question: the overlay's control effort is ~2× baseline under churn — is that a
saturation/windup pathology that an anti-windup or command limiter should fix?

Diagnosis (free first cut from the 8-seed churn CSV, then a snappy check):
`sat_frac` (= Pr(velocity_norm ≥ Vmax), the M6-style metric added in Ciclo 0):
- default regime (τ_a=1, rates 6/12/24/48): `sat_frac = 0.0000` for BOTH baseline
  and B2 (median AND max). Velocities sit at ~3–9% of Vmax (effort_mean_v2
  0.001–0.008). Effort is ~2–2.7× baseline but of a TINY absolute.
- snappy regime (τ_a=0.2, rate 24, 3 seeds, `churn_sweep_results_c4_snappy_tau02.csv`):
  `sat_frac = 0.0000` too (baseline and B2); effort B2 0.0073 (~3.8× baseline,
  still ~8% of Vmax); overlay still helps (adv 1.20).

**Verdict: no windup pathology under churn.** Windup requires actuator
SATURATION (the integrator/accumulator winds up while the command can't be
realized); with `sat_frac = 0` there is no saturation → no windup → the 2× effort
is a benign trade-off (the overlay actively redistributes via feedforward; the
baseline only relaxes — more motion, but well below the actuator limit). A
dedicated anti-windup / limiter is **not warranted** for the studied regime.
The only saturation in the project is the CLEAN single-fault snappy regime
(τ_a=0.2, Cap. 6 §6.6, a declared boundary of the dimensionless law), and there
**M8 already provides anti-windup** by construction (`consume_motion` drains the
shift at the saturation-CLIPPED commanded rotation, not the nominal). A limiter
would only matter for hardware combining small τ_a with dense churn — untested,
flagged as out of scope. Front 4 CLOSED (negative-becomes-data: measured before
building, avoided an unnecessary mechanism). Methodology note: `sat_frac` is the
right discriminator and is now exercised; the effort 2× alone is not evidence of
pathology.

## 2026-07-26 — P0: provenance (seed + git hash + full param set in every row)
Problem, not hypothesis: the main campaign's result CSVs were produced from an
**uncommitted working tree** (2026-05-30 → 2026-06-06) and record neither the
seed, nor the code version, nor the parameters pinned for the run. The defaults
they silently inherited have since moved (`CONTROL_PERIOD` 0.01→0.05,
`AGENT_STATE_TIMEOUT` 5·dt→max(20·dt, 0.2), `DUAL_PULSE_CONSUME_FF_ONLY` and
`DUAL_PULSE_MULTIPLICITY` now default ON), so those rows cannot be re-derived.

Commands (git `fc08491`, clean tree at the point the changes were authored):
```powershell
python -m pytest -q                                    # 152 passed (before)
python experiments/scaling_law/check_provenance.py     # inventory
python -m pytest -q                                    # 205 passed (after)
```

**Inventory (the measurement, not a side note): 0/49 result CSVs, 765 result
rows, carry git provenance.** 42/49 carry a `seed` column; `git_commit` and
`git_dirty` are absent from every single file, and `figure_data.csv` — the input
to the thesis figures (`make_figures.py`, `make_table.py`) — carries neither seed
nor commit. Parameter coverage of the 11 key knobs ranges 0/11
(`trackC_results*`, `mmult_adjacent_results`) to 5/11 (`collapse_results*`).

What was built (no simulation behaviour touched; 152 pre-existing tests
unchanged and still green):
- `provenance.py` — `_git_provenance()` → (short sha, dirty), exception-safe with
  a `("unknown", True)` fallback; `resolved_config()` reads all 99 public
  constants off the imported `config_param` (env overrides already folded in);
  `summary_provenance()` builds the flat row through the single mapping table
  `SUMMARY_FROM_CONFIG`. Git state is captured ONCE, before the sim, and cached.
- `run_manifest.json` per run directory (`main.py`, written pre-sim): argv, cwd,
  python/platform, git (commit/branch/dirty/raw `status --porcelain`), the env
  vars actually SET, and the complete resolved config. ~6 KB.
- `plot_telemetry.SUMMARY_COLUMNS` extended 28 → 51 columns: `git_commit`,
  `git_dirty`, `experiment_seed` + every pinned parameter. Assembled from the
  resolved config, with a mismatch guard that warns instead of writing a blank
  provenance cell. Old `runs_summary.csv` files rotate to `.bak.<ts>` (rule 2:
  nothing deleted).
- `config_param.METRICS_T0` and `EXPERIMENT_REPRODUCIBLE` became env-overridable
  (defaults unchanged: 0.0 and True) — they were the two holes in rule 3.
- `metrics_util.run_provenance(run_dir)` for the sweep runners, reading the
  CHILD's manifest. A runner must NOT call `summary_provenance()` directly: it
  is the parent process and its `config_param` holds the parent's defaults, not
  the values it pinned in the child's env.
- `experiments/scaling_law/check_provenance.py` (the audit above) and
  `docs/experiments/PROVENANCE.md` (schema + operational rule).
- `tests/test_provenance.py` — 53 tests (suite 152 → 205).

Backward compatibility: the only schema changed is `runs_summary.csv`, whose sole
reader is `run_sweep.py::load_completed_combos` (`csv.DictReader` + `.get()`,
tolerant — now regression-tested). No script under `experiments/scaling_law/`
imports `plot_telemetry` or `config_param`; `analyze_collapse.py`,
`make_table.py` and `analyze_comm.py` were re-run unchanged as proof.

**Open gap (deliberate, measured rather than papered over): the runners still do
not stamp provenance onto their `*_results.csv` rows.** P0 provides the
mechanism (`run_provenance(run_dir)`, one line per runner) and the audit that
quantifies the debt; adopting it is P1+ work. Until then `check_provenance.py`
will keep reporting 0/N on core provenance for any newly written result file.

**Thesis impact.** Cap. 7 (metodologia/reprodutibilidade) gains a concrete
provenance schema to describe. Every number currently cited from
`experiments/scaling_law/*.csv` — i.e. the Cap. 3 Θ(N²) baseline, the Cap. 5/6
B2 flat-τ and dimensionless-law results, and the Cap. 7 robustness maps — is, as
of today, **not reproducible from its own row**. Re-running the load-bearing
cells under the new schema (new files, old ones to `_archive/`) is the
prerequisite for citing them as evidence rather than as history.

## 2026-07-26 — P1/E4: churn re-analysed pairwise, every metric (no new simulation)
Question: the introduction argues the MAXIMUM ANGULAR GAP is the mission-critical
quantity, but churn robustness was reported on the MEAN spacing error. What do
the other already-collected metrics say?

Commands (git `a6f3099`, clean tree; analysis only, zero simulations):
```powershell
python experiments/scaling_law/analyze_churn_paired.py    # + churn_paired_results.csv + figure
python experiments/scaling_law/probe_gmax_floor.py        # + gmax_probe_results.csv
```
Full report: [CHURN_PAIRED.md](CHURN_PAIRED.md). First statistical test in this
repository (paired Wilcoxon; `scipy>=1.11` added to requirements/pyproject).

**Semantics first — two findings that reframe the question.**
1. `egap_max` is **not** the maximum angular gap. `E_gap` is the RMS ACROSS THE
   RING of the relative gap error (`protocol_target.py:686`); `egap_max` is the
   MAX OVER TIME of that spatial RMS, over t ∈ [20 s, 155 s]
   (`run_churn_sweep.py:47-56` — a runner-local helper, NOT `metrics_util`). Two
   aggregations stacked: an average that hides one wide gap among 23 narrow ones,
   then an extreme-value pick over ~2700 samples.
2. The real maximum gap is **`G_max`** (`protocol_target.py:685`), written to every
   `target_telemetry.csv` since forever and **never aggregated by any churn
   analysis**. Both it and `E_gap` are normalised by the ALIVE count, so a
   half-dead ring spread perfectly scores `G_max = 1`; the absolute breach in
   radians is `G_max·2π/M` and **M is not logged**, so it is not recoverable from
   the 765 existing result rows. One-line instrumentation gap (`alive_count`,
   ideally `gap_max_rad`) — blocking for any breach-window claim.
   Related: `egap_avg` means DIFFERENT windows in churn CSVs (t≥20 s) vs
   collapse/trackC CSVs (`metrics_util.py:119`, t≥15 s). Not comparable across
   campaigns without naming the runner.

**Paired results** (`c3_churn8_dt05`, 32 pairs by (rate, seed) — same Poisson
stream, since baseline and B2 share `EXPERIMENT_SEED`):
- `egap_avg`  adv 1.19 [1.11, 1.34], **0/32 losses**, p < 0.001, r = 0.87.
  Reproduces §7.2.7 EXACTLY (1.31/1.23/1.15/1.14, adv_min ≥ 1.11). Verified.
- `egap_p90`  adv 1.07 [1.00, 1.18], 1/32, p < 0.001 — and the edge GROWS with
  rate (1.04→1.13).
- `egap_max`  adv 1.05 [0.85, 1.46], **14/32 losses**; per-rate p = 0.945 / 0.250 /
  0.016 / 0.742 — only rate 24 is individually significant. **No reliable effect.**
- `fairness_p95` adv 1.00 [0.57, 1.63], **15/32**, p = 0.73, r = 0.06. **Null.**
- `sat_frac`  identically 0.0 in all 64 cells (test undefined) — confirms the
  2026-06-13 anti-windup diagnosis on 8 seeds.
- `effort_mean_v2` cost 2.41× [1.74, 3.23], **32/32**, p < 0.001. Sharpens the
  logged "~2.2–2.7×" into a characterised interval.

**The pattern is not specific to c3** (Task 3). In all four churn campaigns the
`egap_max` advantage is far smaller than `egap_avg`'s and always has losing pairs
where `egap_avg` has none: c3 1.19(0/32) vs 1.05(14/32); m8off_8seed 1.15(11/32)
vs 1.04(10/32); c1B_m8on_dt01 1.28(0/12) vs 1.13(3/12); c1C_dt05 1.16(0/6) vs
**0.98(5/6)**. The ordering never inverts.

**Mechanism (hypothesis, with in-data support).** `egap_max` is set by the INSTANT
of the event — pure geometry, `G_max` jumps to `2(N-1)/N = 1.92` before any
protocol can act. `egap_avg` is set by the RECOVERY that follows (Θ(N²)≈20 s
baseline vs ≈2 s B2) — that is where the whole advantage lives. Prediction
confirmed in the existing data: as rate rises 6→48, `egap_avg`'s advantage FALLS
(1.31→1.14, the baseline never settles) while `egap_p90`'s RISES (1.04→1.13, the
upper decile becomes recovery- rather than peak-dominated); they converge to
≈1.14. A peak-dominated metric sits outside that convergence and shows no trend.
Sharper corollary: B2's ≈2.1 s ≈ 2·T_FF already equals the actuation-limited floor
(max displacement ≈ r·gap/2 ≈ 2.7 m, i.e. ~2–3·tau_a of first-order lag), so the
remaining lever on the breach window is platform agility, not coordination.

**G_max probe** (`churn_sweep_runs_stamp/`, the only churn run dir whose
`target_telemetry.csv` survived): at sparse churn the peak `G_max` is 2.11
(baseline) / 2.03, i.e. within 10% of the protocol-independent geometric 1.917,
climbing to ~3.5 at rate 48 as concurrent deaths merge more gaps. No `G_max`
statistic separates the methods (p = 0.13–0.97). **BUT** — a P0-shaped caveat —
that directory's BASELINE half matches the dt=0.01 ablation family byte-exactly
while its B2 half matches NO committed CSV and loses on `egap_avg` (0.72). Which
overlay variant produced it is unrecoverable. So this probe cannot decide the
question for the validated B2; it establishes only that `G_max` is extractable at
zero cost and that the geometric floor is a live hypothesis.

**Deciding experiment proposed** (CHURN_PAIRED.md §5.3): NOT a churn sweep —
concurrent deaths contaminate the peak. Single deterministic permanent failure,
N=24, sweeping the KINEMATIC axes `VM_MAX_SPEED_XY ∈ {2.5,5,10,20}` ×
`VM_TAU_XY ∈ {0.5,1,2}` × {baseline, B2} × 5 seeds = 120 runs of ~35 s. Measure
peak `G_max`, `t_close` (time until `G_max` < 1.25 and stays), and the breach AREA
∫max(0, G_max−1.25)dt. Decision rule in the report; the outcome that would most
change the thesis is `t_close` flat for B2 while the baseline's grows with N —
which would make `G_max` the headline result rather than an absent one.

**Thesis impact.** Cap. 7 §7.2.7's 8/8-seed claim is VERIFIED but must name its
metric (`egap_avg`) and gain the negative results for max and fairness; the
`[decidir depois]` note about the unpaired table is now decidable (keep paired,
add p-values). Cap. 9 §9.1 RQ4's "churn (vantagem pareada, 8/8 seeds)" must name
the metric too, or a reader carries it to the max-gap claim of the introduction,
which the data does not support; §9.2 C5 should say the map covers mean spacing
error, not worst-case gap; §9.3 gains one honest limitation. Draft v1
(`5-preliminary-results.tex`, `6-conclusion.tex`) could NOT be located — there is
no `.tex` anywhere in this repo or the sibling projects.

## 2026-07-26 — E4 deciding experiment: the breach window is actuation-limited
Follows the 2026-07-26 churn re-analysis, which found the campaign had never
measured the mission-critical quantity (the maximum angular gap) and proposed a
pre-registered decision rule (CHURN_PAIRED.md §5.3). Full report:
[BREACH_WINDOW.md](BREACH_WINDOW.md).

Instrumentation first (`e062eed`): `target_telemetry.csv` gains `alive_count` and
`gap_max_rad`. Without the alive count, `G_max` and `E_gap` — both normalised by
2*pi/M — cannot be converted back to physical angles, so the absolute breach was
unrecoverable even in principle, including from the 765 rows already collected.
Schema now has a single definition (`protocol_target.TARGET_TELEMETRY_COLUMNS`).
15 tests added (suite 205 -> 220) locking the geometry.

Commands (git `93915a5`, clean tree; 140 runs, ZERO failures):
```powershell
# 6 parallel sweeps: 4 by Vmax (30 runs each) + 2 on the N axis (10 each)
python experiments/scaling_law/run_breach_window.py     # BREACH_VMAX/_TAUS/_N/_SEEDS/_BUDGET
python experiments/scaling_law/analyze_breach_window.py
```
Single deterministic PERMANENT failure (not churn: concurrent deaths contaminate
the peak, 2.11 -> 3.49 from rate 6 to 48). N=24 baseline, Vmax {2.5,5,10,20} x
tau_a {0.5,1,2} x 5 seeds, plus N {12,24,48} at Vmax=10/tau_a=1.

**1. The peak is a geometric floor, exactly.** Predicted 2(N-1)/N = 1.9167;
measured 1.9174 for BOTH methods, identical at every Vmax and tau_a, 0/60 losing
pairs. It happens at the instant of the failure, before any protocol can act —
state it as a bound, do not optimise it.

**2. The duration is actuation-limited, and the overlay buys 1.1-1.5x.**
t_close(1.25) baseline/B2 = 2.30/1.80, 3.15/2.35, 4.75/3.35 s at tau_a =
0.5/1/2 — scales with tau_a. Across Vmax 2.5 -> 20 (8x) the medians are
IDENTICAL to three digits (3.15 / 2.35). So: first-order actuation lag sets it,
top speed does not, coordination buys 1.28-1.42x (60/60 pairs, p = 1e-4). At the
stricter 1.10 threshold the overlay's edge is 2.0-3.1x.

**3. REFUTED — "the reconfiguration time is exactly how long that gap stays
open"** (draft v1 `1-introduction.tex:26`). Both quantities come from the same
telemetry: baseline t_close = 3.15 s vs t_settle(E_gap) = 35.15 s, a factor
**11.2x** [5.3, 17.4]; B2 2.35 vs 6.95 s, 3.0x. Consequence: the overlay's
advantage on RECONFIGURATION TIME is ~5x here (9-149x in the large-N campaign),
but on the BREACH WINDOW it is **1.34x**. Different results about different
quantities, and only the second is about the thing the introduction calls
mission-critical.

**4. REFUTED — "the breach window grows with the square of the swarm size, at
N=100 it stretches to minutes"** (v1 `1-introduction.tex:36-37`,
`6-conclusion.tex:65-68`). Fitted exponent p in t_close ~ N^p: **0.31** (thr 1.25)
and **0.64** (thr 1.10) for the baseline; 0.12 / 0.40 for B2. Extrapolated to
N=100: 4.7 s and 18.2 s. Not N^2, not minutes. Worse for the motivation: the
breach WIDTH shrinks with N — peak gap 60.1 deg / 30.0 deg / 15.1 deg at
N = 12/24/48 — so single-failure risk decreases with swarm size on BOTH axes,
the opposite of "this is also why scale matters".

**What survives.** The Theta(N^2) relaxation and the flat-in-N reconfiguration
are reproduced here (baseline tau_fit 20.29 s vs B2 2.14 s at N=24). What breaks
is the motivational bridge from those results to mission relevance, which ran
through a single-failure breach window that is short, weakly N-dependent and
narrowing with N.

**Reframing the data DOES support** (BREACH_WINDOW.md §5): slow reconfiguration
matters because the ring is not ready for the NEXT event. At N=100 the baseline
relaxation is 0.033*N^2 ~ 330 s, so any realistic failure rate finds the ring
permanently non-uniform, and a failure landing on a non-uniform ring opens a
worse gap — the churn data already shows the compounding (peak G_max 2.11 at
6/min -> 3.49 at 48/min, vs the 1.92 single-failure floor). That argument is
mission-relevant, N^2-driven and supported by data in hand; it is NOT yet the
argument the thesis makes.

**Next experiment, prediction pre-registered** (§6): re-run the churn sweep with
the new telemetry and report breach metrics. Predicted: the overlay's advantage
on peak G_max and time-above-threshold should be LARGER than the 1.34x measured
for a single failure and should GROW with churn rate. If it stays ~1.3x and flat,
the overlay does not buy breach safety at all and C5 must say so.

**Thesis impact.** Draft v1 `1-introduction.tex` §Metrics does not list the
maximum gap among its metrics although §Motivation calls it mission-critical —
that omission is the structural origin of this whole finding. `:26` and `:36-37`
must change (see above); `6-conclusion.tex:65-68` repeats them verbatim.
Draft v2 does NOT carry the breach claim (checked): its edits are the
metric-naming ones from the churn re-analysis (cap7 §7.2.7, cap9 §9.1/§9.2/§9.3).

**Method note (a defect in P0's provenance, now fixed).** Per-cell git capture is
self-referential: the sweep's own untracked results CSV dirties the tree, so cell
1 records dirty=False and cells 2..n record dirty=True for reasons unrelated to
the code (verified: `git status` during the sweep listed ONLY the output CSVs).
`run_breach_window.py` now captures git state ONCE at startup, before writing
anything — git state is a property of the CAMPAIGN, the pinned parameters are a
property of the cell. Other runners adopting `run_provenance()` must do the same.
