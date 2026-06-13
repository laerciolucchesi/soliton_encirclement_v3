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
