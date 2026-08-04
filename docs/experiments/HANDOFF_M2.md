# Handoff — item 9: densified m=2 baseline vs the overlay (192 cells)

**Thesis-facing summary, self-contained.** Canonical scoping and pre-registration:
[SCOPING_M2.md](SCOPING_M2.md) (with its stamped addendum) and the docstring of
`experiments/scaling_law/run_m2_campaign.py`. Data: `m2_campaign_results.csv`, 192 rows, all
`git_dirty=False`, single commit `ba44b18`. Grid: 3 methods {baseline, m2, overlay-B2} × 2 N
{24, 50} × 2 regimes {single death, churn 12/min total} × 2 ranges {below / above the 2-hop
chord} × 8 paired seeds.

**What m=2 is.** Each agent couples DIRECTLY to its 1st and 2nd angular neighbours per side —
one physical transmission, no relay. It requires range ≥ the 2-hop chord `2R·sin(2π/N)`, the
same geometric threshold phase 8a measured for the overlay. The law is the *existing* spacing
law parameterised by hop count (exact reduction at k=1), combined convexly with w₂=2 and the
loop gain renormalised so both laws run at the **same nominal sampled margin** `g·dt·λ_max`
(×1.92010 at N=24, ×1.92429 at N=50 — discrete eigenvalues, pinned by unit test). A guard drops
the k=2 term whenever the fresh visible ring has < 5 members (naive indexing would alias
succ₂ = pred₁); a degraded tick is the baseline computation float-for-float.

## 1. The result table

Clean regime = `t_settle` speedup over baseline (higher is better); churn = `egap_mean_steady20`
ratio to baseline (lower is better). Median over 8 paired seeds; IQRs in the CSV are tight
throughout (near-replicate seeds).

| block | N | regime | range | m2 | overlay B2 | m2 vs B2, direct |
|---|---:|---|---|---|---|---|
| A | 24 | single death | below chord | **= baseline, bit-exact** | 1.32× *worse* | m2 harmless, B2 harmful |
| B | 24 | single death | above chord | 2.93× faster | **5.86× faster** | B2 wins 2.00× |
| C | 24 | churn | below chord | = baseline (0.997) | 1.82× worse | m2 harmless, B2 harmful |
| D | 24 | churn | above chord | 1.18× better | 1.19× better | **tie** (1.003, 4/8) |
| E | 50 | single death | below chord | **= baseline, bit-exact** | 1.37× worse | m2 harmless, B2 harmful |
| F | 50 | single death | above chord | 1.13× (see §3) | **5.62× faster** | B2 wins 4.97× |
| G | 50 | churn | below chord | = baseline (1.000) | **2.59× worse** | m2 harmless, B2 harmful |
| H | 50 | churn | above chord | **1.17× better** | 1.09× better | **m2 wins, 8/8 seeds** |

On the tail time-constant `tau_fit` in the clean blocks (the λ₂ probe):
B (N=24): baseline/m2 = 3.84, baseline/B2 = **9.53**. F (N=50): baseline/m2 = 2.52,
baseline/B2 = **20.4** (B2's flat-tau against the baseline's ~N² growth; note the baseline R² at N=50 sits below the
campaign's 0.9 bar in BOTH clean blocks — 0.825 at E, 0.841 at F — quote with that caveat).

## 2. Verdicts against the pre-registration

**P5 (clean speedup = 3.1565 / 3.1970, derived).** Not met as a point prediction; survives as a
scale on the tail metric. N=24: primary 2.93 (−7.1%), secondary 3.84 (+21.7%) — the true
asymptotic rate is bracketed by the two metrics. N=50: primary **1.13** — but this is metric
saturation, not law failure (§3); secondary 2.52 (−21%). No tolerance band was pre-registered;
the honest statement is that the fair-gain eigenvalue argument predicts the right *scale*
(2.5–3.8×) and the exact digit depends on which metric captures the slow mode.

**P6 (churn ordering overlay ≥ m2 > baseline).** The pre-registered ADVERSE outcome is what
happened, in two stages: a dead tie at N=24 (m2/B2 = 1.003) and an **inversion at N=50**
(m2/B2 = 0.946, m2 better in 8/8 seeds; one B2 seed worse than the baseline itself). Under
churn, the overlay does not beat a passive densification with the same radio requirement, and
the gap moves against it as N grows. The cross prediction (overlay/m2 ≈ 5.0× clean at N=50)
reads 4.97 on `t_settle` — but its components missed in opposite directions (B2 5.62 vs 16
predicted; m2 1.13 vs 3.2), so the ratio hit is partly common-mode cancellation of the same
saturation; on `tau_fit` the components are 20.4 and 2.52 (ratio 8.1). Report the components,
not just the ratio.

**P7 (below the chord, m2 ≈ baseline).** Confirmed in the strongest available form everywhere:
bit-exact identity in the clean blocks (A, E — `t_settle` equal in all 8 seeds at both N) and
statistical identity under churn (C: 0.997; G: 1.000; guard dropping 99.9% of ticks). The
degradation guard does exactly what pin (b) required.

**P8 (chattering does not destabilise).** Confirmed. Block-median toggle rates up to 0.97/s
(block H) and a per-cell maximum of 1.07/s (block D, seed 3), with no degradation attributable
to switching — the highest-toggle blocks are the ones where m2
performs best.

## 3. The metric caveat that must travel with P5

At N=50 the baseline's `t_settle` (46.9 s) is far below its own `tau_fit` (64.5 s): the settle
threshold is crossed while the slow mode is still decaying, because a single death's residual
excitation of the global mode is small relative to the threshold. `t_settle` therefore
**saturates as a discriminator** at large N in the single-death scenario — the mirror image of
the DT_CROSSOVER lesson (there `tau_fit` broke at large N; here, for this scenario, `t_settle`
does). Any future clean-regime comparison at N ≥ 50 should report both metrics and say which
regime each one is valid in.

## 4. Cost — measured, and the message claim proven per cell

`tx_rows_steady20` is **equal across the three methods in every seed of every block** (printed
per cell, not asserted): every method transmits exactly one AgentState broadcast per agent per
tick. m=2 adds zero transmissions and zero payload; the overlay adds pulse payloads
(`pulse_payloads_fullrun` column). The real cost axis is the REQUIRED range: 1-hop chord for
the baseline, 2-hop chord for both m2 and the overlay (`range_required_c` column).

## 5. What this changes for the thesis (§4.1 framing)

1. **The overlay's value proposition narrows to one cell of the design space and strengthens
   there**: single-event reconfiguration above the 2-hop chord, where flat-tau vs the
   baseline's ~N² gives it 20× over the baseline and ~8× over the densification (tau_fit,
   N=50). That is the claim the thesis can defend — and it is the *dissemination architecture*
   (event-triggered feedforward vs diffusive coupling) doing the work, cleanly isolated by an
   equal-margin, equal-range, equal-messages comparison.
2. **Under churn the overlay is not the right tool**: it ties with passive densification at
   N=24 and loses to it at N=50, in every paired seed. The §4.1 argument must not claim churn
   as overlay territory.
3. **Below the 2-hop chord the two methods fail in opposite ways**: m2 degrades to the baseline
   exactly (harmless by construction, verified bit-exact), while the overlay is actively
   harmful and its harm grows with N (1.82× → 2.59×). For deployment guidance: if the radio
   cannot reach the 2-hop chord, run the baseline — never the overlay; m2 is safe to leave on.
4. **The m=2 comparison is now a measured line, not an argument** — the alternative the text
   previously dismissed by reasoning is in the tables with the same provenance standard as
   everything else.

## 6. Caveats

* Near-replicate seeds (uniform init, ideal channel): IQRs measure replication noise. One churn
  rate (12/min total; Π₂′ = 1.6 at both N — a design choice, held constant across N). Two N.
* `t_settle` saturation at N=50 (§3); baseline `tau_fit` R² below the 0.9 bar in both N=50
  clean blocks: 0.825 (E) and 0.841 (F).
* m2's guard has no hysteresis on the k=2 term; toggle rates are reported and benign here, but
  denser churn was not tested.
* Amendment ledger, all stamped pre-analysis in the runner/addendum: (1) w₂=2 + gain ×1.9201
  correcting a factor-2 chain (three prescriptions compared in addendum A.2); (2)
  `PROTECTION_ANGLE_DEG` pin corrected 0 → 360 before any cell ran; (3) canonical literal
  metres after the A4-INERTNESS sentinel aborted on cell 3 over a 0.14 mm rounding difference —
  the abort that *proved* the m2 implementation inert (HEAD reproduces the 8a-(ii) reference
  byte-for-byte with the literal range).

## 7. Reproduction

```powershell
M2C_DRY_RUN=1 python experiments/scaling_law/run_m2_campaign.py   # grid + cost, nothing runs
M2C_GO=1     python experiments/scaling_law/run_m2_campaign.py    # full 192 (~4.5 h)
```

Five sentinels abort the sweep on violation: effective range matrix ≠ default; target alive
count within the superposition band; role census exact; **A4-INERTNESS** (re-run baseline and
overlay cells byte-identical to `comm_churn_runs/`); m2_guard sidecar present iff method = m2.
All 192 cells passed all five.
