# Finite ring range: how much radio does the overlay actually need?

**Why.** Every result in this campaign was measured with a single 200 m communication range —
larger than the swarm's own diameter, so every agent hears every other agent and the ring's
"neighbour" relation is a *logical* construct, not a physical one. That makes one of the
overlay's selling points untestable: `dual_pulse` is a **neighbour-only** protocol, and a
protocol that only talks to its neighbours should keep working when the radio can only *reach*
its neighbours. This experiment gives the ring a finite range and finds the point where it
stops working.

**Answer, in one line.** There are **two** thresholds, not one, and they govern different
quantities. Closing the gap at all needs about **1 hop** and the cliff is *identical* for both
methods (so it is a property of the ring and the controller, not of the overlay). The overlay's
**advantage** needs **2 hops** — `range >= 2·R·sin(2π/N)` — and below that line B2 is not merely
degraded but **actively worse than doing nothing** (2× the baseline's closing time). The reason
is not a weaker correction but the **wrong** one: the departure pulse dies unheard, the ring then
contracts, and the reappearing neighbour is locally indistinguishable from an *arrival*, so the
swarm executes a sign-inverted redistribution for a node that never joined (§2). Above the line
the advantage appears at full strength and then **saturates**: from c = 2 to c = 5 the closing
time does not move.

```powershell
python experiments/scaling_law/run_comm_range_sweep.py
# env: CRS_RANGES / CRS_UPLINK / CRS_N / CRS_SEEDS / CRS_BUDGET / CRS_METHODS / CRS_TAG
#      CRS_DRY_RUN=1 prints the grid and exits
```
80 runs, zero failures, **every row `dirty=False`**. Output: `comm_range_results.csv`.

**Victim rule** (required by the campaign convention): `victim = 2 + ((N//2 + seed) % N)`.
With an equidistant ring the seed is a symmetry check, not scenario variability — see §6.

---

## 0. What "range" means here, and why two of them

GrADyS evaluates range at the **sender only**: `can_transmit()` takes one medium — the
sender's — and there is no receiver sensitivity in the model. That matters because an agent's
`AgentState` is *one* broadcast serving two audiences with opposite requirements:

* the **ring neighbours**, which this experiment wants to restrict; and
* the **target**, which must hear *every* agent. An agent the target stops hearing is pruned
  from `agent_states` and `alive_lambdas` after `AGENT_STATE_TIMEOUT`
  (`protocol_target._prune_expired_states`) — a live drone declared dead. That corrupts
  `alive_count`, the lambda map fed back to the agents, and every M1–M7 metric, **silently**:
  `G_max` and `E_gap` normalise by the number of agents the target *heard*, so a half-observed
  ring still scores ≈ 1.0.

With one global range the interesting region (below R = 20 m, where the target loses the ring)
is therefore **unobservable** — the measuring instrument dies together with the phenomenon.
`comm_role_aware.RoleAwareCommunicationHandler` splits the range per `(sender_role,
receiver_role)` pair, so the ring can be starved while the uplink stays at 200 m. Assertion A2
below is what proves the instrument stayed alive.

## The experiment

| | |
|---|---|
| N | 24 agents, R = 20 m, equidistant start, static target |
| swept axis | `COMM_RANGE_AGENT_AGENT` ∈ {6.3, 8.4, 10.4, 15.7, 26.1} m |
| pinned | `COMM_RANGE_AGENT_TARGET` = 200 m, dt = 0.05, τ_a = 1.0, `K_E_TAU` = 250/N, `AGENT_STATE_TIMEOUT` = 5·dt |
| fault | one deterministic permanent death at t = 5 s |
| seeds | 8, paired (same seed ⇒ same victim for both methods) |
| methods | baseline, B2 |
| budget | 90 s after the fault |

The five grid points sit just above the 1-, 1-, 2-, 3- and 5-hop chords. Normalisation
`c = range / (1-hop chord)`; the 1-hop chord is **5.221 m** pre-death (N = 24) and **5.447 m**
post-death (N = 23), so both `c_pre` and `c_post` are reported — documents in this campaign have
already disagreed about which one they meant.

**Three assertions abort the whole sweep** (not the cell — a violated invariant means the row
cannot be trusted): the effective matrix must differ from the default (else a fully connected
run wears a role-aware label), `alive_count >= N−1` after warmup (else the uplink corruption
above), and the role census must be exactly 1 target + N agents with zero unknown. All 80 cells
passed all three.

---

## 1. Two thresholds, not one

| range | c_pre | c_post | coverage | `t_close_125` baseline | `t_close_125` B2 | advantage |
|---:|---:|---:|---:|---|---|---:|
| 6.3 m | 1.21 | 1.16 | 0.00 | **inf** (8/8) | **inf** (8/8) | — |
| 8.4 m | 1.61 | 1.54 | 0.96 | 3.27 [3.25, 3.30] | **6.45** [6.40, 6.45] | **0.51×** |
| 10.4 m | 1.99 | 1.91 | 1.00 | 3.20 [3.19, 3.21] | **2.30** [2.30, 2.31] | **1.39×** |
| 15.7 m | 3.01 | 2.88 | 1.00 | 3.22 [3.17, 3.30] | 2.30 [2.30, 2.35] | 1.40× |
| 26.1 m | 5.00 | 4.79 | 1.00 | 3.25 [3.24, 3.26] | 2.32 [2.30, 2.35] | 1.40× |

Median [IQR], n = 8 per cell. Advantage = baseline / B2 (> 1 = overlay better).

On the strict threshold the inversion is sharper — `t_close_110`: **16.58 s vs 8.00 s** at
8.4 m (0.48×), then **3.42 s vs 7.65 s** at 10.4 m (**2.23×**), 2.21× and 2.25× above.

**(1) Closing at all — the cliff is at c ∈ (1.21, 1.61], identical for both methods.** Since the
baseline runs no overlay at all, this threshold cannot be a property of `dual_pulse`. It is the
range at which the ring plus the tangential controller stop being able to re-form. Breach area
confirms the collapse rather than a slow close: 17.7 (baseline) and 21.1 (B2) at 6.3 m against
~0.8–0.9 everywhere above it.

**(2) The advantage — the cliff is the 2-hop chord, 10.353 m at N = 24.** 10.4 m is the first
grid point above it, and it is exactly where coverage completes and the sign flips.

## 2. Why the 2-hop chord, mechanically: the ring answers the wrong question

Coverage at 8.4 m is 0.96 = 22/23 with `hop_sum = 23` — the **full** ring traversal — so this is
not pulse truncation; the pulses circle the ring perfectly well. But reading the `event_id`s
rather than just counting completions shows that the 22/23 belongs to a **different event than
the death**, and the whole reading changes.

`event_id` is `"originator_seq"`, so the sequence number is an injection counter per originator.
Retrofitted from every phase (i) cell:

| range | landed events | kind | `seq` of the landed event | coverage |
|---:|---:|---|---:|---:|
| 6.3 m | 0 (5 of 8 seeds) / 13 | mixed | up to **19** | 0.00–0.30 |
| 8.4 m | 1 | **ENTRADA** | **2** | 0.96 |
| 10.4 m | 1 | SAIDA | **1** | 1.00 |
| 26.1 m | 1 | SAIDA | 1 | 1.00 |

Identical in all 8 seeds. At 10.4 m and above the **first** injection lands and it is a SAIDA —
a departure, which is what actually happened. At 8.4 m the landed event is the originator's
**second** injection, and it is an **ENTRADA** — an arrival. The first one left no trace at all.

The chain:

1. The victim dies. Its **predecessor** — the canonical originator — sees its successor go
   stale and injects a **SAIDA** (event #1).
2. The SAIDA's two counter-propagating pulses: the one aimed across the hole cannot reach the
   victim's successor, because that distance is the **2-hop chord** (10.353 m > 8.4 m). A
   receiver applies its shift only after seeing **both** directions, so with one direction dead
   at birth **nobody completes**. Event #1 vanishes silently — the protocol logs completions,
   not injections, which is why it took the `seq` counter to see it at all.
3. The ring contracts under the controller. The successor drifts into the originator's range.
4. Locally, that is a node *appearing* where there was none: `_classify_succ_event` sees
   `succ_changed` with the previous successor still fresh and classifies it **ENTRADA**
   (event #2), which then lands with 22/23 coverage.
5. An ENTRADA shift is **sign-inverted** relative to a SAIDA. So 22 of 23 survivors redistribute
   as if a node had **joined**, widening a gap that a death had already widened.

So B2 at 8.4 m is not performing a degraded death-redistribution. It is performing a complete,
**wrongly-signed arrival-redistribution**, which is why it is 2× *worse* than running no overlay
at all rather than merely weaker.

> **The general statement.** Under a finite radio range, *"a node came into range"* and *"a node
> joined the ring"* are locally indistinguishable. The neighbour-only premise — the protocol's
> main architectural claim — is exactly what makes the ambiguity unresolvable: the only evidence
> a node has is its own neighbour set, and both events look identical in it. The 2-hop chord is
> the range at which the SAIDA reaches the far side of the hole before the geometry can
> re-present itself as an arrival.

Per-node verification (seed 0, victim 14, originator 13, successor 15):

| range | successor completed? | covered | event that landed |
|---:|---|---:|---|
| 6.3 m | no | 0/23 | none |
| 8.4 m | **no** | 22/23 | ENTRADA (seq 2) |
| 10.4 m | **yes** | 23/23 | SAIDA (seq 1) |
| 15.7 m | yes | 23/23 | SAIDA (seq 1) |

The successor abstaining is real, but it is a *symptom*: it abstains from the spurious ENTRADA,
having never been offered the SAIDA. Phase (i-b) tests whether step 4 is geometry or merely a
short failure-detector timeout — see §5.

## 3. Saturation — and the design rule

B2's `t_close_125` is 2.30, 2.30 and 2.32 s at c = 1.99, 3.01 and 5.00; the baseline's is
3.20, 3.22 and 3.25 s. Neither method improves with more range once coverage is complete.

> **Design rule.** Size the ring radio at `2·R·sin(2π/N)` — the 2-hop chord — and stop.
> Below it the overlay is counterproductive; above it, extra transmit power buys nothing.

At N = 24, R = 20 that is 10.35 m, against the 20 m the target needs: the ring radio can be
**half** the uplink and lose nothing.

## 4. The peak is flat, as pre-registered

Predicted before the grid ran: `gmax_peak` should not vary with range, because the peak happens
at the instant of the death, before any protocol can act — it is the exact expectation
`2(M−1)/M` = **1.9167** ([BREACH_WINDOW.md §1.1](BREACH_WINDOW.md)). Measured, for c ≥ 1.61:
**1.914–1.917**, identical between methods, in every cell.

The prediction was registered as a *sentinel*: if the peak did move with range, the finding
would be a different one — the formation failing to hold until t = 5 s — and `egap_pre` (the
spacing error at t = 5⁻) would say so. It does move at c = 1.21 (2.02–2.04, IQR to 2.67), and
`egap_pre` **rules the pre-event explanation out**: it is 0.0037 there, the same as everywhere
else, with zero-width IQR across all 8 seeds. The formation was intact; the inflated peak is
post-event, the gap continuing to open because the ring cannot close it.

`egap_pre` also came back **monotone in range** — 0.0037 (6.3, 8.4 m), 0.0034 (10.4, 15.7 m),
0.0030 (26.1 m). A ~19 % tighter steady state at 4× the range: small, consistent across seeds,
and a steady-state effect nobody asked the column to measure.

## 5. Caveats

**The shortest point is contaminated by the failure detector.** `AGENT_STATE_TIMEOUT` was pinned
at 5·dt, copied from `run_breach_window` — which uses it *because its channel is ideal*. To a
failure detector, an out-of-range neighbour is indistinguishable from a dead one. At c = 1.21
the links flap around the range boundary, each flap injects a fresh SAIDA/ENTRADA, and coverage
comes back **above 1.0** (several events, not one) with `G_max` peaks up to 8.2. That is the
same false-storm pathology this campaign already documented under packet loss, entered through a
different door. **The c = 1.21 row must not be read as "the mechanism fails here"** until phase
(i-b) re-runs the two short points at the campaign's FD-fix value of 20·dt. If the closing cliff
moves left, the cause was the detector; if it stays, it is the range.

**The seeds are near-replicates, so the IQR is narrow by construction.** The scenario is
deterministic — equidistant init, ideal channel, static target, no stochastic churn. The seed
picks the victim (rotationally near-equivalent positions on a uniform ring) and feeds the timer
RNGs. The IQRs above measure replication noise, **not** scenario variability, and must not be
cited as precision "for an arbitrary ring". Varying the initial configuration
(`INIT_ANGLES_EQUIDISTANT=False`, `INIT_RADIUS_RANGE>0`) is what this phase does *not* do.

**One N, one R, one fault.** Everything here is N = 24, R = 20, a single permanent death. The
2-hop rule is derived mechanically (§2) and so should transfer, but adjacent multi-death (where
the originator must redistribute a k-arc gap) and churn are untested at finite range.

**The cliff is bracketed, not resolved.** Phase (i) locates each threshold between two grid
points by design; refining is worth doing *after* knowing where they are, not before.

## 6. Prior claims this corrects

`config_param.py` carried a sizing note that was wrong twice, in opposite directions, and is now
rewritten with both rules:

1. First it asserted the **2-hop chord governs closing**. Measured at N = 10, it did not: the
   flanking survivors are out of range only briefly, the controller closes the gap, and the
   pulses circle the ring instead of crossing it.
2. The correction then said the **1-hop chord governs everything**. Also wrong — it governs
   *closing*, while the 2-hop chord governs *coverage*, and therefore the advantage. The N = 10
   sweep could not see this because it read `t_close` alone; the coverage column added for this
   phase is what separated the two.

## Related

* [README.md](README.md) — locked configuration, metric definitions, evidence index
* [BREACH_WINDOW.md](BREACH_WINDOW.md) — `t_close` / `G_max` peak definitions, and the
  `E[peak] = 2(M−1)/M` theorem this phase's prediction 1 rests on
* [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md) — entry 2026-08-03
* `comm_role_aware.py` — the per-link range handler; `config_param.py` §2 — the two rules
* `comm_results.csv` — the *other* comm axis (loss/delay/redundancy), and the FD-fix history
  that §5's caveat refers to
