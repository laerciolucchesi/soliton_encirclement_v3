# Handoff — finite communication range (phases 8a i, i-b, ii)

**Written for whoever is drafting the thesis chapters, not for the campaign's own record.**
Self-contained: every number below is reproduced here so this file can be read without repo
access. The canonical experiment write-up is [COMM_RANGE.md](COMM_RANGE.md); the
hypothesis→evidence→decision log is [CAMPAIGN_LOG.md](CAMPAIGN_LOG.md), entries 2026-08-03.

Data: `experiments/scaling_law/comm_range_results.csv` (80 rows, phase i),
`comm_range_results_ib.csv` (32), `comm_churn_results.csv` (80) + `comm_churn_events.csv`
(per-event). All rows `git_dirty=False`.

---

## 1. What was asked, and why it had never been asked

Every prior result in this campaign used a single 200 m communication range — larger than the
swarm's own diameter — so every agent heard every other agent and the ring's "neighbour"
relation was a *logical* construct, never a physical one. That left the overlay's central
architectural claim untested: `dual_pulse` is a **neighbour-only** protocol, so it should keep
working when the radio can only *reach* its neighbours.

The question could not be asked with one global range, and not by accident. The agent's
`AgentState` is **one broadcast serving two audiences**: the ring neighbours, and the target.
The target must hear *every* agent, or it prunes the silent ones as dead and drops them from
`alive_lambdas`, corrupting `alive_count`, the lambda map fed back to the agents, and every
M1–M7 metric — **silently**, because `G_max` and `E_gap` normalise by the number of agents the
target *heard*, so a half-observed ring still scores ≈ 1.0. Below R = 20 m the measuring
instrument dies together with the phenomenon.

GrADyS evaluates range at the **sender only**, so this cannot be fixed with a per-sender radio.
A per-link handler (`comm_role_aware.py`) supplies a range per `(sender_role, receiver_role)`
pair: the ring can be starved while the uplink stays at 200 m.

**For the thesis this is a methods contribution, not just plumbing:** the neighbour-only claim
is only falsifiable once observation is decoupled from communication.

## 2. Geometry and the normalisation

For a ring of N agents at radius R, the chord spanning k hops is `2·R·sin(kπ/N)`. Ranges are
reported as `c = range / (1-hop chord)`. At N = 24, R = 20 m: 1-hop = **5.221 m**, 2-hop =
**10.353 m**, and their ratio is

> **`2·cos(π/N) = 1.9829`** — the 2-hop threshold expressed in `c`, independent of R.

## 3. Result A — two thresholds, uniform ring, single fault (phase i)

80 runs, N = 24, R = 20 m, uplink 200 m, one deterministic permanent death at t = 5 s, 8 paired
seeds, baseline vs B2.

| range | c | coverage | `t_close_125` baseline | `t_close_125` B2 | advantage |
|---:|---:|---:|---|---|---:|
| 6.3 m | 1.21 | 0.00 | **inf** (8/8) | **inf** (8/8) | — |
| 8.4 m | 1.61 | 0.96 | 3.27 [3.25, 3.30] | 6.45 [6.40, 6.45] | **0.51×** |
| 10.4 m | 1.99 | 1.00 | 3.20 [3.19, 3.21] | 2.30 [2.30, 2.31] | **1.39×** |
| 15.7 m | 3.01 | 1.00 | 3.22 [3.17, 3.30] | 2.30 [2.30, 2.35] | 1.40× |
| 26.1 m | 5.00 | 1.00 | 3.25 [3.24, 3.26] | 2.32 [2.30, 2.35] | 1.40× |

Median [IQR], n = 8. Advantage = baseline / B2 (> 1 = overlay better). On the strict 1.10
threshold the inversion is sharper: 16.58 vs 8.00 s below, 3.42 vs 7.65 s above.

**There are two thresholds, for two different quantities.**

1. **Closing at all** needs ≈ 1 hop. The cliff at `c ∈ (1.21, 1.61]` is *identical for both
   methods*, so it belongs to the ring and the tangential controller, not to the overlay.
2. **The overlay's advantage** needs the **2-hop chord**. Below it B2 is not merely weaker but
   **2× worse than running no overlay at all**; above it the advantage appears at full strength
   and then **saturates** — 2.30–2.32 s from c = 2 to c = 5, so extra transmit power adds
   nothing.

> **Design rule:** size the ring radio at `2·R·sin(2π/N)` and stop. At N = 24, R = 20 that is
> 10.35 m against the 20 m the target needs — the ring radio can be **half** the uplink.

## 4. Result B — the mechanism (this is the part worth a section of its own)

Coverage at 8.4 m is 22/23 with `hop_sum = 23`, the full ring traversal, so the pulses circle
the ring; it is not truncation. But reading the `event_id`s — encoded `originator_seq`, so the
sequence number is an injection counter — shows the 22/23 belongs to a **different event than
the death**:

| range | landed event | kind | its `seq` | coverage |
|---:|---:|---|---:|---:|
| 8.4 m | 1 | **ENTRADA** (arrival) | **2** | 0.96 |
| 10.4 m | 1 | SAIDA (departure) | 1 | 1.00 |

Identical in all 8 seeds. The chain:

1. The victim dies; its **predecessor** (the canonical originator) injects a **SAIDA**.
2. That pulse's across-the-hole direction cannot reach the victim's **successor**, because that
   distance *is* the 2-hop chord. A receiver applies its shift only after seeing **both**
   directions, so **nobody completes it** and the event vanishes without a trace — the protocol
   logs completions, never injections, which is why only the `seq` counter exposes it.
3. The ring contracts; the successor drifts into the originator's range.
4. Locally that reads as a node **appearing**: a spurious **ENTRADA** fires and 22/23 survivors
   apply a **sign-inverted** shift for a node that never joined.

So below the chord B2 is not a weaker correction — it is **the wrong one**, which is why it
loses to doing nothing rather than merely underperforming.

> **The general statement, and the sentence the thesis should carry:** under a finite radio
> range, *"a node came into range"* and *"a node joined the ring"* are **locally
> indistinguishable**. The neighbour-only premise — the protocol's main architectural claim —
> is precisely what makes the ambiguity unresolvable, because a node's only evidence is its own
> neighbour set and both events look identical in it.

## 5. Result C — range, not the failure detector (phase i-b)

Phase (i) pinned `AGENT_STATE_TIMEOUT` at 5·dt, copied from a campaign whose channel is ideal.
With a finite range an out-of-range neighbour is indistinguishable from a dead one, so both
short points were re-run at the campaign's FD-fix value of 20·dt (32 cells).

* The closing cliff **does not move**: 0/16 cells close at 6.3 m either way. Pure range.
* The B2 inversion at 8.4 m **persists** (0.51× → 0.56×), and the *event structure is
  identical*: landed SAIDA 0, landed ENTRADA 1, `seq` 2, coverage 22/23 — all 8 seeds, both
  timeouts. Quadrupling the detector's tolerance changes nothing about which events fire,
  because the successor genuinely enters the neighbour set when the ring contracts. **Trigger
  semantics, not tuning.**

**Unpredicted, and reusable beyond this experiment:** the FD-fix costs exactly its own timeout.
Baseline 3.27 → 4.00 s and B2 6.45 → 7.20 s against a +0.75 s change; the baseline runs no
overlay, so this is the detector alone entering `t_close` **additively, as pure detection
latency**. The loss campaign could not have measured this — with infinite range the only cause
of silence was packet loss, so a longer timeout was pure robustness. With finite range there is
a **second population of silent neighbours**, the permanently out-of-range ones, and for those a
longer timeout is pure delay. `AGENT_STATE_TIMEOUT` is not a robustness dial to be maximised; it
arbitrates between two causes of silence and only one had ever been measured.

## 6. Result D — churn under locality (phase ii)

80 runs. Churn 12/min total (per-agent 0.5/min), recovery 8 s, budget 150 s, N = 24, uplink
200 m, 8 paired seeds. Under continuous churn there is no settling, so the primary metric is the
mean spacing error in regime (`t ≥ 20 s`), and `t_close` is censored by construction.

| c | timeout | `egap` baseline | `egap` B2 | advantage | B2 better in |
|---:|---:|---|---|---:|---:|
| 1.61 | 0.25 s | 0.1361 [0.1263, 0.1427] | 0.2625 [0.1881, 0.2868] | 0.55× | 0/8 |
| 1.61 | 1.0 s | 0.1328 [0.1207, 0.1414] | 0.2016 [0.1641, 0.2421] | 0.69× | 0/8 |
| 1.99 | 0.25 s | 0.1269 [0.1175, 0.1320] | 0.1906 [0.1749, 0.2193] | 0.64× | 0/8 |
| 1.99 | 1.0 s | 0.1230 [0.1135, 0.1287] | 0.1922 [0.1839, 0.1942] | 0.65× | 0/8 |
| **3.01** | 0.25 s | 0.1289 [0.1190, 0.1334] | **0.1060** [0.0992, 0.1131] | **1.19×** | **8/8** |

**Pre-registered prediction P3 — confirmed.** The prediction, committed before the grid ran, was
that the effective threshold `c**` rises above `2·cos(π/N) = 1.9829` under churn, because gaps
reach ≈ 2× the ideal and the departure pulse must cross further. The discriminator was `c = 2.0`,
which sits **+0.46 %** above the uniform threshold. It **fails** under churn (0.64×) having
**succeeded** under a single fault (1.39×). So

> **`c** ∈ (1.99, 3.01]` under churn at 12/min** — the uniform-ring threshold is *not*
> conservative once the ring is disturbed, and a radio sized exactly at the 2-hop chord is
> sized for the wrong regime.

No inversion anywhere: 0/8 in all four harmful conditions, 8/8 in the good one.

**Pre-registered prediction P4 — the spurious/legitimate ratio.** Under churn, real recoveries
produce *legitimate* ENTRADAs, so spurious ones must be separated from them. Classification rule
(fixed before the data, amended before any grid result was read — see §8): an ENTRADA is
legitimate iff the originator's **measured angular successor** at that instant has a recovery in
`[t−W, t]`, `W =` timeout + 0.5 s, reported also at 2W.

| c | timeout | spurious / legitimate | ratio |
|---:|---:|---:|---:|
| 1.61 | 0.25 s | 421 / 90 | **4.68** |
| 1.61 | 1.0 s | 214 / 91 | 2.35 |
| 1.99 | 0.25 s | 165 / 165 | 1.00 |
| 1.99 | 1.0 s | 108 / 167 | 0.65 |
| 3.01 | 0.25 s | 32 / 182 | **0.18** |

Monotone in range. **But the ratio is not the cause** — the departure pulse is. Completions by
kind, median per run (28 deaths, 24 recoveries per run in every condition):

| c | SAIDA completions | ENTRADA completions |
|---:|---:|---:|
| 1.61 | **33** | 878 |
| 1.99 | 222 | 735 |
| 3.01 | **532** | 520 |

Below the 2-hop chord the SAIDA **does not circulate at all** — 33 completions for 28 deaths per
run. The overlay is not "doing too much work for events that don't exist"; it is executing
**almost exclusively sign-inverted corrections, because the correct one never arrives**. At
c = 3.01 the SAIDA finally circulates (532 ≈ 520) and the advantage flips positive.

**Cost while it fails.** Control effort is ~2× the baseline in every condition
(0.0055 vs 0.0025 at c = 1.61; 0.0021 vs 0.0008 at c = 3.01) with **zero saturation** — the
swarm is not fighting a velocity limit, it is acting continuously on the wrong target. Time in
breach (`G_max > 1.25`) is 0.86 for B2 vs 0.51 for the baseline at c = 1.61, and inverts to 0.35
vs 0.44 at c = 3.01.

**Censoring as an instrument.** The strict criterion (1.10) separates the methods **only** at
c = 3.01: baseline 8/8 censored, B2 5/8. Where the overlay works, it closes the tight criterion
in 3 of 8 runs the baseline never closes.

**The peak is untouched.** Per-event `gmax_peak` is 1.84–1.96 in all five conditions, identical
between methods, including at c = 3.01 where B2 wins on everything else. Consistent with the
exact expectation `2(M−1)/M` established in [BREACH_WINDOW.md](BREACH_WINDOW.md): the peak
precedes any protocol response and no coordination scheme moves it.

## 7. What this changes for the thesis

1. **A new axis with a derived threshold, not a fitted one.** `2·cos(π/N)` comes from the
   geometry, is expressed in a dimensionless `c`, and was tested at a point 0.46 % above it.
   That is a sharper test than the campaign's usual sweeps allow.
2. **A defect of the overlay, characterised rather than hidden.** Below the threshold B2 is
   worse than nothing, in 40 of 40 paired comparisons across four conditions, and the reason is
   mechanical and stated exactly. A thesis that reports this is stronger than one that reports
   only the regime where the method wins.
3. **A design rule with a number** — `2·R·sin(2π/N)` for a uniform ring, and strictly more under
   churn.
4. **An architectural limit of the neighbour-only premise**, which is the thesis's own central
   claim: locality makes "came into range" and "joined the ring" indistinguishable, and no
   parameter fixes it. The fix would be a protocol change (e.g. the originator carrying the
   departed node's identity so an arrival can be checked against it), which is future work.
5. **A methods point about the failure detector** that generalises past this experiment: with a
   finite range, the FD timeout arbitrates between two populations of silence and its value is
   regime-dependent.

## 8. Caveats — read these before quoting any number

* **The seeds are near-replicates.** The scenario is deterministic by construction: equidistant
  initialisation, ideal channel (no loss, no delay), static target. The seed picks the victim
  (phase i) or the failure stream (phase ii) and feeds the timer RNGs. IQRs measure replication
  noise, **not** scenario variability, and must not be quoted as precision "for an arbitrary
  ring". Varying the initial configuration is what these phases do *not* do.
* **One N, one R, one churn rate.** Everything is N = 24, R = 20 m, and phase (ii) is 12/min
  only. The 2-hop rule is derived mechanically so it should transfer, but adjacent multi-death
  and other rates are untested at finite range.
* **`c**` is bracketed, not resolved.** Phase (ii) locates it between two grid points by design.
* **The P4 classification is heuristic**, though identity-based rather than coincidence-based.
  Three rules are reported (angular identity — primary; cyclic id; live-count) and they agree in
  direction. `W` tracks the timeout treatment, so the classifier's resolution differs between
  the two timeout conditions; the 2W sensitivity exists for that reason.
* **Two pre-analysis amendments and one erratum are on the record**, all in the runner's
  pre-registration block: (a) the P4 primary rule changed from cyclic-id successor to *measured
  angular* successor, because ids do not identify ring position — the `order_swap_frac` sentinel
  shows the two orders disagree 54–74 % of the time under churn, so the old premise was wrong
  almost always; (b) the uplink sentinel was redefined twice before the model was right (the
  target's alive count is the *union* of the live set over a trailing window, not a lagged
  sample); (c) the pre-registration text said `c = 2.0` sits "0.9 % above" the threshold — the
  correct figure is **+0.46 %**, an arithmetic error of mine that propagated into the approval
  and into `PLANO_8_9_10.md`. It does not change the prediction or the verdict; the test is
  *more* severe than stated.

## 9. Reproduction

```powershell
python experiments/scaling_law/run_comm_range_sweep.py                       # phase (i)
$env:CRS_TAG="ib"; $env:CRS_RANGES="6.3,8.4"; $env:CRS_FD_TIMEOUT="1.0"
python experiments/scaling_law/run_comm_range_sweep.py                       # phase (i-b)
python experiments/scaling_law/analyze_comm_range_ib.py                      # (i) vs (i-b)
python experiments/scaling_law/run_comm_churn_sweep.py                       # phase (ii)
```

Three assertions abort each sweep on a violated invariant rather than producing a plausible
wrong row: the effective range matrix must differ from the default, the target's live count must
track the reconstructed truth within a fixed tolerance, and the role census must be exactly
1 target + N agents with zero unknown.
