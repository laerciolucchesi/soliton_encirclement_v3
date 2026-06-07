# Related Work

> Draft chapter. Framing: Computer Science / distributed systems, not control theory.
> Citations use author–year; keys map to PDF filenames in `11 Doc References/` (root,
> `encirclement control/`, `.../estimadores aplicados a encirclement/`, `soliton-inspired/`,
> `WBC/`, and `Aditional References June 2026/`). Claims are grounded in the adversarially
> verified survey (`pesquisa_literatura_encirclement.md`) and in a full pass over the
> local reference library (Doc_References v3.0 + the four topical sub-folders).

Uniform-spacing target encirclement on a ring — a team of $N$ mobile agents surrounding a
target while maintaining equal angular separation and tolerating member failure and
recovery — sits at the intersection of three research traditions that have, so far,
remained largely disjoint: (i) the *control-theoretic* literature on encirclement,
circumnavigation, and circular formation, which has solved the equilibrium geometry but
reasons about it through Lyapunov stability rather than distributed complexity; (ii) the
*distributed-computing* literature on self-stabilization, locality, and information
dissemination, which provides the right vocabulary (rounds, message complexity, diameter
lower bounds, fault coverage) but has not been applied to physical encirclement; and (iii)
the *scaling/spectral* literature, which characterizes how coordination time grows with the
network. This chapter reviews each tradition — drawing heavily on the encirclement,
estimation, soliton/wave-based-control, and formation-control collections assembled for
this thesis — identifies the closest individual neighbors, and argues that their
intersection is, to the best of our knowledge, unoccupied.

## 2.1 Encirclement, circumnavigation, and circular formation control

This is by far the most developed tradition, and the thesis library reflects it. Five
sub-threads recur.

**Cyclic pursuit and collective circular motion.** The foundational observation that equal
angular spacing on a circle can *emerge* from a purely local, leaderless rule is due to
cyclic pursuit: Marshall et al. (2004, 2006) show that a ring of unicycles, each sensing
only its successor, settles into generalized regular polygons. Variants enrich the geometry
and convergence: generalized cyclic pursuit (Mukherjee and Ghose, 2015), hierarchical
cyclic pursuit that accelerates convergence — an explicitly *scalability*-oriented result
(Smith et al., 2005), deviated cyclic pursuit producing rotating polygons around a point
(Mallik and Sinha, 2016; Rezaee and Abdollahi, 2015), collective circular motion about a
virtual beacon (Ceccarelli et al., 2008), and recent spacing-controlled cyclic-pursuit
circular formation (Fujioka and Hayashi, 2024; Park et al., 2024). Crucially, all are
*continuous control laws* analyzed by stability arguments, with no notion of round or
message complexity, and only a subset of the polygonal equilibria are stable.

**Encirclement with prescribed or arbitrary angular spacing.** A large body designs
distributed laws that drive agents to user-specified inter-agent angles around a target:
anonymous-robot encirclement at arbitrary spacing (Yao et al., 2017), distributed
encirclement with arbitrary spacing (Sen and Sahoo, 2021; Song et al., 2019),
double-integrator dynamic encirclement of anonymous agents (Huang et al., 2024),
rotating encirclement of multiple targets under nonconvex input constraints (Zhang et al.,
2020), dense even encirclement inside an annular virtual tube (Gao et al., 2022), and
range-based cooperative encirclement and tracking (Jia et al., 2023). Cyclic-pursuit circular
formation needing only leading-neighbor information and tolerating limited ranges and delays
(Zheng et al., 2023) is the closest in *locality* to our setting, yet is proved by a
Lyapunov functional, not round complexity. Order-preserving, non-uniform circle formation
by oblivious/anonymous agents in the Suzuki–Yamashita tradition (Défago and Souissi, 2008;
Wang, Xie and Cao, 2013) is the nearest *problem* match but assumes no inter-agent messaging
and no fault model.

**Bearing- and range-only circumnavigation.** When the target's position is unknown, agents
must localize while orbiting. This line couples an estimator to the controller: bearing-only
circumnavigation of a moving target (Yu et al., 2019; Ji et al., 2025), evenly-spaced
bearing-only target tracking and circumnavigation via adaptive PI estimation and Kalman
filtering (Zhou et al., 2024) and its fixed-time extension with uniform spacing (Zhou et al.,
2026), range-only circumnavigation of nonholonomic agents (Wang et al., 2024) and of a group
of non-cooperative targets in GPS-free settings (Huang et al., 2025), robust vector-field
circumnavigation under wind (Muslimov, 2023), and — notably for our fault story —
*communication-free*
bearing-only circumnavigation explicitly robust to packet loss and jamming (Sui and Deghat,
2023). The estimator-coupled framework is developed in detail in §2.3.

**Event-triggered and resilient encirclement (control sense).** Recent work attaches
"event-triggered" to surrounding/enclosing control: event-triggered surrounding control
(Babazadeh et al., 2025), event-based multi-target enclosing resilient to DoS attacks
(Zhang et al., 2025), and event-triggered moving-target encirclement by UAV formations (Jia
et al., 2024). As with the broader formation literature (Xu et al., 2020; Psomiadis and
Tsiotras, 2025), "event-triggered" here denotes *control-update / communication parsimony* —
a controller recomputes when a local measurement-error threshold is exceeded — and **not**
the *event-triggered dissemination of a topology event* that defines this thesis. We adopt
the explicit distinction **event-triggered control $\neq$ event-triggered dissemination**;
Xu et al. (2020), which shares our exact vocabulary (self-trigger, circular formation, ring
neighbors $i^{+}/i^{-}$, angular spacing $\alpha^{*}$) yet none of our distributed-systems
machinery, is the cleanest illustration.

**Learning-based and pursuit variants.** A parallel, methodologically distinct trend uses
reinforcement learning for encirclement of faster or evasive targets (Ma et al., 2019; Li
et al., 2024; Qu et al., 2025, 2026; Mu et al., 2026), cooperative pursuit forming an
angle-even trap of a faster evader (Fang et al., 2020), and GNN-based scalable perimeter
defense (Lee et al., 2023; Guerrero-Bonilla et al., 2021). These optimize empirically and,
like the control-theoretic line, provide no round/message-complexity or self-stabilization
guarantees.

**Surveys.** The area is consolidated in several reviews used here as anchors: a survey of
circular formation of multi-agent systems (Litimein et al., 2021), the canonical formation-
control survey by sensing capability (Oh, Park and Ahn, 2015), a containment-control survey
(Thummalapeta and Tsai, 2023), and a UAV target-tracking survey (Wu et al., 2025).

> Across all five sub-threads — and across ~60 papers in the local `encirclement control`
> collection — every method solves the equiangular ring geometry, yet none provides a
> distributed-complexity characterization (rounds, messages, scaling in $N$) or a
> self-stabilization guarantee in the sense of Dijkstra. The thesis's own library thus
> independently corroborates the gap.

## 2.2 Collective motion, coupled oscillators, and synchronization

A complementary tradition models circular collective motion through coupled phase
oscillators. Sepulchre, Paley and Leonard (2007) stabilize planar collective motion and
identify the Kuramoto complex order parameter $\rho = |\tfrac{1}{N}\sum_k e^{i\theta_k}|$
as the natural measure of how evenly agents are distributed; their two-time-scale design
*decouples* phase coordination from spacing actuation — but via singular perturbation inside
a single continuous controller, not via a discrete distributed algorithm — and they
explicitly flag the all-to-all coupling assumption as unrealistic, leaving sparse/local
topologies open. Geometric generalizations place synchronization on Lie groups and manifolds
(Sarlette, Sepulchre and Leonard, 2009; Markdahl et al., 2021; Butcher, 2025), and flocking
provides the foundational distributed-motion framework (Olfati-Saber, 2006). Of particular
relevance, Liu et al. (2023) drive target enclosing with a *coupled-oscillator* desired
motion combined with recursive relative localization — the closest existing tie between the
Kuramoto view and ring encirclement, though still a continuous scheme without fault-driven
dissemination or scaling analysis.

## 2.3 Target localization and estimation for encirclement

Because the target is often non-cooperative and GPS-denied, a substantial sub-literature
fuses *distributed estimation* with the encirclement controller — the "estimator-coupled"
paradigm this thesis contrasts with. Seminal distance-measurement circumnavigation under
slow drift (Shames et al., 2012) and its bearing-measurement counterpart (Deghat et al.,
2014) establish exponential localize-and-orbit guarantees. The Franchi line gives
experimentally validated distributed mutual-localization and encirclement, later extended to
2D/3D with guaranteed collision avoidance (Franchi et al., 2010, 2015). The KTH line treats
collective circumnavigation of an unknown/drifting target with convergence proofs and
resilience to agents joining and leaving (Boccia et al., 2017; Swartling et al., 2014).
Bearing-only estimation with explicit error bounds (Parayil and George, 2020), application
to algal-bloom tracking (Fonseca et al., 2019), single-time-scale TDOA estimation for UAV
formations (Doostmohammadian et al., 2022), and neural center-estimation for multiple
GPS-denied targets (Liu et al., 2024) round out the thread. These works are control- and
estimation-theoretic; the resilience they offer (e.g., to agent add/removal in Boccia et
al., 2017) is by continuous re-estimation, not by an event-triggered dissemination protocol
with bounded round complexity.

A closely related building block is **dynamic average consensus** (DAC), which several
library items treat directly: the canonical tutorial — whose motivation explicitly cites the
$O(N^2)$ cost and single-point-of-failure of centralized estimation (Kia et al., 2019) —
together with robust discrete-time DAC (Montijano et al., 2014), nonlinear DAC (Nosrati et
al., 2012), DAC robust to agents joining/leaving (Gudeta et al., 2022), and optimal-rate DAC
estimators (Van Scoy et al., 2015). DAC is the continuous counterpart of the discrete
dissemination this thesis studies, and its convergence-rate limits (§2.6) frame our scaling
argument.

## 2.4 Self-stabilization and locality in distributed computing

The distributed-systems framing rests on two mature but, until now, separate bodies of
theory. Self-stabilization, introduced by Dijkstra (1974) and developed in Dolev's monograph
(2000), is the property that a system started in an *arbitrary* state converges to and stays
in a legitimate state (convergence + closure) — the right abstraction for crash/recovery-
and message-loss-tolerant ring coordination. It is historically striking that Dijkstra's own
EWD386 posed the geometric problem of "distributing $N$ points equally on a circle,"
explicitly noted by Ghosh Dastidar and Herman (2009) as the geometric ancestor of behavioral
ring-coordination algorithms — a direct conceptual lineage from self-stabilization to
equiangular spacing. Locality theory bounds what is computable from bounded-radius
information: Naor and Stockmeyer (1995) formalize locally checkable labelings, and Linial
(1992) proves the canonical locality lower bounds (e.g., $\Omega(\log^{*} n)$ rounds for ring
coloring in the LOCAL model). With the textbook fact that information needs
$\Theta(\text{diameter})$ rounds to cross a network — $\Theta(N)$ on a ring — these supply
the $\Omega(N)$ lower bound against which our protocol is measured (the formal treatment of
$\Omega(\text{diameter})$ for global problems is given by Peleg, 2000). Most directly, a small
literature studies self-stabilizing *uniform spacing* on a ring of processes: uniform
deployment of mobile agents on a dynamic ring (Shibata et al., 2020, 2022), uniform
separation of circulating tokens (Ghosh Dastidar and Herman, 2009), and ring load balancing
(Gehrke, Plaxton and Rajaraman, 1999). These are the closest *computational* analogs of
equiangular redistribution but are purely discrete/graph-theoretic, with no continuous
actuation, no crash-/recovery-triggered event dissemination, and no spatial embedding around
a moving target.

## 2.5 Information dissemination and its latency

Our overlay is, in essence, a dissemination protocol, so the relevant baselines come from
the dissemination literature. Epidemic / gossip algorithms (Demers et al., 1987) spread an
update in $O(\log n)$ time; Karp et al. (2000) give the canonical $O(\log n)$-round,
$O(n\log\log n)$-message analysis of randomized rumor spreading, later shown asymptotically
time-optimal by Doerr and Fouz (2011). A defining contrast is that randomized gossip trades a
*diameter-proportional* latency guarantee for a high-probability bound, whereas our mechanism
is deterministic and event-triggered. Deterministic flooding and wave algorithms terminate in
time linear in the diameter; Hussak and Trehan (2023) show that even *stateless* (amnesiac)
flooding terminates in $\Theta(\text{diameter})$ rounds without maintaining a spanning
structure — a theoretical precedent for the refractory-cache, structure-free pulse forwarding
used here. The population-protocol model (Angluin et al., 2006, 2007) formalizes computation
by anonymous, finite-state agents through pairwise interactions and provides the lens for
reasoning about minimal per-agent state.

## 2.6 Scaling laws and convergence time

The cost the overlay aims to beat is the slow relaxation of purely local coordination on a
ring, and this cost can now be pinned down by citation. Olfati-Saber and Murray (2004) prove
the consensus disagreement decays as $\lVert\delta(t)\rVert \le \lVert\delta(0)\rVert
e^{-\kappa t}$ with rate $\kappa=\lambda_2$, the algebraic connectivity (Fiedler, 1973), and
note explicitly that a ring is "a relatively slow way" to reach consensus because its
$\lambda_2$ is small; the survey of Olfati-Saber, Fax and Murray (2007) consolidates this and
Boyd et al. (2006) connect averaging time to the mixing time / spectral gap. For the cycle
graph $C_N$, the Laplacian eigenvalues are $2-2\cos(2\pi k/N)$, so
$\lambda_2 = 2(1-\cos(2\pi/N)) \approx (2\pi/N)^2 = \Theta(1/N^2)$ (Spielman, 2009; Brouwer
and Haemers, 2012), yielding a $\Theta(N^2)$ stabilization time on the ring. Worst-case
convergence-time lower bounds for linear distributed averaging (Olshevsky and Tsitsiklis,
2011) reinforce this from the algorithmic side, and hierarchical cyclic pursuit (Smith et
al., 2005) confirms it from the control side by deliberately restructuring the ring to speed
convergence. The analytical baseline is independently corroborated empirically by DESYNC
(Degesys et al., 2007), whose local "jump-to-the-midpoint-of-your-two-neighbors" rule
provably converges in $O(N^2)$ rounds. DESYNC — which achieves equiangular spacing *in phase*
on a logical TDMA ring, is self-stabilizing and churn-tolerant, and builds on the
pulse-coupled-oscillator model of Mirollo and Strogatz (1990) — is thus the single most
relevant baseline: it is exactly the phase-space analog of our spacing problem, and exactly
the $O(N^2)$ relaxation our overlay is designed to break. What DESYNC lacks (a fast
dissemination overlay, a spatial/physical embedding, a continuous 2-DOF controller, and a
scaling characterization) delineates our contribution.

## 2.7 Fault tolerance, churn, and dynamic networks

Robustness to faults is treated in distributed control through resilient consensus. LeBlanc
et al. (2013) introduce *network robustness* ($r$- and $(r,s)$-robustness) and the W-MSR
algorithm, with a necessary-and-sufficient $(F\!+\!1,F\!+\!1)$-robustness condition under the
$F$-total adversary; Usevitch and Panagou (2020) show that determining robustness is NP-hard
(decision version coNP-complete) and give a MILP method; Saulnier et al. (2017) port W-MSR to
mobile robot teams via algebraic-connectivity management — the closest bridge from resilient
consensus to swarms, though it addresses adversarial *values* rather than crash/recovery
*redistribution*. On the theory side, Kuhn, Lynch and Oshman (2010) model worst-case dynamic
networks via $T$-interval connectivity and bound counting / token dissemination at $O(n^2)$
(and $O(n+n^2/T)$) rounds, while Casteigts et al. (2012) give the time-varying-graph taxonomy
whose inclusion hierarchy transfers feasibility and impossibility results across fault
regimes. Within swarms, fault-tolerant coordination is studied empirically (Chandran and
Vipin, 2024) and, most relevantly, Liu et al. (2024) achieve self-healing formation via
continuous dynamic-average-consensus estimation that is robust to ~50% packet loss and to
robots being added or removed — but through emergent re-convergence rather than an
event-triggered dissemination of the topology change, with constant message *length* rather
than a proven round-complexity improvement. (This paper is the closest existing "self-healing
under message loss" comparator and recurs in §2.10.)

## 2.8 Wave-based control and the "soliton-inspired" framing

The thesis originates from a soliton metaphor — counter-propagating, shape-preserving,
collision-surviving pulses — and a dedicated review of the `soliton-inspired/` and `WBC/`
collections clarifies what is genuinely citable. There is **no prior work that uses an actual
KdV/sine-Gordon soliton as a coordination signal in a robot swarm**: items invoking the word
"soliton" do so metaphorically for unrelated mechanisms (e.g., soliton-shaped distributions
for particle-swarm optimization, or nonlinear-Schrödinger cognitive maps), and the soliton
texts on hand (Drazin and Johnson, 1989; Dorey and Cremonesi, 2024) are physics background.
The *legitimate, rigorous* lineage is **wave-based control (WBC)**: launch/absorb
counter-propagating waves on 1-D mechanical chains (O'Connor and McKeown, 2007) and
wave-absorbing vehicular-platoon controllers that cure string instability (Martinec, Šebek
and Hurák, 2013) — directly analogous to the counter-propagating CCW/CW pulses on a ring used
here — together with **wave-equation methods on graphs**: consensus recast as a damped wave
PDE (Galbusera, Ferrari-Trecate and Scattolini, 2012) and wave propagation that lets nodes
compute Laplacian spectra locally, far faster than diffusion (Sahai, Speranzon and Banaszuk,
2011). Accordingly, this thesis frames its overlay as **event-triggered feedforward
signaling**, citing WBC and wave-PDE consensus as the technical precedent, and retains
"soliton-inspired" only as an origin-story label — an honest position consistent with the
absence of any true soliton-coordination precedent.

## 2.9 Broad context: UAV-swarm formation control

For completeness, the general UAV-swarm formation-control literature (catalogued in the
project's `Doc_References` database) provides the applied backdrop: recent surveys (Ouyang et
al., 2023; Bu, Yan and Yang, 2024; Wan et al., 2023; Luthra, 2023); leaderless/time-varying
formation and consensus, frequently under communication delays and switching topologies
(Abdessameud and Tayebi, 2011; Dong et al., 2019; Kang et al., 2023; Wang et al., 2020); and
the event-triggered *control* cluster for formation/consensus (Cheng and Li, 2019; Deng et
al., 2020; Antonio et al., 2021; Ji et al., 2023; Lin and Ling, 2023). Three items are
specifically pertinent: an explicitly distributed-systems take that runs a *Raft-style*
consensus for formation (Tariverdi and Torresen, 2023); an aerial *escort* task with
networked UAVs, the closest application sibling of encirclement (Jia et al., 2021); and
circle formation via a virtual viscoelastic model (Khaldi and Cherif, 2016). These situate
the work for a UAV audience without altering the distributed-systems contribution.

## 2.10 Closest neighbors and what separates them

Three works deserve explicit positioning.

The principal one is Gilbert, Lynch, Mitra and Nolte (2009), *Self-Stabilizing Robot
Formations over Unreliable Networks*. It simultaneously occupies several elements of our
slice: strict Dijkstra/Dolev self-stabilization (proved via timed I/O automata and a
simulation relation); a genuinely spatial/physical setting (robots distribute uniformly along
an arbitrary planar curve, of which an equiangular ring is a special case); tolerance to
crash/recovery, join/leave, and message loss; and a discrete-vs-continuous decoupling via a
Virtual-Stationary-Automata layer. It is the strongest threat to novelty and must be cited as
such. It is nonetheless separable on three decisive axes: (1) *mechanism* — coordination is a
periodic, round-based *diffusion* over virtual nodes, with no event-triggered,
counter-propagating, hop-count pulses fired by a topology change; (2) *analysis* — only
correctness/convergence is proved, with no $O(N^2)\!\to\!O(N)$ scaling, no $\Omega(N)$
ring-diameter lower bound, and no dimensionless characterization; and (3) *problem* — the
target is a fixed geometric curve rather than a moving encircled target, and the per-robot
controller is a bang-bang law rather than a 2-DOF gap-biasing controller.

The second is the DESYNC line (Degesys et al., 2007; §2.6) with the CS-pure ring-deployment
results (Elor and Bruckstein, 2011; Shibata et al., 2022): together they establish that
equiangular/uniform spacing on a ring is an established self-stabilizing-style problem with
$O(N^2)$ relaxation, but they are either timing/phase-only or communication-free (oblivious)
and never couple a fast discrete dissemination layer to a continuous actuator. The third is
the self-healing swarm of Liu et al. (2024), the closest *fault-driven redistribution* work,
which nonetheless re-converges through continuous consensus rather than event-triggered
hop-count dissemination. On the dimensionless-characterization front, the nearest neighbor is
in statistical physics: Geiss, Kroy and Holubec (2022) characterize a diffusive-to-ballistic
transition of information spreading in a delayed Vicsek model but stop short of defining a
*named* dimensionless ratio of information latency to physical response time — confirming the
Péclet-style number proposed here has, to our knowledge, no established counterpart. Robotic
pulse-coupled-oscillator work (Anglea and Wang, 2019) remains in the heading/timing domain
rather than spatial positioning, and fault-tolerant multi-robot perimeter patrol (Portugal
and Rocha, 2013) uses graph-based decision-making rather than event dissemination.

## 2.11 Positioning and gap

The table summarizes the comparison along the dimensions that define the contribution
(✓ present, ◐ partial, ✗ absent).

| Work / family | Self-stab. (Dijkstra) | Event-triggered *dissemination* | Spatial/physical encirclement | Cyber-physical 2-DOF coupling | Scaling $O(N^2)\!\to\!O(N)$ vs $\Omega(N)$ | Péclet-type number |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Cyclic pursuit / collective motion (Marshall 2004; Smith 2005; Sepulchre 2007) | ✗ | ✗ | ✓ | ◐ | ◐ (rate only) | ✗ |
| Encirclement w/ spacing + circumnavigation SOTA (Yao 2017; Sui 2023; Zhou 2026; Jia 2024) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Estimator-coupled circumnavigation (Shames 2012; Deghat 2014; Franchi 2015; Boccia 2017) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| RL-based encirclement (Ma 2019; Qu 2026) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| ET/self-triggered *control* (Xu 2020; Babazadeh 2025; Psomiadis 2025) | ✗ | ✗ (control sense) | ✓ | ✗ | ✗ | ✗ |
| DESYNC / ring deployment (Degesys 2007; Shibata 2022; Elor 2011) | ✓ | ✗ | ✗ (phase/graph) | ✗ | ◐ ($O(N^2)$ only) | ✗ |
| Self-stab. robot formations (Gilbert 2009) | ✓ | ✗ (diffusion) | ✓ | ◐ (2-layer) | ✗ | ✗ |
| Wave-based control / wave-PDE (O'Connor 2007; Martinec 2013; Sahai 2011) | ✗ | ◐ (waves, not events) | ◐ (1-D chains) | ◐ | ✗ | ✗ |
| Resilient / dynamic-network theory (LeBlanc 2013; Kuhn 2010; Liu 2024) | ◐ | ✗ | ◐ | ✗ | ◐ | ✗ |
| **This thesis** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Each individual ingredient has strong prior art — self-stabilizing spatial formation
(Gilbert et al., 2009), equiangular self-stabilization in phase (Degesys et al., 2007) and on
graphs (Elor and Bruckstein, 2011; Shibata et al., 2022), two-layer discrete/continuous
separation (Gilbert et al., 2009), counter-propagating wave control (O'Connor and McKeown,
2007; Martinec et al., 2013), event-triggered *control* (Xu et al., 2020; Babazadeh et al.,
2025), estimator-coupled circumnavigation (Shames et al., 2012; Franchi et al., 2015; Boccia
et al., 2017), and fault-tolerant coordination (LeBlanc et al., 2013; Liu et al., 2024). What
no prior work combines — and what therefore constitutes the defensible, likely-original slice
of this thesis — is the intersection:

1. an **event-triggered dissemination protocol** based on counter-propagating hop-count
   pulses fired by crash/recovery, tolerant to message loss, driving angular redistribution
   (absent as an integrated mechanism; the only counter-propagating-hop-count ring precedents
   are centralized telecommunications fault *detection*);
2. a **cyber-physical coupling** that injects the discrete protocol's output, via gap-biasing,
   into a continuous 2-DOF spacing controller *without destabilizing it*;
3. a **scaling result** that breaks the $O(N^2)$ relaxation toward $O(N)/O(\sqrt N)$ and
   matches the $\Omega(N)$ ring-diameter lower bound; and
4. a **dimensionless (Péclet-type) characterization** of *when* decoupling coordination from
   actuation pays off, as a ratio of information latency to actuation time.

**Honest novelty statement.** Fast information propagation, by itself, is not new — flooding
already achieves $O(\text{diameter})$, and wave-based control already uses counter-propagating
signals on 1-D chains. The defensible novelty lies not in speed per se but in (a) the
cyber-physical coupling that lets a discrete distributed-algorithm result drive a continuous
2-DOF controller without destabilizing it, on a *ring around a moving target*; (b) the
crash/recovery fault model with message loss under which it operates; and (c) the dimensionless
characterization of the regime in which the decoupling is worthwhile. Because an originality
claim of this kind is a proof of absence, bounded by the coverage of indexed venues, it is
stated as holding "to the best of our knowledge," with the closest neighbors — Gilbert et al.
(2009), Degesys et al. (2007), Shibata et al. (2022), Liu et al. (2024), and Xu et al. (2020)
— cited and explicitly distinguished.

---

### Notes for revision (not part of the chapter text)
- **Duplicate to clean:** `Aditional References June 2026/Liu2024b.pdf` (arXiv 2312.07523) is
  the SAME paper as `encirclement control/Liu_Lin2024.pdf` (Self-Healing Swarm via Image
  Moments). Keep one; cite as Liu et al. (2024).
- **Zhou et al. (2024)** (*Actuators* 13(9):323, DOI 10.3390/act13090323) and the range-only
  **Huang et al. (2025)** (Aerospace Sci. & Tech. 158:109924) are now in the library as
  `Zhou2024.pdf` and `Huang2025b.pdf`, and both are cited in §2.1. Note: `Huang2025b` is the
  range-only circumnavigation paper, distinct from the existing `Huang2024.pdf` (dynamic
  encirclement, TCNS) and `Huang2025.pdf` (MADRL) — keep the `.bib` keys distinct.
- Prefer **Brouwer and Haemers (2012)** or Chung over the Spielman (2009) lecture notes for
  the formal ring-spectrum citation; both are listed for manual acquisition.
- Some learning-based / surface-vehicle encirclement items (Li 2024; Qu 2025/2026; Mu 2026;
  Mardiyanto 2024) are cited only as a parallel trend; deepen or prune to taste.
- Citation labels map to the PDF filenames across `11 Doc References/` and its sub-folders;
  reconcile with your `.bib` keys when integrating (e.g., `Olfati-Saber2006`, `Liu_Lin2024`,
  `Wang2013`, `Defago2008`).
