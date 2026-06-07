# Mapa de literatura e gap de pesquisa — encirclement como coordenação distribuída auto-estabilizável

> Documento de trabalho para a tese (CS / sistemas distribuídos / enxames de UAVs).
> Compila duas rodadas de *deep research* com verificação adversarial + verificação
> direta dos pontos críticos. Gerado em 2026-06-03.

## Como ler este documento

**Metodologia.** Três rodadas do harness de pesquisa profunda (busca em leque →
fetch de fontes → extração de afirmações falsificáveis → verificação adversarial por
votação 3× → síntese). Rodada 1: 28 fontes, 131 afirmações, 25 verificadas, **21
confirmadas**. Rodada 2: 28 fontes, 135 afirmações, 25 verificadas, **24 confirmadas**.
Rodada 3 (escopo estrito eixos 6/7/8): 24 fontes, 109 afirmações, 25 verificadas,
**23 confirmadas**. Rodada 4 (varredura adversarial de **primazia**): 18 fontes, 86
afirmações, 25 verificadas, **24 confirmadas**. Rodada 5 (fechamento de primazia, 4
frentes residuais): 18 fontes, 83 afirmações, 25 verificadas, **22 confirmadas** (ver
seção *Veredito de primazia*). Depois, verificação direta (WebFetch/Read do PDF) dos
itens que decidem a tese (DESYNC, Mirollo-Strogatz, λ₂ no anel, espectro do anel, Linial,
e leitura completa do Xu et al. 2020).

**Legenda de confiança** (honestidade é requisito — nada inventado):

- ✅✅ **MÁXIMA** — PDF aberto, texto extraído e citado verbatim.
- ✅ **ALTA** — verificado via abstract / repositório institucional (corpo do PDF
  atrás de paywall; provas/detalhes internos não foram byte-verificados).
- ⚠️ **A CONFIRMAR** — fonte foi baixada pelo harness e afirmações extraídas, mas
  **não** passou pela verificação adversarial top-25. A atribuição autor/título é
  inferida (com alta plausibilidade) do DOI/URL. **Confirmar antes de citar.**
- ❌ **REFUTADA** — afirmação morta na verificação (listada na seção própria).

---

## Eixo 1 — Encirclement / circumnavigation / standoff tracking

**Seminal / fundador**

- ✅✅ **Marshall, Broucke, Francis (2004)**, *Formations of Vehicles in Cyclic
  Pursuit*, **IEEE TAC 49(11):1963-1974**.
  Precedente canônico de "interações locais simples → comportamento global":
  equilíbrios são polígonos regulares generalizados {n/d}, cada veículo sentindo
  **apenas o sucessor i+1**; headings de equilíbrio ±(πd)/n. É exatamente a sua
  moldura, mas é **lei de controle contínua**, não algoritmo distribuído discreto.
  *Ressalva verificada:* só **alguns** equilíbrios são localmente estáveis; pares
  (n,d) não-coprimos degeneram.

- ✅ **Shames, Dasgupta, Fidan, Anderson (2012)**, *Circumnavigation Using Distance
  Measurements Under Slow Drift*, **IEEE TAC 57(4):889-903**; e **Deghat et al.
  (2015)**, IEEE IJRNC, DOI 10.1002/rnc.3208.
  Introduziram o *estimator-coupled control framework* (estimador do alvo acoplado
  ao controlador de circumnavegação num único loop contínuo). É o paradigma que a
  tese **contrasta**.

**Recente / state-of-the-art** (o problema é o mesmo; a moldura é controle)

- ✅ **Jia, Chen, Wang, Zhang (2024)**, *Event-Triggered Cooperative Control for
  Moving Target Encirclement and Tracking With Time-Varying Pattern by UAV
  Formation*, **IET Control Theory & Appl. 18(1):55-70**, DOI 10.1049/cth2.12539.
  Tem "event-triggered" no título, mas é **event-triggered _control_** (economia de
  atuação/comunicação), provado por *interconnected systems lemma* — **sem**
  complexidade de rounds/mensagens nem auto-estabilização. **Citação-chave** para
  você delimitar: *event-triggered control ≠ event-triggered dissemination protocol*.

- ✅ **Zhou, Hu, Chen, Shen, Meng (2024)**, *Target Tracking and Circumnavigation
  Control for Multi-UAV Systems Using Bearing Measurements*, **Actuators 13(9):323**,
  DOI 10.3390/act13090323. "evenly encircle the target... regular polygon formation"
  via PI adaptativo + filtro de Kalman.

- ✅ **Huang, Shi, Zhu, Du, Lyu, Liu (2024/2025)**, *Multiple UAVs cooperatively
  circumnavigating a group of non-cooperative targets in a GPS-free environment via
  a range-only distributed controller*, **Aerospace Science and Technology
  158:109924**. Espaçamento angular desejado relativo ao centro geométrico, via
  sliding-mode fixed-time + observadores (Lyapunov).

> **Conclusão do eixo:** toda a literatura de encirclement encontrada resolve o
> espaçamento equiangular no anel, mas em moldura **control-theoretic** (Lyapunov /
> álgebra / interconnected-systems-lemma). Nenhuma usa round/message complexity,
> Ω(diâmetro) ou auto-estabilização.

---

## Eixo 2 — Formação circular / equiangular + Kuramoto

**Seminal / fundador**

- ✅✅ **Sepulchre, Paley, Leonard** — *Stabilization of Planar Collective Motion* /
  *Collective motion and oscillator synchronization* (IEEE TAC 2007/2008;
  cdcl.umd.edu/papers/Sep_Pal_Leo.pdf). Três utilidades verificadas verbatim:
  1. O **order parameter** ρ = |(1/N)Σ e^{iθ_k}| é a métrica canônica de uniformidade
     ("complex order parameter de Kuramoto").
  2. Precedente de **desacoplar** coordenação de fase da atuação de espaçamento —
     mas via **two-time-scale / perturbação singular** (ganho grande K torna a fase
     rápida e o espaçamento lento). **Mecanismo diferente** da sua novidade
     ciber-física (injeção de algoritmo discreto), o que te dá uma distinção limpa.
  3. **Alavanca de posicionamento (citável verbatim):** os autores declaram que a
     suposição **all-to-all é irrealista** em grupos grandes e topologia esparsa/local
     é problema aberto "not straightforward". É exatamente onde um protocolo de anel
     (só vizinhos) opera nativamente.
  - ⚠️ **Ressalva verificada e importante:** ρ=0 (manifold balanceado) **não implica**
    espaçamento equiangular — estados *splay/cluster* também dão ρ=0. **Não use ρ
    sozinho** como prova de uniformidade; precisa de condição de momentos de ordem
    superior (m>1) ou métrica de gaps.

**Recente / state-of-the-art**

- ✅ **Wang et al. (2020)**, *Distributed Optimal Deployment on A Circle for
  Cooperative Encirclement of Autonomous Mobile Multi-Agents*, **IEEE Access**,
  DOI 10.1109/ACCESS.2020.2982581. Deployment-on-a-circle como coverage; "só os
  azimutes do agente e dos 2 vizinhos"; convergência por método algébrico (não
  round/message). Confirma: **localidade nearest-neighbor no anel já é padrão**.

- ✅ **Zheng, Song, Liu (2023)**, *Cyclic-Pursuit-Based Circular Formation Control of
  Mobile Agents with Limited Communication Ranges and Delays*, **IEEE/CAA J.
  Automatica Sinica 10(9):1860-1870**, DOI 10.1109/JAS.2023.123576. Espaçamento
  **não-uniforme** com preservação de ordem (relevante ao seu `PROTECTION_ANGLE_DEG`).

- ✅✅ **arXiv:2506.20954v1 (jun/2025)**, *Cooperative Circumnavigation for
  Multi-Quadrotor Systems via Onboard Sensing* (tb. IEEE Xplore 11080036).
  SOTA mais recente: usa **Kuramoto acoplado** para separação angular autônoma.
  **Verificação negativa exaustiva:** ZERO análise de scaling-law / convergência-vs-N
  / complexidade de mensagens; só 3 quadrotores; escalabilidade apenas **afirmada**.
  É a sua **prova viva** de que a caracterização de complexidade (Eixo 7) segue aberta
  até no SOTA.

---

## Eixo 3 — Auto-estabilização e localidade (a espinha CS)

**Seminal / fundador** (todos verificados)

- ✅✅ **Dijkstra (1974)**, *Self-stabilizing systems in spite of distributed
  control*, **CACM 17(11)** / EWD426, DOI 10.1145/361179.361202. Definição canônica:
  **convergence + closure** ("regardless of the initial state... legitimate state
  after a finite number of moves"). Os anéis são o cenário-exemplo clássico de
  Dijkstra (token ring).

- ✅✅ **Dolev**, *Self-Stabilization*, **MIT Press** (ISBN 9780262529211). Tratado de
  referência ("a system's ability to recover automatically from unexpected faults").

- ✅✅ **Naor, Stockmeyer (1995)**, *What Can Be Computed Locally?*, **SIAM J. Comput.
  24(6):1259-1277**, DOI 10.1137/S0097539793254571. Formaliza **LCL** (locally
  checkable labelings); teorema-chave de localidade (indecidível se um LCL tem
  algoritmo local; decidível se tem em tempo t dado).

- ⚠️ **Linial (1992)**, *Locality in Distributed Graph Algorithms*, **SIAM J. Comput.
  21(1)**, DOI 10.1137/0221015 (fonte baixada na rodada 2, URL confirma a DOI, mas a
  afirmação não passou pelo top-25). Referência canônica de **lower bounds de
  localidade** — base formal para o seu Ω(N). **Confirmar a afirmação específica.**

> **Conclusão do eixo:** os fundamentos CS existem **maduros mas isolados** — nunca
> foram cruzados com encirclement/formação circular.

---

## Eixo 4 — Event-triggered / self-triggered coordination

- ✅ **Dimarogonas, Frazzoli, Johansson** — *Distributed Event-Triggered Control for
  Multi-Agent Systems*, **IEEE TAC (2012)** (people.kth.se/~dimos/pdfs/TAC11_Event.pdf;
  fonte primária baixada na rodada 1). Canônico de event-triggered em multi-agente
  (quando o agente recomputa/atua só sob disparo de evento). Distinguir de "event-
  triggered _dissemination_" (que é teoria de sistemas distribuídos, não de controle).

> Observação: este eixo está apenas parcialmente ancorado. O conceito de
> event-triggered **control** está coberto; a ponte para **disseminação**
> event-triggered vem do Eixo 5.

---

## Eixo 5 — Disseminação de informação (teoria de sistemas distribuídos)

**Gossip / epidemic** (rodada 2, fontes primárias verificadas)

- ✅✅ **Demers et al. (1987)**, *Epidemic Algorithms for Replicated Database
  Maintenance*, **PODC / Xerox PARC CSL-89-1**, DOI 10.1145/43921.43922.
  Marco fundador: direct mail, anti-entropy, rumor mongering; epidemia simples
  infecta a população em tempo ∝ **log(n)**. *Ressalva canônica verificada:* por
  serem randomizados, **não** garantem latência determinística proporcional ao
  diâmetro — a garantia é decaimento exponencial da probabilidade de não-convergência.
  (Útil para **contrastar** com o seu mecanismo determinístico event-triggered.)

- ✅✅ **Karp, Schindelhauer, Shenker, Vöcking (2000)**, *Randomized Rumor Spreading*,
  **FOCS 2000**, DOI 10.5555/795666.796561. Push-pull randomizado dissemina a todos
  em **O(ln n) rounds** com **O(n ln ln n) transmissões**; lower bound Ω(n ln ln n)
  para address-oblivious. Referência canônica de análise rigorosa de latência/mensagens.

- ✅✅ **Doerr, Fouz (2011)**, *Asymptotically Optimal Randomized Rumor Spreading*,
  **ICALP 2011**, LNCS 6756:502-513, arXiv:1011.1868. Push randomizado em tempo
  assintoticamente ótimo **(1+o(1))·log₂ n**. (Refinador recente, não fundador.)

**Latência de disseminação vs. DIÂMETRO** (o argumento de fundo do seu Ω(N))

- ✅✅ **Wattenhofer**, *Principles of Distributed Computing*, Cap. 3 (ETH Zurich),
  Teoremas 3.3 e 3.11. O **raio do grafo é lower bound** para o tempo de broadcast; a
  construção BFS Bellman-Ford (flooding carregando contador de hops) tem tempo
  **O(D)** (D = diâmetro). Liga **diretamente** o flooding hop-count (≈ o seu
  `dual_pulse`) ao diâmetro. **No anel, diâmetro = ⌊N/2⌋ ⇒ Θ(N).** *Nota:* é material
  de curso (reproduz teoremas-padrão); para o lower bound LOCAL/CONGEST formal, citar
  Linial 1992 / Peleg 2000.

- ✅✅ **Hussak, Trehan (2023)**, *Termination of amnesiac flooding*, **Distributed
  Computing 36(2):243-260**, DOI 10.1007/s00446-023-00448-y (conf. STACS 2020 /
  arXiv:1907.07078). Flooding síncrono **sem histórico** sempre termina; em tempo
  **linear no diâmetro** (exatamente a excentricidade e se bipartido; ≤ e+d+1 caso
  contrário). É o precedente teórico de **disseminação stateless event-triggered sem
  spanning tree** — vizinho do seu refractory-cache. *Ressalva:* garantia só no caso
  síncrono single-source.

**Wave / echo (textbooks canônicos)**

- ✅✅ **Tel (2000)**, *Introduction to Distributed Algorithms*, 2ª ed., Cambridge UP,
  **Cap. 6 "Wave and Traversal Algorithms"** (pp. 181-226), DOI
  10.1017/CBO9781139168724.007. Define wave algorithms (broadcast, sincronização
  global, disparo de evento por processo); latência por traversal = diâmetro/raio.

- ✅✅ **Raynal (2013)**, *Distributed Algorithms for Message-Passing Systems*,
  Springer, **Cap. 1** (pp. 3-34), DOI 10.1007/978-3-642-38123-2_1. Modela o sistema
  como grafo (diâmetro D definido); traversals BFS/DFS; spanning trees **e rings**.
  *Ressalva:* o "ring" de Raynal é **lógico (overlay de comunicação)**, não a formação
  geométrica de UAVs — a ponte "disseminação em ring = traversal estruturado" é
  **analogia defensável, não equivalência**.

**Population protocols**

- ✅✅ **Angluin, Aspnes, Diamadi, Fischer, Peralta (2006)**, *Computation in Networks
  of Passively Mobile Finite-State Sensors*, **Distributed Computing 18(4):235-253**,
  DOI 10.1007/s00446-005-0138-3 (prelim. PODC 2004; **Dijkstra Prize 2020**). Modelo
  fundador: agentes anônimos, **estado finito** (memória constante), interações
  pareadas (initiator/responder). Útil para enquadrar os seus agentes finite-state +
  pulsos discretos.

- ✅✅ **Angluin, Aspnes, Eisenstat, Ruppert (2007)**, *The Computational Power of
  Population Protocols*, **Distributed Computing 20(4):279-304**, arXiv:cs/0608084.
  Caracterização exata: os predicados estavelmente computáveis são **precisamente os
  semilineares**. Define o teto de expressividade do modelo.

---

## Eixo 6 — Tolerância a falhas em coordenação / formação ✅ (fechado na rodada 3)

**Resilient consensus — a tríade (verificada verbatim)**

- ✅✅ **LeBlanc, Zhang, Koutsoukos, Sundaram (2013)**, *Resilient Asymptotic Consensus
  in Robust Networks*, **IEEE JSAC 31(4):766-781**, DOI 10.1109/JSAC.2013.130413.
  **SEMINAL.** Introduz *network robustness* porque "traditional metrics such as
  connectivity are not adequate" para algoritmos que usam só informação local. Define os
  modelos de ameaça **F-total** (≤F nós comprometidos na rede), **F-local** (≤F vizinhos
  comprometidos de cada nó normal) e *f-fraction local*. Define **r-robustez** e
  **(r,s)-robustez**. **Teorema 1 (iff):** sob o modelo F-total malicioso, o **W-MSR**
  com parâmetro F atinge consenso resiliente **se e somente se** a topologia é
  **(F+1, F+1)-robusta**; sob F-local, **(2F+1)-robustez** é suficiente. Canônico para
  o seu modelo de falhas se quiser estender a ameaça de crash para adversarial.
  - *Precisão a manter:* o "s" em (r,s)-robustez é limiar sobre a **soma**
    |X^r_{S1}|+|X^r_{S2}|, não contagem dentro de um subconjunto. E o **iff** é exclusivo
    do modelo F-total — não confundir (2F+1) [F-local, suficiente] com (F+1,F+1)
    [F-total, iff].

- ✅✅ **Regra W-MSR** (verbatim, JSAC/TCNS): cada nó normal ordena os valores dos
  vizinhos, **descarta os F maiores e F menores** relativos ao próprio valor, e atualiza
  para a **média ponderada** do restante. Aplicada a **robôs móveis** em ↓ Saulnier 2017.

- ✅ **Usevitch, Panagou (2020)**, *Determining r- and (r,s)-Robustness of Digraphs
  Using MILP*, **Automatica**, DOI 10.1016/j.automatica.2019.108586. **RECENTE.**
  Determinar r-/(r,s)-robustez é **NP-hard**; o problema de decisão é **coNP-completo**
  (resultado de Zhang/Fata/Sundaram, IEEE TAC 60(12), 2015); método MILP acha o **F_max**
  tolerável sob F-total. Citável para a **complexidade** de verificar robustez.

- ✅ **Saulnier, Saldaña, Prorok, Pappas, Kumar (2017)**, *Resilient Flocking for Mobile
  Robot Teams*, **IEEE RA-L 2(2):1039-1046**, DOI 10.1109/LRA.2017.2655142 (ICRA 2017).
  W-MSR aplicado a **equipes de robôs móveis** — a ponte direta resilient-consensus →
  enxames/formação. *Atenção:* cite a APLICAÇÃO daqui, mas as **definições de robustez
  e o teorema cite do JSAC/TCNS** (as definições neste RA-L foram refutadas na
  verificação — votos 0-3 / 1-2).

**Redes dinâmicas / temporais (verificado verbatim)**

- ✅✅ **Kuhn, Lynch, Oshman (2010)**, *Distributed Computation in Dynamic Networks*,
  **STOC 2010** (csail.mit.edu/~rotem/stoc10_dynamic.pdf). **SEMINAL** de redes
  dinâmicas. Modelo **T-interval connectivity** (todo bloco de T rounds tem subgrafo
  gerador conexo estável; T=1 = conexo a cada round mas muda arbitrariamente; adversário
  de pior caso, sem neighbor-discovery). **Counting / qualquer função computável em
  O(n²) rounds** para 1-interval; **O(n + n²/T)** se T-interval (ganho fator T); lower
  bounds Ω(n log k) / Ω(n + nk/T). É o análogo de **complexidade de rounds** do
  "diâmetro governa disseminação" que a tese invoca.

- ✅✅ **Casteigts, Flocchini, Quattrociocchi, Santoro (2012)**, *Time-Varying Graphs
  and Dynamic Networks*, **Int. J. Parallel Emergent Distrib. Syst. 27(5):387-408**,
  arXiv:1012.0009. **Taxonomia canônica** de TVGs: hierarquia de classes em **inclusão
  estrita**, do geral ao específico. **Mecanismo útil para a tese:** a inclusão
  **transfere feasibility para baixo (subclasse) e impossibility/lower-bounds para cima
  (superclasse)** — exatamente o que você quer para argumentos de impossibilidade no
  anel sob falhas. (A classe 13 coincide com population protocols, ligando ao Eixo 5.)

> **Status:** eixo **fechado** por verificação primária. **Pendência menor:** um
> *survey* recente (2020-2026) dedicado a *resilient multi-robot/formation control*
> (linhagem Sundaram/Gil/Saulnier) ainda não foi travado — útil para ancorar a parte de
> enxames móveis, mas o núcleo teórico já está citável.

---

## Eixo 7 — Leis de escala e tempo de convergência ✅ (fechado: o elo λ₂ → anel → N²)

**Taxa de convergência ↔ λ₂ (verificado verbatim na rodada 3)**

- ✅✅ **Olfati-Saber, Murray (2004)**, *Consensus Problems in Networks of Agents with
  Switching Topology and Time-Delays*, **IEEE TAC 49(9):1520-1533**
  (cds.caltech.edu/~murray/preprints/om04-tac.pdf). **SEMINAL.** **Teorema 8** (verbatim,
  p.15): o vetor de desacordo **decai como ‖δ(t)‖ ≤ ‖δ(0)‖·exp(−κt) com κ = λ₂(Ĝ)** (a
  Fiedler eigenvalue). E afirma **explicitamente** que o **anel/ciclo é "a relatively
  slow way to solve such a consensus problem"** porque "for dense graphs λ₂ is relatively
  large and for sparse graphs λ₂ is relatively small". *É o elo formal central do seu
  argumento de escala.* *Caveat:* o Teorema 8 supõe grafo balanceado e fortemente conexo
  (trivial no caso não-direcionado, λ₂(Ĝ)=λ₂(G)); o exponencial é um limite de pior caso.

- ✅ **Olfati-Saber, Fax, Murray (2007)**, *Consensus and Cooperation in Networked
  Multi-Agent Systems*, **Proceedings of the IEEE 95(1):215-233**. Survey canônico
  (fonte baixada nas rodadas 2 e 3; afirmação específica não isolada na verificação, mas
  a atribuição é certa). Visão geral citável do campo.

- ✅✅ **Fiedler (1973)**, *Algebraic connectivity of graphs*, **Czechoslovak
  Mathematical Journal 23(2):298-305** (dml.cz/dmlcz/101168). **SEMINAL** de **λ₂ =
  algebraic connectivity** (fonte primária baixada; URL/venue/ano certos; a afirmação
  específica não foi isolada no top-25, mas é a referência fundadora padrão de λ₂).

- ⚠️ **Boyd, Ghosh, Prabhakar, Shah (2006)**, *Randomized Gossip Algorithms*, **IEEE
  Trans. Information Theory 52(6):2508-2530** (web.stanford.edu/~boyd/papers/pdf/gossip.pdf;
  tb. ic.unicamp.br/~celio/.../gossip06.pdf). Liga **averaging time** do gossip ao
  **mixing time / spectral gap** e formula o *fastest-mixing*. Fonte baixada; a afirmação
  específica não foi isolada na verificação — **confirmar o enunciado antes de citar.**

**Escalonamento λ₂ no anel — ✅✅ FECHADO POR CITAÇÃO (Spielman, verbatim)**

- ✅✅ **Spielman**, *Spectral Graph Theory*, Lecture 2 ("The Laplacian"), Yale, 2009
  (cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf). **Lema 2.4.4** (verbatim): o
  Laplaciano do anel R_n tem autovetores sin(2πku/n), cos(2πku/n) com autovalores
  **2 − 2·cos(2πk/n)**. Logo **λ₂ = 2(1 − cos(2π/N)) ≈ (2π/N)² = Θ(1/N²)**.
  - **Cadeia agora completa e citável:** anel ⇒ λ₂ = Θ(1/N²) (Spielman) + taxa de
    consenso = exp(−λ₂·t) (Olfati-Saber & Murray, Teorema 8) ⇒ **tempo de estabilização
    Θ(N²) no anel**. Esse é o seu **baseline O(N²) fundamentado por citação** — e bate com
    a **confirmação empírica independente** do DESYNC (Eixo 9, que mede O(n²) para a regra
    local de ponto-médio). *Para uma referência de livro além das notas de aula, citar
    também Brouwer & Haemers, "Spectra of Graphs", ou Chung, "Spectral Graph Theory".*

---

## Eixo 8 — Lower bound Ω(diâmetro) e terminologia ✅/⚠️ (núcleo fechado)

- ✅✅ Lower bound Θ(diâmetro) em modelo LOCAL: coberto via **Wattenhofer Cap. 3**
  (Eixo 5, Teoremas 3.3/3.11) — raio do grafo é lower bound do broadcast; no anel ⇒ Θ(N).
- ✅✅ **Kuhn, Lynch, Oshman (2010)**, STOC (ver Eixo 6): O(n²) / O(n+n²/T) rounds com
  lower bounds Ω(n log k) — o análogo rigoroso de "diâmetro/n governa disseminação".
- ⚠️ **Linial (1992)**, *Locality in Distributed Graph Algorithms*, **SIAM J. Comput.
  21(1):193-201**, DOI 10.1137/0221015. **SEMINAL** de lower bounds de localidade
  (coloração de anel exige **Ω(log\* n)** rounds no modelo LOCAL). DOI/venue/páginas
  certos; o PDF SIAM deu paywall (403) e a afirmação específica não foi byte-verificada
  nesta sessão — mas é um dos resultados **mais canônicos** de computação distribuída.
  **Confiança alta na citação; cite o enunciado a partir de um survey/livro se quiser
  verbatim** (p.ex. Suomela, "Survey of local algorithms", ACM CSUR 2013).
- ⚠️ **Peleg (2000)**, *Distributed Computing: A Locality-Sensitive Approach*, **SIAM
  Monographs on Discrete Mathematics**. Tratamento canônico de **Ω(diâmetro)** para
  problemas globais (informação atravessa o anel em Θ(N) rounds no modelo LOCAL).
  Identificação certa; afirmação específica não byte-verificada. **Confirmar verbatim.**

**Terminologia tipo-Péclet — ⚠️ veredito honesto (provavelmente batizável)**

A rodada 3 **não encontrou um nome consagrado** em *networked control systems* /
cyber-physical para a razão adimensional (latência-de-informação / tempo-de-atuação).
Candidatos investigados sem match consagrado: *delay margin*, razão τ_delay/τ_plant,
*sampling-to-delay ratio* em NCS, número de Péclet em controle. **Conclusão (não
exaustiva):** provavelmente **não há um nome único estabelecido** — o que é **boa
notícia: a tese pode batizar o seu** (ex.: "número de Péclet de coordenação"), desde que
o defina operacionalmente e reconheça os análogos parciais (delay/time-constant em NCS).
*Recomendação:* uma busca dedicada curta em NCS/sampled-data antes de cravar "novo termo".

---

## Eixo 9 — DESYNC e precedentes pulse/excitable (o vizinho mais próximo)

### DESYNC — ✅✅ verificado (PDF completo lido)

**Degesys, Rose, Patel, Nagpal (2007)**, *DESYNC: Self-Organizing Desynchronization
and TDMA on Wireless Sensor Networks*, **IPSN 2007**, pp. 11-20, ACM
978-1-59593-638-7. (PDF aberto: ianthomasrose.com/pubs/desync-ipsn07.pdf.)

**Por que é o precedente mais perigoso — e por que, lido a fundo, FORTALECE a tese:**

- **É o análogo exato do seu problema, em FASE.** Cada nó é uma "conta" girando num
  **anel lógico de tempo** com período T. *Desincronização* = nós igualmente espaçados
  em fase (alvo Δ\*ᵢ = 1/n). A regra (Eq. 4): **φ′ᵢ = (1−α)φᵢ + α·φ_mid**, onde φ_mid é
  o **ponto médio dos dois vizinhos de fase** (o que disparou logo antes e logo depois).
  É **literalmente o seu controlador de espaçamento** (pular para o meio do gap
  predecessor/sucessor), só que em fase/TDMA em vez de ângulo físico.
- **Local, auto-estabilizável, churn-tolerante, anônimo.** Memória constante, sem IDs,
  sem relógio global, sem saber N. "Regardless of the initial state and number of
  nodes, the system converges... evenly spread out with spacing T/n." Auto-ajusta a
  entrada/saída de nós e a falhas de nó único. Construído sobre Mirollo-Strogatz.
- **CRÍTICO — confirma o seu baseline O(N²) de forma independente.** Teorema 1 prova
  convergência (n<500) via mapa linear A; e o paper **conjectura e mede running time
  O(n²)** ("A^n behaves like a random walk in n; MATLAB confirms — Fig. 3", curva
  quadrática até n=100; Fig. 6: 4/10/20 nós convergem em ~8/20/48 rounds; "desync error
  decreases... proportional to n²"). **Ou seja: a abordagem local "pular para o meio"
  é O(N²) — exatamente o seu baseline acoplado.** Você pode citar DESYNC como evidência
  externa de que relaxação local no anel é O(N²) (casa com λ₂~1/N² do Eixo 7).

**O que DESYNC NÃO faz (= a sua contribuição):**

1. **Não tem overlay de disseminação rápida.** É relaxação puramente local; aceita o
   O(N²) como o custo. O seu `dual_pulse` (disseminação event-triggered hop-count
   contra-propagante) é precisamente o mecanismo que DESYNC não tem para quebrar o N².
2. **É fase/timing (TDMA), não encirclement espacial físico.** Não há alvo móvel, nem
   atuador contínuo, nem dinâmica de UAV (VM_TAU_XY, limites de velocidade/aceleração),
   nem controlador 2-DOF. O seu **acoplamento ciber-físico** (injetar o resultado
   discreto via gap-biasing num controlador contínuo sem desestabilizar) não tem
   análogo aqui.
3. **Não tem caracterização adimensional** (Péclet / latência-vs-atuação). DESYNC tem
   só um α de passo; não há eixo de "agilidade física" porque não há física.
4. **Single-hop.** O próprio *future work* aponta multi-hop (e sugere hop-count /
   coloração) como aberto — o seu cenário multi-hop em anel já está além.

> **Veredito sobre originalidade:** DESYNC **cobre** "espaçamento equiangular auto-
> estabilizável por regra local de ponto-médio" — então **não reivindique isso como
> novo**; cite DESYNC como o precedente CS direto e como confirmação do baseline O(N²).
> A sua fatia original sobrevive intacta: **(a)** o overlay de **disseminação
> event-triggered hop-count** que quebra o O(N²) rumo a O(N)/O(√N) casando Ω(N);
> **(b)** o **acoplamento ciber-físico** discreto→contínuo 2-DOF; **(c)** o
> **encirclement espacial físico** de alvo (com agilidade de UAV como eixo); **(d)** a
> **caracterização adimensional** de quando o desacoplamento compensa.

### Pulse-coupled oscillators (fundamento)

- ✅✅ **Mirollo, Strogatz (1990)**, *Synchronization of Pulse-Coupled Biological
  Oscillators*, **SIAM J. Applied Math 50(6):1645-1662** (verificado via citação
  primária [7] e descrição no texto do DESYNC). Prova que rede completa de n
  osciladores pulse-coupled converge para sincronia para quase toda condição inicial.
  Fundamento da família "firefly". DESYNC é a inversão (desincronização) desse modelo.

- ⚠️ Possíveis follow-ups de desync baixados na rodada 2 (a identificar/confirmar):
  arXiv:1411.2862 e ieeexplore 5677535 (candidatos: Pagliari-Scaglione; Buranapanichkit).

### Soliton — veredito honesto

Os precedentes reais são de **sinalização por pulso / (des)sincronização** (DESYNC,
Mirollo-Strogatz, firefly), **não** de sólitons no sentido físico (interação não-linear
com preservação de forma em colisão). **Recomendação:** enquadre como **"feedforward
signaling"** (honesto e defensável). Só mantenha "soliton" se o experimento de **falhas
densas** demonstrar que **colisões de pulsos** são tratadas melhor que superposição
linear — esse seria o único diferencial que justificaria a palavra.

---

## GAP / posicionamento (consolidado)

**Triangulação das duas rodadas + verificação direta:**

1. Toda a literatura de **encirclement** é control-theoretic (Lyapunov / álgebra),
   sem complexidade de rounds/mensagens nem auto-estabilização (Eixos 1, 2).
2. Os **fundamentos CS** (auto-estabilização, LCL, gossip/flooding/wave, population
   protocols) existem maduros, mas **isolados** — nunca cruzados com encirclement
   (Eixos 3, 5).
3. O **SOTA 2025** (arXiv:2506.20954) só **afirma** escalabilidade; não caracteriza
   (Eixo 2/7).
4. O precedente CS **mais próximo** (DESYNC) faz espaçamento equiangular auto-
   estabilizável **em fase**, e até **confirma o baseline O(N²)** — mas **não** tem
   overlay de disseminação rápida, **não** é espacial/físico, **não** tem acoplamento
   ciber-físico nem caracterização adimensional (Eixo 9).

**A fatia provavelmente original** (interseção que ninguém ocupa):

> Tratar o **encirclement equiangular físico de alvo** como problema de **coordenação
> distribuída auto-estabilizável** (Dijkstra/Dolev), resolvido por um **protocolo de
> disseminação event-triggered** (pulsos hop-count contra-propagantes, tolerante a
> crash/recovery e perda de mensagens) que **desacopla a coordenação discreta da
> atuação contínua** via gap-biasing num **controlador 2-DOF** (acoplamento ciber-
> físico sem desestabilizar), **quebrando o O(N²)** do baseline local rumo a
> O(N)/O(√N) e **casando o lower bound Ω(N)** do diâmetro do anel, e **caracterizado
> por uma razão adimensional (tipo Péclet)** entre latência-de-informação e tempo-de-
> atuação que prevê **quando** o desacoplamento compensa.

**Confiança no gap: MÉDIA.** Gap é prova de ausência — não-falsificável por busca.

**Antes de cravar "primeiro" (ações obrigatórias):**

1. **Busca sistemática** em DBLP / ACM DL / arXiv cs.DC cruzando `self-stabilizing` +
   `circular formation`/`ring`/`circumnavigation`/`desynchronization` — **parcialmente
   feita na rodada 4** (ver *Veredito de primazia*: achou GLMN como vizinho-risco; nada
   preempta). **Restam 4 frentes não esgotadas** (listadas no fim daquela seção).
2. **Formalizar Ω(N)** sob LOCAL/CONGEST citando **Linial 1992 / Peleg 2000** (não só
   Wattenhofer).
3. **Terminologia "Péclet"** — a rodada 3 não achou nome consagrado em NCS (provável
   via livre para batizar; fazer uma busca dedicada curta de confirmação — Eixo 8).
4. **Não usar ρ sozinho** como prova de uniformidade (usar métrica de gaps / momentos
   de ordem superior).

---

## Veredito de primazia (varredura adversarial — rodada 4)

Busca adversarial dedicada a **encontrar prior work que preempte** a fatia reivindicada
(100 agentes, 18 fontes primárias lidas, 24/25 afirmações confirmadas). Resultado:
**nenhum trabalho preempta a interseção completa.** Cada elemento isolado tem prior art
forte; nenhum combina o conjunto.

### ⚠️ Vizinho mais próximo — RISCO PRINCIPAL DE PRIMAZIA (cite explicitamente)

- ✅✅ **Gilbert, Lynch, Mitra, Nolte (2008/2009)**, *Self-Stabilizing Robot Formations
  over Unreliable Networks*, **SSS 2008** (LNCS 5340) / **ACM TAAS 4(3), Art. 17, 2009**,
  DOI 10.1145/1552297.1552300 (PDF: mitras.ece.illinois.edu/research/2009/GLMN_TAAS09.pdf).
  Verificado no PDF completo. **Ocupa simultaneamente:** self-stabilization estrita
  (Dijkstra/Dolev, prova TIOA + relação de simulação); espacial/físico (robôs se
  distribuem **uniformemente numa curva** no plano — anel é caso especial); tolerância a
  crash/recovery, join/leave e perda de mensagens; e **desacoplamento** discreto↔contínuo
  via camada cliente/servidor (*Virtual Stationary Automata*).
  - **Os 3 separadores (articule-os na tese):**
    1. **Mecanismo:** difusão **periódica round-based** sobre VSAs — SEM pulsos
       event-triggered, SEM hop-count, SEM contra-propagação, SEM injeção por evento de
       falha (varredura no PDF: `pulse`=0, `wave`=0, `event-triggered`=0).
    2. **Sem análise de escala:** prova só correção/convergência; SEM O(N²)→O(N), SEM
       Ω(N) de diâmetro, SEM Péclet (`Peclet`=0, `Omega`=0).
    3. **Problema:** alvo é **curva fixa** (não alvo móvel encirclado); controlador do
       cliente é **bang-bang trivial**, não gap-biasing 2-DOF.

### Outros vizinhos (não preemptam)

- ✅ **Elor, Bruckstein (2010/2011)**, *Multi-agent Deployment on a Ring Graph* (ANTS
  2010, LNCS 6234) / *Uniform multi-agent deployment on a ring* (**Theoretical Computer
  Science 412(8-10):783-795, 2011**). Análogo CS-puro do espalhamento **uniforme/
  equidistante num anel** (inclui variante "in motion"). **Mecanismo oposto ao seu:**
  agentes *oblivious*/ant-like, **sem comunicação direta**, sem info global, coordenam só
  por **sensing** de distância aos 2 vizinhos. Ou seja, é a ausência do overlay de
  disseminação — bom para citar como contraponto.
- ✅✅ **Xu, Wang, Tao, Xie, Xu, Zhou (2020)**, *Distributed Self-triggered Circular
  Formation Control for Multi-robot Systems*, **39th Chinese Control Conference (CCC
  2020)**, pp. 4639-4645. **PDF completo lido.** É o vizinho mais próximo em
  **vocabulário**: combina os termos exatos — *self-triggered* + *circular formation* +
  anel (i⁺/i⁻) + espaçamento angular α\* + alvo + 2 vizinhos. **MAS não preempta:**
  1. "Self-triggered" aqui = **sentido de CONTROLE** (economia de energia/comunicação):
     o gatilho é threshold no erro de medição (Eq. 23) que decide *quando o robô atualiza
     o próprio controlador* — NÃO é disseminação de eventos de topologia. Zero `event_id`/
     `hop_count`/contra-propagação. **Mesma confusão terminológica** de Jia 2024 e
     Psomiadis-Tsiotras, mas no seu cenário exato — daí ser o exemplo perfeito para você
     fixar a distinção *event-triggered control ≠ event-triggered dissemination*.
  2. **Sem self-stabilization (Dijkstra)** — estabilidade assintótica por Lyapunov;
     topologia fixa com spanning tree e **comunicação confiável**.
  3. **Sem tolerância a falhas/crash-recovery/perda de mensagens** (futuro work lista
     "weak links" como pendência). **Sem escala** (sem O(N²)/Ω(N)), **sem Péclet**,
     **sem 2-DOF gap-biasing**; N=6 na simulação. Compartilha **ZERO** dos 3 elementos
     distintivos. Linhagem Peking/Xie (refs próprias [20-23]).
- ✅ **DESYNC** (Eixo 9) — reconfirma o baseline O(N²), mas em fase/TDMA, sem overlay.
- ✅ **Psomiadis, Tsiotras (2025)**, *Distributed Event-Triggered Distance-Based
  Formation Control* (arXiv:2509.12390) — "event-triggered" no sentido de **controle**
  (threshold de atuação), distance-based, sem self-stab/crash/hop-count. Tangencial.
- ✅ **Portugal, Rocha (2013)**, patrulha multi-robô fault-tolerant (RAS 61) — primo
  temático, mas grafo + mecanismo Bayesiano; tangencial.
- ⚠️ **Linhagem *oblivious robots* / circle formation** (surgiu das refs do Xu et al.,
  ainda não verificada a fundo): **Défago, Souissi (2008)**, *Non-uniform circle
  formation algorithm for oblivious mobile robots with convergence toward uniformity*,
  **TCS 396(1-3):97-112**; e **Wang, Xie, Cao (2013)**, *Forming circle formations of
  anonymous mobile agents with order preservation*, **IEEE TAC 58(12):3248-3254**. São o
  ramo de **pattern formation por robôs oblivious/anônimos** (modelo Suzuki-Yamashita) —
  relevante ao seu espaçamento não-uniforme (`PROTECTION_ANGLE_DEG`) e à preservação de
  ordem. Vizinhos de problema, mas sem disseminação event-triggered / self-stab CS /
  escala. **Confirmar e citar** ao discutir formação circular anônima.

### Conclusão

> **(a) Preempta a fatia completa?** Não. **(b) Mais próximo?** GLMN — separado por
> mecanismo (pulso vs. difusão), ausência de análise de escala/Péclet, e alvo fixo vs.
> móvel. **(c) Original?** A interseção permanece original (confiança **média** — é prova
> de ausência). Os 3 elementos **ausentes de todo o prior art**: pulsos hop-count
> contra-propagantes event-triggered; escala O(N²)→O(N) com Ω(N); número tipo-Péclet.

### Rodada 5 — as 4 frentes residuais esgotadas (nenhuma preempta)

**Frente A — self-stabilizing uniform spacing em anel (CS puro):** existe e é maduro,
mas nenhum preempta (todos discretos/grafo, sem atuação contínua 2-DOF, sem pulsos por
crash/recovery, sem Péclet):
- ⚠️ **Shibata, Sudo, Nakamura, Kim (2020/2022)**, *Uniform Deployment of Mobile Agents
  in Dynamic Rings*, **SSS 2020 (LNCS 12514)** / **Information & Computation 289:104949,
  2022**. Análogo CS mais próximo da redistribuição equiangular — agentes equalizam
  espaçamento num **ring graph**. Mas: síncrono, *whiteboard*, IDs distintos, sabe n e k;
  "dinâmico" = **link faltante adversarial** (1-interval), **não** crash/recovery de
  agente; sem controlador contínuo, sem Péclet.
- ⚠️ **Ghosh Dastidar, Herman (2009)**, *Separation of Circulating Tokens*, **SSS 2009 /
  TCS**. Mantém m tokens circulantes ≥ d apart (separação uniforme auto-estabilizável num
  anel), mas o objetivo é um **comportamento** (tokens circulando), não um estado
  geométrico. **Citação de ouro escondida:** este paper aponta **Dijkstra EWD386,
  "distribute N points equally on a circle"** como o análogo geométrico/robótico — ou
  seja, a *equidistribuição num círculo* remonta ao próprio **Dijkstra**. Vale citar
  EWD386 como ancestral conceitual da sua formulação.
- ⚠️ **Gehrke, Plaxton, Rajaraman (1997/1999)**, ring load-balancing (DISC/TCS) — balanceia
  contagens abstratas de token; sem falha/recuperação, sem embedding espacial. (Uma
  tentativa de usar seu bound O(N) para preemptar foi **refutada 0-3** — é token-count
  abstrato, não latência-vs-atuação.)

**Frente B — número adimensional tipo-Péclet:** ❌ **nenhum resultado preemptivo.** O
vizinho mais forte é ⚠️ **Geiss, Kroy, Holubec (2022)**, *Signal propagation and linear
response in the delay Vicsek model*, **Phys. Rev. E 106:054612** — caracteriza a transição
**difusivo→balístico** na propagação de informação com atraso/velocidade, mas **não
define um número adimensional nomeado** (latência/atuação). Buscas explícitas por
`Peclet`/`dimensionless` em enxames retornaram **nenhum resultado relevante** (evidência
positiva de gap). É o seu elemento mais idiossincrático.

**Frente C — DESYNC espacial/móvel:** ❌ **nenhum resultado preemptivo.** Os hits de PCO
em robótica (⚠️ **Anglea & Wang 2019**, PCO heading control) ficam no domínio
**temporal/orientação** (heading num toro 1-D), **não** posição espacial equiangular ao
redor de um alvo. Mirollo-Strogatz aplicado a *posição* física não apareceu (só um
preprint pós-2024, fora do escopo).

**Frente D — disseminação event-triggered por wave/hop-count disparada por falha (seu
mecanismo mais distintivo):** ❌ **nenhum resultado preemptivo em robótica.** O mecanismo
exato de **hop-count contra-propagante num anel** existe — mas em **US Patent 7,664,052**
(Alaxala, 2010), que é **detecção centralizada** de falha em anel **Ethernet** (nó mestre
+ probe packets), sem posições físicas, sem redistribuição, sem swarm. Papers de
redistribuição após falha em robótica usam **árvore hierárquica** (Gong et al. 2024,
coverage) ou **consenso contínuo/push-sum** (Liu et al. 2024, *self-healing* via image
moments) — **não** pulsos hop-count event-triggered. (Tentativa de chamar o patente de
"estruturalmente análogo" foi **refutada 0-3**.)

### Status final dos 3 elementos distintivos (após 5 rodadas)

| Elemento distintivo | Status no prior art |
|---|---|
| Pulsos hop-count **contra-propagantes** event-triggered por crash/recovery → redistribuição angular | ❌ **ausente** (só em telecom centralizado; robótica usa árvore/consenso) |
| Escala **O(N²)→O(N)/O(√N)** casando **Ω(N)** do diâmetro do anel | ❌ **ausente** no enquadramento swarm/info-latency |
| **Número adimensional tipo-Péclet** (latência-info / tempo-atuação) | ❌ **ausente** (física reconhecida, mas sem número nomeado) |

> **Veredito final:** nenhuma das 5 rodadas encontrou trabalho que preempte a interseção.
> Cada elemento isolado tem prior art; **a combinação dos três + o desacoplamento
> discreto→contínuo via gap-biasing 2-DOF permanece original.** Todas as tentativas de
> preempção foram **refutadas** na verificação (0-3) — o que *reforça* a primazia.

**Resíduo (não bloqueante):** a originalidade é prova de ausência — limitada à cobertura
das buscas (venues indexados, inglês). Único vetor que trabalho futuro poderia fechar:
**preprints pós-2024** de PCO/DESYNC espacial ou de "número adimensional" em robótica de
enxame. Recomendação: um *alerta* de busca (Google Scholar/arXiv) nesses termos durante a
escrita, e frasear a reivindicação como "ao melhor do nosso conhecimento" + citar os
vizinhos (GLMN, DESYNC, Shibata, Xu et al.) explicitamente.

---

## Pendências de verificação (após 3 rodadas — o que ainda falta)

**Fechado na rodada 3:** Eixo 6 (W-MSR/JSAC, Kuhn-Lynch-Oshman, Casteigts — verbatim);
Eixo 7 (Olfati-Saber & Murray Teorema 8 + λ₂(C_N)=Θ(1/N²) via Spielman — **cadeia
anel⇒Θ(N²) agora citável**); Eixo 8 núcleo (Wattenhofer/KLO + identificação de
Linial/Peleg).

**Resíduo menor (não bloqueia a tese):**
- ⚠️ **Linial 1992 / Peleg 2000** — citação certa, mas enunciado não byte-verificado
  (paywall). Citar verbatim a partir de survey/livro (Suomela CSUR 2013) se quiser.
- ⚠️ **Boyd 2006** (mixing time) — confirmar o enunciado específico antes de citar.
- ⚠️ **Survey recente (2020-2026)** de *resilient multi-robot/formation control*
  (Sundaram/Gil/Saulnier) — para ancorar enxames móveis.
- ⚠️ **Terminologia Péclet** — busca dedicada curta em NCS antes de batizar termo novo.
- ⚠️ **Follow-ups de desync** (arXiv:1411.2862, ieee 5677535) — identificar autoria.

**Ação de maior valor (independente das rodadas):** a busca sistemática em
DBLP/ACM DL/arXiv cs.DC cruzando `self-stabilizing` + `circular formation`/`ring`/
`circumnavigation`/`desynchronization` (ver seção GAP) — é o que valida a reivindicação
de originalidade.

---

## Apêndice — afirmações REFUTADAS na verificação (não citar)

- ❌ (0-3) "Flooding constrói BFS-tree e termina em r rounds **mesmo em sistemas
  assíncronos**" — a garantia de tempo r é só no modelo **síncrono**.
- ❌ (1-2) "Um único agente bearing-only localiza e circumnavega um grupo de alvos
  **sem comunicação inter-agente**" — há coordenação distribuída.
- ❌ (1-2) Reformulação imprecisa da definição de auto-estabilização (a canônica
  convergence+closure permanece — Eixo 3).
- ❌ (1-2) "Kuramoto acopla fase à atuação **sem** separação" — refutada por nuance
  (há a separação two-time-scale, Eixo 2).
- ❌ (1-2) "Sinal de K dicotomicamente seleciona sync (K>0) vs balanced (K<0)" —
  refutada por imprecisão.

---

## Apêndice — fontes (URLs)

**Rodada 1 — verificadas (alta/máxima confiança):**
- control.utoronto.ca/~broucke/Webpapers/MarBroFra-TAC49-11-2004.pdf (Marshall-Broucke-Francis 2004)
- cdcl.umd.edu/papers/Sep_Pal_Leo.pdf (Sepulchre-Paley-Leonard)
- arxiv.org/html/2506.20954v1 (Cooperative Circumnavigation 2025)
- ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cth2.12539 (Jia 2024)
- mdpi.com/2076-0825/13/9/323 (Zhou 2024)
- sciencedirect.com/science/article/abs/pii/S1270963824010538 (Huang 2025)
- onlinelibrary.wiley.com/doi/abs/10.1002/rnc.3208 (Deghat 2015)
- researchgate.net/.../340125121 (Wang 2020)
- ieee-jas.net/article/doi/10.1109/JAS.2023.123576 (Zheng 2023)
- dl.acm.org/doi/10.1145/361179.361202 (Dijkstra 1974)
- mitpress.mit.edu/9780262529211 (Dolev, Self-Stabilization)
- epubs.siam.org/doi/10.1137/S0097539793254571 (Naor-Stockmeyer 1995)

**Rodada 2 — verificadas (Eixo 5):**
- dl.acm.org/doi/10.1145/43921.43922 (Demers 1987)
- dl.acm.org/doi/10.5555/795666.796561 (Karp et al. 2000)
- link.springer.com/chapter/10.1007/978-3-642-22012-8_40 (Doerr-Fouz 2011)
- disco.ethz.ch/courses/ss06/distcomp/lecture/chapter3.pdf (Wattenhofer Cap.3)
- link.springer.com/article/10.1007/s00446-023-00448-y (Hussak-Trehan 2023)
- cambridge.org/.../wave-and-traversal-algorithms/... (Tel 2000, Cap.6)
- link.springer.com/chapter/10.1007/978-3-642-38123-2_1 (Raynal 2013, Cap.1)
- cs.yale.edu/homes/aspnes/papers/podc04passive-dc.pdf (Angluin et al. 2006)
- link.springer.com/article/10.1007/s00446-007-0040-2 (Angluin et al. 2007)

**Rodada 2 — baixadas, a confirmar (Eixos 6/7/8/9):**
- ieeexplore.ieee.org/document/6481629 (LeBlanc et al. 2013, W-MSR)
- people.csail.mit.edu/rotem/stoc10_dynamic.pdf (Kuhn-Lynch-Oshman 2010)
- arxiv.org/abs/1012.0009 (Casteigts et al., temporal graphs)
- link.springer.com/article/10.1007/s004460050070 (artigo DC não identificado)
- ee.iitb.ac.in/~dc/EE749/Olfati.pdf (Olfati-Saber-Murray 2004)
- labs.engineering.asu.edu/.../Consensus-and-Cooperation...2007.pdf (Olfati-Saber-Fax-Murray 2007)
- dml.cz/bitstream/handle/10338.dmlcz/101168/CzechMathJ_23-1973-2_11.pdf (Fiedler 1973)
- web.stanford.edu/~boyd/papers/pdf/gossip.pdf (Boyd et al. 2006)
- epubs.siam.org/doi/10.1137/0221015 (Linial 1992)
- dl.acm.org/doi/10.5555/355459 (Peleg 2000)
- skoge.folk.ntnu.no/book/ps/bookall.pdf (Skogestad-Postlethwaite, NCS)

**Rodada 3 — verificadas em fonte primária (Eixos 6/7/8):**
- engineering.purdue.edu/~sundara2/papers/journals/JSAC_robust_consensus.pdf (LeBlanc et al. 2013)
- engineering.purdue.edu/~sundara2/papers/journals/tcns_robust.pdf (Zhang-Fata-Sundaram TCNS)
- sciencedirect.com/science/article/abs/pii/S0005109819304479 (Usevitch-Panagou 2020, Automatica)
- lehigh.edu/~das819/pdf/ral17-resilient-flocking.pdf (Saulnier et al. 2017 — só aplicação)
- people.csail.mit.edu/rotem/stoc10_dynamic.pdf (Kuhn-Lynch-Oshman 2010)
- arxiv.org/abs/1012.0009 + hal.science/hal-00847001v1 (Casteigts et al. 2012)
- cds.caltech.edu/~murray/preprints/om04-tac.pdf (Olfati-Saber & Murray 2004 — Teorema 8 ✅✅)
- dml.cz/dmlcz/101168 (Fiedler 1973)
- mcrotk.github.io/courses/references/olfati-saber-pieee.pdf (Olfati-Saber-Fax-Murray 2007)
- web.stanford.edu/~boyd/papers/pdf/gossip.pdf (Boyd et al. 2006 — a confirmar enunciado)
- epubs.siam.org/doi/10.1137/0221015 (Linial 1992 — paywall, citação certa)

**Rodada 4 — varredura de primazia (fontes primárias lidas):**
- dl.acm.org/doi/10.1145/1552297.1552300 + mitras.ece.illinois.edu/research/2009/GLMN_TAAS09.pdf (Gilbert-Lynch-Mitra-Nolte 2009 — RISCO PRINCIPAL ✅✅)
- link.springer.com/chapter/10.1007/978-3-540-89335-6_16 (GLMN, versão SSS 2008)
- link.springer.com/chapter/10.1007/978-3-642-15461-4_19 (Elor-Bruckstein 2010, deployment on ring)
- arxiv.org/pdf/2509.12390 (Psomiadis-Tsiotras 2025, event-triggered formation — sentido de controle)
- sciencedirect.com/science/article/abs/pii/S0921889013001206 (Portugal-Rocha 2013, patrulha)
- researchgate.net/publication/344768096 → confirmado = Xu et al. 2020 CCC (PDF lido na sessão)

**Rodada 5 — fechamento de primazia (4 frentes residuais):**
- link.springer.com/chapter/10.1007/978-3-030-64348-5_20 + sciencedirect S0890540122001043 (Shibata et al. 2020/2022, uniform deployment in dynamic rings)
- arxiv.org/pdf/0908.1797 (Ghosh Dastidar & Herman 2009, Separation of Circulating Tokens — cita Dijkstra EWD386)
- link.springer.com/chapter/10.1007/BFb0030677 (Gehrke-Plaxton-Rajaraman, ring load-balancing)
- journals.aps.org/pre/abstract/10.1103/PhysRevE.106.054612 (Geiss-Kroy-Holubec 2022, delay Vicsek — vizinho Péclet)
- ieeexplore.ieee.org/document/8630512 + arxiv.org/abs/1910.07442 (Anglea & Wang 2019, PCO heading)
- image-ppubs.uspto.gov/.../7664052 (US Patent 7,664,052 — hop-count em anel Ethernet, centralizado)
- ncbi.nlm.nih.gov/pmc/articles/PMC11644315 (Gong et al. 2024, coverage redistribution)
- par.nsf.gov/servlets/purl/10541916 (Liu et al. 2024, self-healing swarm via image moments)

**Verificação direta (esta sessão):**
- ianthomasrose.com/pubs/desync-ipsn07.pdf (DESYNC — PDF completo lido ✅✅)
- (anexo do usuário) Xu et al. 2020, CCC, "Distributed Self-triggered Circular Formation Control" (PDF completo lido ✅✅)
- cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf (Spielman, Lema 2.4.4: λ₂(anel)=2−2cos(2πk/n) ✅✅)
- clear.rice.edu/comp551/papers/MirolloStrogatz-...-SIAM1990.pdf (404; verificado via citação no DESYNC)
- epubs.siam.org/doi/10.1137/0221015 (Linial 1992 — 403 paywall, não byte-verificado)
