# Capítulo 7 — Robustez: comunicação degradada e churn

> Rascunho/esqueleto (PT). Status: `[a fazer]`. **Coração de sistemas distribuídos.** Trabalho
> detalhado no plano de campanha, Fase 3 (`docs/tese_estrutura.md`).

## 7.1 Tese do capítulo
O overlay foi provado no regime limpo (Cap. 5). Aqui se testa onde ele **deixa de ser plano**:
sob perda/atraso de comunicação e sob churn (falhas concorrentes). O feedforward é malha
aberta — é tão bom quanto o $\delta_D$ que recebe —, então comunicação degradada é o estresse
mais informativo.

## 7.2 A fazer (Fase 3)

> **Plano de execução detalhado:** [`docs/plano_fase3_robustez.md`](../plano_fase3_robustez.md)
> (knobs reais, Track A comm-degradada, Track B churn com os 3 fixes de código, ordem).
> Confirmado no código (protocol_agent:1102-1126): o gate de churn usa só o **sinal do delta de
> `alive_count`** → dep+arr simultâneas (líquido-zero) mascaram os eventos; ENTRADA exige
> originador canônico vivo (sem fallback). São 2 dos 3 fixes da Track B.
> **Track C (alvo MÓVEL):** [`docs/plano_fase3_trackC.md`](../plano_fase3_trackC.md) — o cenário-mãe
> (encircle de alvo em movimento); re-testa os estresses sob movimento. Hipótese-chave: o
> `consume_motion` come o shift do overlay sob manobra (sub-redistribuição). Código de movimento
> verificado e suficiente (basta `TARGET_MOTION_SPEED_XY>0`).
- [x] **Comunicação degradada — FEITO** (§7.2.1 perda + §7.2.2 atraso + §7.2.3 síntese).
- [x] **Churn / falhas concorrentes — FEITO; CONCLUSÃO REVISTA (ver §7.2.7).** A §7.2.4 (gate) foi
      **superada**: a fragilidade ao churn era **artefato do gatilho global de `alive_count`**.
      Com o **gatilho premissa-limpo** (vizinho-apenas), o overlay **AJUDA** sob churn (vantagem
      1,02–1,42); gate/M2/M5/condicional **descartados**.
- [x] **Track C — alvo MÓVEL — FEITO (§7.2.5; churn re-validado em §7.2.7).** Tracking nunca
      degradado; com gatilho-limpo + **M8 default** o overlay ajuda também sob churn+movimento
      (constante e manobra). Fase 3 ✅.
- [x] **Premissa vizinho-apenas — PROVADA (§7.2.7).** Alcance curto (25 m, ~±5 vizinhos) → overlay
      ≡ global → o algoritmo não depende de comunicação global. **+ stress combinado** (tudo junto →
      ajuda).

## 7.2.1 Resultado: robustez à perda de pacote (2026-06)

Sweep perda ∈ {0; 0,05; 0,1; 0,2; 0,4} × {baseline, B2}, N=24, τ_a=1, 3 *seeds*, evento único.
Sob perda o decaimento de $E_{\text{gap}}$ deixa de ser exponencial (o ajuste de $\tau$ fica
inválido); a métrica robusta é o **resíduo final** $E_{\text{gap}}^{\text{fim}}$ (reconfigurou
$\Rightarrow \approx 0$). Mediana de 3 *seeds*:

| perda | baseline ($E_{\text{gap}}^{\text{fim}}$ / assenta) | B2 overlay ($E_{\text{gap}}^{\text{fim}}$ / assenta) |
|---|---|---|
| 0 | 0,0001 / ✅ | 0,0000 / ✅ (vantagem 9×) |
| 0,05 | 0,0001 / ✅ | 0,0001 / ✅ |
| **0,1** | **0,0001 / ✅** | **0,16 / ❌** |
| 0,2 | 0,0012 / ✅ | 0,64 / ❌ |
| 0,4 | 0,008 / ❌ | 0,72 / ❌ |

A tabela acima usa o **default `BROADCAST_REPEATS=2`** e *sugeria* que o overlay era frágil à
perda (quebra em $\sim 0{,}1$). **Um segundo sweep refutou essa leitura.**

**A fragilidade era artefato do `BROADCAST_REPEATS`.** Varrendo
`DUAL_PULSE_BROADCAST_REPEATS \in \{1,2,3,5\}$ sob perda ($E_{\text{gap}}^{\text{fim}}$ mediano):

| perda | **repeats=1** | repeats=2 (default) | repeats=3 | repeats=5 |
|---|---|---|---|---|
| 0 | ✅ 0,0001 | ✅ 0,0000 | — | — |
| 0,05 | ✅ 0,0001 | ✅ 0,0001 | — | — |
| 0,1 | ✅ 0,0001 | ❌ 0,16 | ❌ 0,19 | ❌ 0,21 |
| 0,2 | ✅ 0,0011 | ❌ 0,64 | ❌ 0,91 | ❌ 0,65 |
| 0,4 | ⚠️ 0,007 (2/3) | ❌ 0,72 | — | — |

### Diagnóstico: a causa-raiz é o detector de falhas, não a deduplicação

Instrumentando o `events.csv` (`diag_repeats.py`): sob perda 0,2 com `repeats=2`, **uma única
falha** gerava **64 injeções / 91 eventos** — incluindo **366 ENTRADA espúrias** (sem que ninguém
se recuperasse) — com `N_new` corrompido (20–25 em vez de 23). Ou seja, não é dupla aplicação no
dedup: é uma **tempestade de eventos falsos**.

A causa é clássica de sistemas distribuídos: o gatilho do overlay é *event-triggered*, mas o
**evento é detectado amostrando liveness** — um vizinho é "morto" se não ouvido em
`AGENT_STATE_TIMEOUT`. O default era `5\,dt = 0{,}05$ s (5 *broadcasts*). A 20% de perda, **5
perdas consecutivas** têm probabilidade $0{,}2^5 \approx 3\times10^{-4}$ e ocorrem **centenas de
vezes** ao longo do *run* → vizinhos **vivos "piscam" mortos/vivos** → enxurrada de SAIDA/ENTRADA
falsos. É o **falso-positivo de detector de falhas sob perda**: não se distingue "pacote perdido"
de "nó morto". `repeats\ge 2` apenas *amplifica* (entrega o lixo); com `repeats=1` o lixo morre em
trânsito e o overlay fica **mudo** (o baseline reconfigura sozinho) — a "robustez" do `repeats=1`
era ilusória.

### Correção: dimensionar o timeout à taxa de perda

`AGENT_STATE_TIMEOUT` tornou-se *env-overridable* (default $5\,dt$ preservado). A regra é a do
detector de falhas: $\text{timeout} \gg (\text{perdas consecutivas})\cdot dt$; p.ex. $20\,dt = 0{,}2$ s
torna $p^{20}$ desprezível até $p=0{,}4$ ($0{,}4^{20}\approx10^{-8}$). Sweep de confirmação
(`repeats=2` default, timeout $0{,}2$ s, $E_{\text{gap}}^{\text{fim}}$ mediano):

| perda | baseline | **B2 (repeats=2)** |
|---|---|---|
| 0 | ✅ 0,0001 | ✅ 0,0000 |
| 0,05 | ✅ 0,0001 | ✅ 0,0000 |
| 0,1 | ✅ 0,0001 | ✅ 0,0001 |
| 0,2 | ✅ 0,0001 | ✅ 0,0003 |
| 0,4 | ✅ 0,0001 | ✅ 0,0001 |

**Achado central (final).** Com o timeout dimensionado, **o overlay é robusto à perda até $\ge 40\%$**
(B2 assenta em todas as perdas; antes quebrava em $0{,}1$), o **caso limpo não regride**, e o baseline
também melhora em $0{,}4$ (ele também sofria *flicker*). A vantagem do overlay **sobrevive a comunicação
degradada**, com **degradação graciosa**: sob perda alta o overlay completa menos eventos (fica
parcialmente mudo) e o baseline auto-corretivo assume — **sem corromper**. `pytest` 97/97 após o fix.
A robustez à perda é, portanto, um **parâmetro O(1) do detector de falhas** (teoria clássica), não um
limite fundamental do overlay. (Dados: `comm_results.csv`, `comm_results_repeats.csv`,
`comm_results_fix.csv`; diag `diag_repeats.py`.) **Implicação para o mapa de fases do Cap. 6:** o
eixo de comunicação é benigno desde que o detector de falhas seja sintonizado à perda.

## 7.2.2 Resultado: atraso de comunicação (2026-06)

Sweep `COMMUNICATION_DELAY \in \{0; 1; 5; 10\}\cdot dt$ (= 0; 0,01; 0,05; 0,1 s), sem perda,
N=24, τ_a=1 (sem perda ⇒ caso determinístico ⇒ 1 *seed*; aqui o $\tau$-ajuste é válido):

| atraso | baseline ($\tau$ / $E_{\text{gap}}^{\text{fim}}$) | B2 ($E_{\text{gap}}^{\text{fim}}$ / assenta) |
|---|---|---|
| 0 | 19,5 / 0,001 | 0,0001 / ✅ |
| 1·dt | 19,7 / 0,001 | 0,0001 / ✅ |
| 5·dt | 20,6 / 0,001 | 0,0156 / ⚠️ |
| 10·dt | 21,7 / 0,0014 | 0,108 / ❌ |

**O baseline é praticamente imune ao atraso** ($\tau$ +11% em 10·dt; resíduo ~0). O **overlay
degrada a partir de ~5·dt** e quebra em ~10·dt.

**Mecanismo distinto do da perda (verificado).** Re-rodar com `AGENT_STATE_TIMEOUT=0,3` (≫ atraso)
**não muda** o resíduo (0,108 → 0,109): sob atraso os pacotes **chegam** (só tarde), o `rxtime`
permanece recente e **nenhum vizinho expira** → não há o falso-positivo de detector de falhas que
governa a perda. A degradação do atraso vem de **estado defasado**: o *feedforward* (overlay)
precisa de posições/gaps **atuais** para aplicar o $\delta_D$ correto e para o `consume_motion`
contabilizar o giro realizado; com feedback defasado o alvo é realizado de forma imprecisa e sobra
resíduo. O *feedback* do baseline é tolerante porque apenas anula o **erro corrente**, sem depender
de timing. (Dados: `comm_results_delay.csv`, `comm_results_delaytmo.csv`.)

## 7.2.3 Síntese da Track A — dois eixos de robustez de comunicação

A robustez do overlay à comunicação degradada tem **duas condições independentes**, ambas com o
baseline como rede de segurança:

1. **Perda de pacote:** robusto se o **timeout do detector de falhas** for dimensionado à taxa de
   perda ($\gg k\cdot dt$); então robusto até $\ge 40\%$. Senão, falsos eventos corrompem (bug
   consertado).
2. **Atraso:** robusto enquanto $\text{atraso} \lesssim 5\cdot dt$; além disso o estado defasado
   degrada o $\delta_D$ (mecanismo distinto, não mitigável pelo timeout).

Fora dessas faixas, o overlay **cede ao baseline** (que é robusto a ambos) — *gating* é a defesa
de projeto. Em vocabulário CS: o **feedforward event-triggered** exige disseminação **confiável e
oportuna**; quando isso falha, o sistema degrada para o **feedback time-triggered** auto-estabilizante.

## 7.2.4 Resultado: churn (falhas/recuperações concorrentes) — Track B (2026-06)

> **⚠️ CONCLUSÃO SUPERADA pela §7.2.7.** Esta seção (e o "gate" como defesa) reflete o **gatilho
> ANTIGO**, que detectava eventos pelo `alive_count` **global** — uma violação da premissa
> vizinho-apenas. A §7.2.7 mostra que **a fragilidade ao churn era artefato desse gatilho**: com o
> gatilho local, o overlay **ajuda** sob churn e o gate fica **desnecessário/ruim**. Mantida abaixo
> como **registro da investigação** (a trilha que levou à descoberta da causa-raiz).

Churn **isolado** (sem perda/atraso), N=24. Métrica: `t_settle`/`egap_settle` (evento; via
`metrics_util`, robusto onde o ajuste exponencial falha) e `egap_avg` (churn contínuo). Reorientação
de projeto: o **baseline é sempre a rede de segurança**, então as "fragilidades" são "o overlay não
acelera" (passiva, benigna) vs "$\delta_D$ errado" (dano ativo) — caracterizar, consertar só o ativo.

**(a) Eventos discretos simultâneos — graciosos.** Cenários determinísticos (falha permanente):

| cenário | $t_{\text{settle}}$ (5%) | $E_{\text{gap}}^{\infty}$ | $N_{\text{new}}$ |
|---|---|---|---|
| 1 falha (k1) | 6,6 s | 0,0001 | 23 ✓ |
| 2 adjacentes | 23,3 s | 0,0007 | 22 ✓ |
| 3 adjacentes | 30,8 s | 0,0012 | 21 ✓ |
| 2 não-vizinhos | 6,6 s | 0,0000 | 22 ✓ |
| 3 não-vizinhos | 7,1 s | 0,0000 | 21 ✓ |

**Não-vizinhos:** $\delta_D$ **aditivo** (acumula por evento) → ~perfeito, tão rápido quanto evento
único. **Adjacentes:** **sub-correção** (só 1 evento dispara — o predecessor do 2º morto) → mais
lento, **mas o baseline fecha** o $E_{\text{gap}}$. $N_{\text{new}}$ **sempre correto**, zero eventos
espúrios. **Sem dano ativo; degradação graciosa.** (O `t_settle` separa adj/non robustamente, onde o
ajuste exponencial era ruidoso — $R^2{=}0{,}80$ no adj2.)

**(b) Churn contínuo (Poisson + recovery) — vantagem evapora + dano ocasional.** Varrendo a taxa
(`egap_avg`, baseline vs B2): vantagem **0,5–1,2, ruidosa**, com **outliers onde o B2 é muito pior**
(taxa 12/min, um *seed*: $E_{\text{gap}}^{\text{avg}}$ 0,51 vs 0,10 do baseline). Diagnóstico do
outlier (`diag_outlier.py`): o teste decisivo ($e_\tau$ **virtual** $\approx$ $e_\tau$ **físico**,
`shift` pequeno) **refutou** "viés preso"; é **feedforward incoerente** sob eventos sobrepostos (o
anel muda durante o voo do pulso → $N_{\text{new}}$ plausível-mas-velho; ~14% até impossíveis). Um
**clipe de sanidade** em $N_{\text{new}}$ (`DUAL_PULSE_N_CLIP`, A/B) **NÃO** cura ($E_{\text{gap}}$
segue ~0,45) → o dano vem dos 86% plausíveis-velhos = **incompatibilidade de regime, não bug de
álgebra**. (O clipe fica gated, default off, como guarda de sanidade.)

**(c) Gate — torna o overlay seguro.** `DUAL_PULSE_GATE_ENABLE`: cada agente conta eventos de
topologia recentes; se frequentes (> `GATE_MAX_EVENTS` em `GATE_WINDOW` s), **suprime injeção e decai
o shift** → degrada para o baseline. Resultado (varredura de taxa, vantagem $= E_{\text{gap}}^{\text{base}}/E_{\text{gap}}^{B2}$):

| taxa/min | 6 | 12 | 24 | 48 |
|---|---|---|---|---|
| sem gate | 1,22 | **0,48** | 1,03 | 0,84 |
| **com gate** | **1,00** | **1,01** | **0,99** | **1,00** |

Com o gate, **$B2 \approx$ baseline em todas as taxas e os outliers somem** (o caso 0,51→0,10 vira
0,096). O overlay **deixa de atrapalhar** sob churn.

**Conclusão (Track B).** O overlay é um **acelerador de eventos discretos esparsos** (9× no evento
único; aditivo nos multi-drops não-vizinhos). Sob **churn contínuo** sua vantagem evapora e ele pode
atrapalhar (feedforward incoerente — regime mismatch, sem fix barato); o **gate** o torna **seguro**
(degrada para o baseline auto-estabilizante). Arquitetura: **acelerador event-triggered gated ao seu
regime, com o baseline time-triggered como rede de segurança** — a mesma forma das defesas da Track A
(perda → timeout do FD; atraso/churn → gating). (Dados: `churn_sweep_results[_gated].csv`,
`diag_churn.py`, `diag_outlier.py`; `pytest` 97/97 com todos os knobs novos default-off.)

## 7.2.5 Resultado: alvo MÓVEL — Track C (2026-06)

O cenário-mãe: encircle de um alvo que se move, com a formação acompanhando. Re-testa os estresses
das Tracks A/B sob movimento, com os **fixes A/B ligados** (FD-timeout p/ perda; gate p/ churn).
Dois tipos: **constante** (velocidade fixa) e **manobra** (direção aleatória nova a cada 1 s),
3 m/s, N=24, 3 *seeds*. **Métrica dupla:** espaçamento (`E_gap`→`egap_avg`/`t_settle`, trabalho do
overlay) e **tracking radial** (`E_r`, que o overlay não deve piorar). (Sob movimento o `E_gap`
flutua pela perseguição → `egap_avg` é a métrica robusta; `t_settle` só é limpo em `fail`/`loss`
constante.)

Matriz (`egap_avg` mediano; `E_r` sempre idêntico B2=baseline — omitido):

| cenário | const base | const **B2** | manobra base | manobra **B2** |
|---|---|---|---|---|
| nenhum | 0,0001 | 0,0001 | 0,0487 | 0,0487 |
| **falha** | 0,0110 | **0,0013** | 0,0499 | 0,0546 |
| **perda** (FD-fix) | 0,0111 | **0,0031** | 0,0522 | 0,0520 |
| atraso (5·dt) | 0,0117 | 0,0155 | 0,0580 | 0,0605 |
| churn esparso (gate) | 0,0647 | 0,0659 | 0,0938 | 0,0913 |
| churn denso (gate) | 0,2086 | 0,2094 | 0,2139 | 0,2185 |

**Três achados.**
1. **Movimento constante (cruzeiro) → as conclusões A/B se mantêm.** `falha`: overlay **ajuda**
   ($\sim 8{,}5\times$ menor `egap`; `t_settle` B2 $\approx 6{,}8$ s vs baseline $42{,}7$ s — e o B2 é
   **inafetado** pelo movimento, `t_settle` $=$ estacionário). `perda` (com o FD-timeout da Track A):
   overlay **ajuda** (reconfigura sob 10% de perda + movimento — o fix carrega). `atraso` $5\cdot dt$:
   overlay **levemente pior** (a degradação de estado-defasado da Track A persiste). `churn` (gate):
   **B2 $\approx$ baseline** (o gate mantém seguro sob churn+movimento).
2. **Manobra agressiva → o benefício se dilui.** B2 $\approx$ baseline em todos (dominado pelo erro
   de perseguição $\sim 0{,}05$ + `consume_motion` comendo o shift nos transientes de mudança de
   direção). Nunca catastrófico. (Movimento **constante** não mexe no `theta_rel` → `consume_motion`
   nulo; só a **manobra** morde — confirmando a previsão, sem spin.)
3. **Tracking nunca é afetado pelo overlay.** `E_r` é **idêntico** entre B2 e baseline nas **12
   células** (o overlay é puramente tangencial; não toca o controle radial de acompanhamento).

**Conclusão (Track C).** O movimento **modula** o benefício do overlay — **pleno em cruzeiro/constante,
diluído em manobra agressiva** — mas **não introduz nenhum modo de falha novo**: nada de eventos
espúrios por movimento (cenário "nenhum": 0 injeções mesmo em manobra), os fixes das Tracks A/B
carregam intactos, e o overlay **jamais degrada o tracking radial**. O regime de competência do
overlay é **{estacionário ou cruzeiro} × {evento discreto, comunicação confiável}**; fora disso ele
**degrada graciosamente para o baseline** (via gate/diluição), seguro. (Dados: `trackC_results.csv`;
`run_trackC.py`; `pytest` 97/97.)

> **⚠️ REVISÃO (§7.2.7).** As linhas de **churn** desta matriz usavam o **gate** (e o gatilho
> antigo). Re-rodadas **sem gate** com o gatilho premissa-limpo, o overlay **AJUDA** sob churn +
> movimento constante (vantagem 1,26–1,42) em vez de empatar; e a **diluição na manobra** (achado 2)
> era **pré-M8** — com o M8 (agora default) o caso churn+manobra sobe para **1,16–1,20**. O regime de
> competência do overlay é, portanto, **bem mais amplo** do que esta seção concluiu. O achado 3
> (tracking nunca afetado) **permanece**.

## 7.2.6 Redesign proposto (overlay v2)

> **⚠️ NOTA (§7.2.7).** Este redesign foi **explorado e majoritariamente DESCARTADO**. A causa real
> do "churn problem" não era o overlay, e sim o **gatilho global de `alive_count`** (§7.2.7). Após
> consertá-lo: **M8 foi MANTIDO** (manobra; agora default) e validado; **gate, estampa-N (M2),
> idempotente (M5) e acumulação condicional** foram testados e **REJEITADOS** (M2 viola a premissa e
> não ajuda; M5 perde as quedas simultâneas; condicional é a pior em tudo; gate atrapalha com o
> gatilho consertado). A "topologia percebida graduada" (M1) não foi necessária. Mantido abaixo como
> registro da exploração.

Diagnóstico + propostas implementáveis para o overlay **deixar de piorar** (e voltar a ajudar) sob
churn denso e alvo em manobra: [`docs/plano_overlay_robusto_v2.md`](../plano_overlay_robusto_v2.md).
Resumo: churn → **feedforward incoerente** por topologia/N defasados (não viés preso, não N
impossível) → topologia percebida graduada (confiança+histerese) + δ_D por `M_eff` idempotente +
blending contínuo. Manobra → **`consume_motion` come a rotação de tracking** → consumir só a rotação
de redistribuição (M8, prioridade 1) + FF condicionado por confiança. Primeiro a implementar: **M8**
(barato, ataca a causa medida) → M1+M2 (topologia) → M4 (blending) → M5 (N-stamp/idempotente).

## 7.2.7 Resolução: a fragilidade ao churn era artefato do gatilho GLOBAL (premissa vizinho-apenas) (2026-06)

As conclusões das §7.2.4 ("gate é a defesa contra churn") e §7.2.6 (redesign do overlay) **foram
revistas** por uma auditoria de **premissa**. A premissa do projeto é **comunicação só entre
vizinhos**; o `dual_pulse` é justamente o mecanismo *distribuído* (hops vizinho-a-vizinho) para
coordenar **sem** visão global — se houvesse visão global, ele seria desnecessário.

**A auditoria.** O meio de simulação tem `COMMUNICATION_TRANSMISSION_RANGE = 200` m ≫ diâmetro do
anel (40 m) → cada agente recebe o broadcast de **todos**. A premissa vizinho-apenas é, portanto,
uma **disciplina algorítmica**, não imposta pelo meio. E o **gatilho** do overlay a violava:
detectava SAIDA/ENTRADA pelo **delta do `alive_count` GLOBAL** (contagem de todos os vivos) —
informação que um sistema de alcance curto **não teria**.

**O conserto (gatilho premissa-limpo).** O gatilho passou a ser **estritamente local**:
- **Direção SAIDA/ENTRADA** pelo **frescor do próprio succ** (`_agent_is_alive(succ anterior)`:
  morreu → SAIDA; ainda vivo → um drone entrou entre nós → ENTRADA), em vez do delta global.
- **Contagem do gate** por `succ_changed` (local).
- **Tamanho `N`** do $\delta_D$ pelo **hop-sum** (já era; a estampa global M2 foi removida).
`pytest` 97/97; det (k1/adj/non) **idêntico** ao anterior (gatilho-invariante p/ evento único).

**O resultado INVERTEU a conclusão da §7.2.4.** Re-rodando o churn **sem nenhum `alive_count` global**
(8 *seeds*; vantagem $= E_{\text{gap}}^{\text{base}}/E_{\text{gap}}^{B2}$):

| taxa/min | 6 | 12 | 24 | 48 |
|---|---|---|---|---|
| gatilho ANTIGO (global) | 1,22 | **0,48** | 1,03 | 0,84 |
| **gatilho premissa-limpo** | **1,42** | **1,21** | **1,02** | **0,96** |
| gate (premissa-limpo) | 1,21 | 0,90 | 0,81 | 0,80 |

**O overlay AJUDA sob churn** (vantagem ≥ 1 até taxa 24; neutro 0,96 no extremo) — o desastre do
rate 12 (0,48 → **1,21**) **sumiu**. E o **gate agora ATRAPALHA** (0,80–0,90): consertado o gatilho,
suprimir o overlay joga fora o benefício. **A "fragilidade ao churn" da §7.2.4 era artefato do
gatilho global**, que mis-atribuía eventos (o delta da contagem não diz *qual* vizinho mudou nem a
direção) → $\delta_D$ miscoordenados → agitação. O gatilho local dispara os eventos **certos** →
coordenação coerente.

**M8 (manobra) — agora default.** O `consume_motion` passou a abater só a **rotação comandada pelo
feedforward** ($(v_{ff}/r)\,dt$), não o $\Delta\theta$ total (que sob manobra inclui a rotação de
*tracking*) → recupera a redistribuição sob manobra. `DUAL_PULSE_CONSUME_FF_ONLY` **default True**;
seguro em todo regime (no-op no estacionário/cruzeiro).

**Re-validação completa (premissa-limpo + M8), tudo positivo:**
- **Track C churn (sem gate):** overlay **ajuda** sob churn + movimento constante (vantagem 1,26–1,42);
  **churn + manobra com M8** sobe de ~0,9 para **1,16–1,20**; tracking intacto.
- **Perda:** B2 assenta até **≥ 40%** (com o FD-timeout da §7.2.1); robustez mantida.
- **Atraso:** degrada a partir de ~5·dt (limite conhecido, inalterado).
- **Recuperação (ENTRADA) controlada:** a detecção local dispara corretamente (eventos
  `completed_entrada` + self-shift) e **ajuda** (1,88× no ciclo falha+recuperação).
- **Stress combinado** (churn 18/min + perda 10% + atraso 0,02 + manobra, **tudo junto**): B2 **ajuda**
  (vantagem 1,10–1,15), tracking intacto.
- **Cap. 6 (lei):** $\tau_{B2}$ constante em N (2,12 em N=48 ≈ 2,17 em N=24) → a lei
  $A\approx 0{,}014\,N^2/\tau_a$ **intacta** (sem reescrita).

**Prova da premissa (alcance curto).** Restringindo o meio a **25 m** (≈ ±5 vizinhos; o alvo a 20 m
ainda é ouvido, mas o lado oposto do anel — 40 m — **não**), o overlay rende **idêntico** ao alcance
global (falha: $E_{\text{gap}}^{B2}$ 0,0014 = global; churn esparso: 0,0469 vs 0,0455). **Prova que o
algoritmo é genuinamente vizinho-apenas** — não depende de comunicação global. (Nota: $<20$ m
quebra tudo, pois o alvo, a 20 m, é ouvido pelo mesmo meio → 25 m é o teste local mais apertado que
preserva o controle radial.)

**Conclusão revisada (Track B/C).** Consertada a premissa (gatilho local), o **`dual_pulse` original
(add) + M8** é **robusto em todos os regimes** — evento discreto (rápido, Cap. 6), quedas
simultâneas (aditivo), churn (ajuda/neutro, estacionário e em movimento), recuperação, perda ≤40%, e
**todos juntos** (stress) — **sem nunca prejudicar o tracking**, e **vizinho-apenas** (provado). O
único limite real é o **atraso > ~5·dt** (estado defasado). As defesas exploradas nas §7.2.4/§7.2.6
— **gate, estampa-N (M2), idempotente (M5), acumulação condicional — eram remédios para o bug do
gatilho e foram DESCARTADAS** (gated, default off; documentadas como exploração). **Arquitetura
final: gatilho premissa-limpo (vizinho-apenas) + `dual_pulse` (add) + M8.** (Dados:
`churn_sweep_results_{add,over,gate}_clean.csv`, `..._add_clean8.csv`, `comm_results_loss_clean.csv`,
`trackC_results_{churnclean,m8clean,churnm8,recover,stress,srange}.csv`, `churn_runs_n48`; `pytest` 97/97.)

## 7.3 Casos de borda já mapeados (do `dual_pulse`)
Mascaramento de `alive_count`, inferência de $N$ sob eventos simultâneos, e
ENTRADA-com-originador-falho (~3/24 em runs densos de Fase 3) — com os consertos previstos
acima. Esses casos vêm do modelo de falhas do próprio protocolo (Cap. 4, §4.4).

## 7.4 Risco
Sob churn/perda o $\delta_D$ degrada → a Fase 3 pode revelar onde o B2 deixa de ser "plano".
Esse é justamente o limite que o capítulo deve **caracterizar honestamente**, não esconder.
