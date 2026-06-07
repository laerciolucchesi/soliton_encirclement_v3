# Capítulo 5 — Quebrando o trilema: integração por feedforward (2-DOF)

> Rascunho (PT). Status: `[confirmado; refinamento em curso]`. Resultados:
> `experiments/scaling_law/` (run_optionB_test.py e correlatos).

Este capítulo toma o alvo de deslocamento $\delta_D$ do Cap. 4 como **dado** e foca em **como
injetá-lo** no controle de modo a escapar do trilema do Cap. 3. A questão é de arquitetura de
controle (1 vs 2 graus de liberdade), não de algoritmo distribuído.

## 5.1 Option A (viés de gap através do ganho) NÃO escapa

A primeira tentativa — "Option A" — usa $\delta_D$ para **enviesar os gaps** que o controlador
de espaçamento enxerga (a integração "Option A" do `dual_pulse`: ele modifica `pred_gap`/
`succ_gap`, não alimenta um canal $u_\text{prop}$). O problema: o viés passa **através do
mesmo ganho** instável do controlador local. O experimento (P3) refuta: a vantagem sobre o
baseline **encolhe** com $N$ (1,68 → 1,30 → 1,14). Option A não desacopla — herda o trilema.

## 5.2 Option B / B2 (feedforward direto, por fora do ganho) ESCAPA

A solução — "Option B" — injeta $\delta_D$ por **feedforward direto, por fora do ganho**, com
um **feedback ciente do plano**: cada agente subtrai o viés cancelador construído a partir dos
*shifts* dos vizinhos. A análise da "briga" feedforward-vs-feedback leva à **identidade do
viés cancelador completo**:

$$\text{succ} - \text{pred} = 2\,s_i - s_\text{succ} - s_\text{pred},$$

onde $s_i$ é o shift do próprio nó. A versão **mínima** do viés (succ + s\_succ / pred − s\_pred)
já escapa a direção, mas deixa um resíduo; a versão **completa B2** (succ + (s\_succ − s\_self),
pred − (s\_pred − s\_self)) remove o **duplo-drive**, o feedforward entrega ~100% e o resíduo
some.

## 5.3 Resultado: trilema quebrado, sem penalidade de escala

```
Tempo de estabilização (tau do modo lento), falha única, regime limpo:
                                 N=24     N=40     N=50
Option A (ganho estável)        11.63    42.02    74.71   adv 1.7->1.1 ENCOLHE   -> não escapa
Option B-min (scale 0.5)         3.27     7.78    12.20   adv 6->7 CRESCE        (escapa; tau ~N^1.8)
Option B2 (scale 1.0)            2.17     2.13     2.12   adv 9->26->40          *** tau PLANO -> ESCAPE COMPLETO ***
```

O diagnóstico decompôs a cauda: a **fase rápida do feedforward é PLANA** (~1,5 s, independente
de $N$) e havia um resíduo ~25% limpo por um feedback lento $O(N^2)$. O viés cancelador
**completo** (B2) remove o duplo-drive; com `scale=1.0` o feedforward entrega ~100% e o resíduo
desaparece. Resultado: $\tau_{B2} = 2{,}17/2{,}13/2{,}12$ s em $N=24/40/50$ — **plano**
($R^2$ 0,93–0,95), estável, com vantagem que **cresce como $N^2$** (9 → 26 → 40×). **O trilema
está quebrado, sem penalidade de escala.**

Confirmação em $N$ grande (multi-seed): após corrigir o TTL (Cap. 4), o $\tau$ plano se mantém
**até $N=100$** (B2 $\tau \approx 2{,}1$ s em $N=50/75/100$, dois seeds, $R^2$ 0,95–0,97). A
quebra observada em $N=75/100$ antes do conserto era configuração (TTL), não o mecanismo.

## 5.4 Caveats honestos (a travar na campanha)

O mecanismo está provado no **regime limpo**: 3 pontos de $N$ originais (depois estendidos),
poucos seeds, evento único, ganho estável, início equidistante, sem perda de comunicação.
Falta a **campanha** ($N$ maior, multi-seed sistemático, regimes de robustez) para transformar
"prova de mecanismo" em "evidência de tese" — ver Caps. 6–8 e o plano de campanha
(`docs/tese_estrutura.md`, Fases 1–3).

> **A formalizar (Fase 4):** o argumento 2-DOF — por que B2 dá $\tau \approx T_\text{FF}$
> (malha aberta) independente de $N$, e por que o viés cancelador completo zera o duplo-drive.
