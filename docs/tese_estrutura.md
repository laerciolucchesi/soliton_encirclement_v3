# Tese — estrutura reposicionada (trilema + Option B)

> **Índice/roadmap.** Documento de trabalho, editável. A **prosa draft de cada capítulo**
> está em `docs/draft/` (1 arquivo por capítulo, em PT). Os resultados numéricos que sustentam
> cada capítulo estão em `experiments/scaling_law/` (scripts, CSVs, PNGs). As decisões e o
> histórico de achados estão na memória do assistente (`.claude/projects/.../memory/`).
> **Mudança 2026-06:** Trabalhos Relacionados virou **Cap. 2** próprio (era entrelaçado); os
> capítulos técnicos foram renumerados (+1). Fonte do Cap. 2: `docs/related_work.md` (EN) →
> `docs/draft/cap2_trabalhos_relacionados.md` (PT).

**Área:** Ciência da Computação / sistemas distribuídos / enxames de UAVs / controle de formação.
**Tipo de contribuição:** princípio + caracterização (NÃO "meu método é mais rápido").
**Validação:** simulação em larga escala + drone real / SITL (pequena escala).

---

## Frase-tese (uma sentença)

Manter espaçamento uniforme num anel de UAVs sob falhas é coordenação distribuída
auto-estabilizante. O controlador local enfrenta um **trilema fundamental** —
estabilidade × velocidade × tamanho do enxame — porque a informação só se propaga
montada na própria dinâmica física ("difusão de ganho"). Proponho um **overlay** que
computa o alvo de reconfiguração por **hop-count distribuído** e o injeta por
**feedforward, por fora do ganho** do controlador, com um **feedback ciente do plano
(2-DOF)**. Isso **quebra o trilema** (estável + rápido + escalável), e eu **caracterizo —
de forma adimensional, válida para qualquer plataforma — quando e quanto compensa.**

---

## Os 8 capítulos (índice dos drafts)

| # | Capítulo | Draft | Status |
|---|---|---|---|
| 1 | Problema, modelos e enquadramento de sistemas distribuídos | [cap1_problema_modelos.md](draft/cap1_problema_modelos.md) | `base pronta` |
| 2 | **Trabalhos Relacionados** (estado da arte; ~90 refs) | [cap2_trabalhos_relacionados.md](draft/cap2_trabalhos_relacionados.md) | `rascunho pronto` |
| 3 | O trilema do controlador local (resultado negativo central) | [cap3_trilema.md](draft/cap3_trilema.md) | `provado` |
| 4 | Algoritmo distribuído de redistribuição por hop-count | [cap4_algoritmo_hopcount.md](draft/cap4_algoritmo_hopcount.md) | `implementado; análise a formalizar` |
| 5 | Quebrando o trilema: feedforward 2-DOF (Option B/B2) | [cap5_feedforward_2dof.md](draft/cap5_feedforward_2dof.md) | `confirmado; refinamento` |
| 6 | Caracterização adimensional (**N²/τ_a** + robustez ao dt) | [cap6_caracterizacao_adimensional.md](draft/cap6_caracterizacao_adimensional.md) | `núcleo medido; redação inicial` |
| 7 | Robustez: comunicação degradada e churn | [cap7_robustez.md](draft/cap7_robustez.md) | `a fazer` |
| 8 | Validação, escopo e ponte para o hardware | [cap8_validacao_hardware.md](draft/cap8_validacao_hardware.md) | `a fazer` |

> A descrição detalhada de cada capítulo agora vive no respectivo arquivo de draft. Abaixo
> ficam o **log de resultados** e o **plano de campanha** (a trilha acionável).

---

## Resultados-chave (estado em 2026-05)

```
Tempo de estabilização (tau_fit do modo lento), falha única controlada:
                                    N=24      N=40      N=50
baseline (ganho fixo alto)          7.08      12.26    140.1(INSTÁVEL)   -> O(N) só até ~N=40, depois estoura
baseline (ganho estável ~1/N)      19.48      54.79     85.35            -> O(N^2.02) (preço da estabilidade)
Option A (ganho estável)           11.63      42.02     74.71  adv 1.7->1.1 ENCOLHE  -> não escapa
Option B-min (estável, scale 0.5)   3.27       7.78     12.20  adv 6->7 CRESCE   (escapa a direção; tau ~N^1.8)
Option B-min (estável, scale 1.0)  16.51      43.00     62.59  adv 1.2->1.4      (pior: duplo-drive em over-drive)
Option B2  (estável, scale 1.0)     2.17       2.13      2.12  adv 9->26->40     *** tau PLANO -> ESCAPE COMPLETO ***
```

RESOLUÇÃO (2026-05): o diagnóstico decompôs a cauda do B-mínimo -> a fase rápida do feedforward
é PLANA (~1.5s, independente de N) + um resíduo ~25% limpo pelo feedback lento O(N^2)
(tau_slow ≈ base_tau). O conserto: viés cancelador COMPLETO (B2: succ+(s_succ-s_self),
pred-(s_pred-s_self)) remove o duplo-drive, o feedforward (scale=1.0) entrega ~100%, o resíduo
some. Resultado: tau_B2 = 2.17/2.13/2.12 s em N=24/40/50 -> PLANO (R^2 0.93-0.95), estável,
vantagem CRESCE como N^2 (9->26->40x). O TRILEMA ESTÁ QUEBRADO, SEM PENALIDADE DE ESCALA.
(A previsão "scale=1.0 achata" só falhava com o viés MÍNIMO; com o COMPLETO, vale.)
Caveats para travar na tese: 3 pontos de N, 1 seed, evento único, regime limpo (ganho estável,
equidistante, sem perda de comm) -> falta a campanha (N maior, multi-seed, regimes de robustez).

QUALIFICAÇÃO (confirmação N grande, 2026-05): o tau plano vale só até N~50. Em N=75/100 o B2
QUEBRA (tau 2,12 -> 71,3 -> 86,1; R² 0,95 -> 0,75; 2 seeds batem -> real). Hipótese principal
(a DIAGNOSTICAR, não assumir): DUAL_PULSE_TTL_HOPS=50 pequeno demais para N>~51 (um receptor
precisa de pulsos de até N-1 hops; o originador, N hops para o retorno) -> delta_D incompleto ->
feedforward parcial -> resíduo lento. É limite de ALGORITMO (afeta A/B/B2), ajustável por env.
Próximo: diagnosticar (cobertura de shift em N=50 vs 75), depois consertar o TTL e re-testar.

RESOLVIDO (2026-05): diagnóstico confirmou o TTL (cobertura caiu 96%->36%->1% e max_hop travou
em 50). TTL virou env-overridable; com TTL=3N a confirmação MULTI-SEED PASSA: B2 tau ~2,1s em
N=50/75/100, ambos seeds, R² 0,95-0,97 -> tau PLANO TRAVADO até N=100. A quebra era config, não
o mecanismo. Cap. 5 fechado (mecanismo). MENSAGENS (diag_messages.py): disseminação O(N) (~3,9
payloads/agente em N<=50, = 2 direções x 2 BROADCAST_REPEATS; por-agente O(1)); em N=75/100 infla
(8,6/6,0) por pulsos espúrios de flapping de vizinho no transiente (efeito histerese-vs-gap, da
Fase 0). Baselines sendo re-rodados com orçamento adequado para corrigir o artefato N^1,36.

## O que mudou (honesto)

- ANTES: "overlay corta o expoente O(N)->O(N^0.6)" — frágil (era medido no regime de ganho
  fixo que é INSTÁVEL em N grande).
- AGORA: "o controlador local tem um TRILEMA estabilidade-velocidade-N; o overlay com
  feedforward (2-DOF) o QUEBRA" — mais profundo, honesto, e CS de verdade.
- Hipóteses minhas que os experimentos REFUTARAM (e isso é força, não fraqueza): flapping de
  histerese (era instabilidade de ganho); overshoot no ramo ágil (era duas escalas de tempo);
  "Option A escapa o trilema" (refutado -> levou ao Option B).

## Itens abertos (resumo)

- [x] Cap. 5: B2 dá tau PLANO (provado: ~2.1s em N=24/40/50, travado até N=100). MECANISMO fechado.
- [ ] Travar o mecanismo numa CAMPANHA de validação (plano detalhado abaixo).
- [ ] Formalizar a teoria (Cap. 3 trilema, Cap. 4 algoritmo, Cap. 5 argumento 2-DOF).
- [ ] Caps. 6-8: caracterização adimensional, robustez, hardware.
- [x] Cap. 2: Trabalhos Relacionados rascunhado (5 rodadas de pesquisa + varredura de primazia).

---

# Plano de campanha (acionável)

Estado (2026-06): o MECANISMO está provado E a **Fase 1 (lei de escala) está fechada**
(baseline Θ(N²) e B2 plano até N=100, 2 seeds, vantagem ~N², mensagens O(N), 9 figuras).
A **Fase 0** está quase toda feita (offset confirmado; multi-seed com 2 seeds; histerese
adimensional analisada e adiada p/ Fase 3). A **Fase 2** está parcial (agilidade + lei
`scale*(τ_a)`). Falta o grosso de **robustez (Fase 3)**, a formalização (Fase 4) e o hardware
(Fase 5) — além de itens residuais marcados abaixo. Infra em `experiments/scaling_law/`.
(Numeração de capítulos já atualizada para a estrutura de 8 capítulos.) Status por item ↓.

## Itens ADIADOS conscientemente (voltar depois) — decidido 2026-06

**Decisão:** priorizar o **colapso adimensional (Fase 2)** antes da blindagem, porque (i) as
sims para N grande estão lentas e (ii) faz mais sentido blindar depois de a espinha dorsal
estar melhor pensada/testada. **Adiados (NÃO abandonados):**
- **Multi-seed amplo** (resíduo Fase 0): `INIT_RADIUS_RANGE` + `INIT_ANGLES_EQUIDISTANT=False`
  + vários seeds; reportar mediana + dispersão. Blinda contra "começou de um círculo perfeito".
- **Robustez do feedforward a erro no δ_D** (resíduo Fase 1): injetar ruído (±5/10/20%) no δ_D
  e medir a margem da malha aberta. Blinda o ponto frágil do 2-DOF **antes** da Fase 3.
- **Ponto N=150** (resíduo Fase 1, opcional).
- **Histerese adimensional** (já estava adiada p/ Fase 3 — churn).

**Gatilho para retomar:** após o colapso adimensional (Cap. 6) estar fechado — e, se o colapso
valer, ele pode **baratear estas próprias rodadas** (ver o item do período de controle `dt` na
Fase 2: rodar com `dt` maior + reescalar por Pe).

## Fase 0 — Higiene / guardas metodológicos  [Cap. 8 §8.4]  — QUASE TODA FEITA
- [🔁] `HYSTERESIS_RAD` adimensional: ANALISADO — na falha única limpa a ordem cíclica é
      preservada (sem switching), então só morderia em N~126 e sob REORDENAÇÃO (churn/alvo
      móvel). Item real, mas pertence à **Fase 3**, não agora. (Diagnóstico na memória.)
- [~] Multi-seed: PARCIAL — confirmação com **2 seeds** (variando o nó que falha) PASSA
      (B2 determinístico). FALTA: `INIT_RADIUS_RANGE` + ângulos não-equidistantes (robustez a
      condição inicial); reportar mediana + dispersão.
- [x] Offset de regime: CONFIRMADO — late_std ~0 e egap_final ~0.001 em N≤100 (B2 settled).

## Fase 1 — Travar a lei de escala (Caps. 3 e 5)  — ESSENCIALMENTE FEITA
- [x] Lei de escala FECHADA: baseline Θ(N²) (tau 19.5/54.8/85.4/183.6/311.4 s em
      N=24/40/50/75/100, R² 0.89-0.94, fit N^1.97 ancorado em λ₁=(2π/N)²); B2 PLANO (~2.1 s em
      N=50/75/100, 2 seeds, R² 0.95-0.97); vantagem ~N² (9→25.7→40.3→87.4→149×).
      (`run_baseline_longbudget.py` + `run_optionB_test.py`.) FALTA opcional: ponto N=150.
- [x] Mensagens/falha do B2 vs N: O(N) LIMPO — 4·(N-1) payloads totais (nó morto não emite),
      O(1)/agente. (`diag_messages.py`; re-confirmado no pacote de figuras.)
- [ ] Robustez do feedforward a ERRO no `delta_D` (injetar ruído; medir margem da malha aberta) — NÃO feito.
- [+] BÔNUS feito: pacote de 9 figuras p/ orientador (`experiments/scaling_law/figures/`, `FIGURAS.md`).

## Fase 2 — Caracterização adimensional (Cap. 6)  — PARCIAL
- [~] Eixo de agilidade: FEITO em parte — varredura `VM_TAU_XY` (N=24, sintonia fixa) dá vantagem
      NÃO-monotônica (corcova; pico ~2.7× em τ~0.5); o teste de `DELTA_SCALE` mostrou que a
      corcova é artefato de sintonia → lei `scale*(τ_a)` DECRESCENTE (1ª lei adimensional);
      com ganho ADAPTADO a vantagem vira MONOTÔNICA. FALTA: grade fina de scale × τ_a, multi-seed,
      N>24. (Infra: `run_agility_sweep.py` + B2.)
- [ ] Leis adimensionais / COLAPSO: checar se `tau_B2` adimensionalizado COLAPSA contra os grupos
      (Pe=N*dt/tau_a, dt/tau_a, N) — a prova de "vale para qualquer drone". NÃO feito.
- [ ] Mapa de fases (N x agilidade x comm): onde o overlay paga vs onde é limitado por atuação. NÃO feito.
- [ ] **Período de controle `dt` (atual 10 ms) como eixo do Pe** — questão levantada 2026-06.
      `dt` entra em Pe=N·dt/τ_a, então VARRER `dt` percorre o eixo de Péclet sem mexer em N nem
      τ_a. Estudar: (a) até onde `dt` MAIOR (menos frequência de sensing/mensagens/controle/
      atuação) mantém o τ plano vs onde a **estabilidade sampled-data quebra** (gancho ao trilema,
      Cap. 3 — `dt` maior encolhe a margem de estabilidade); (b) **PAYOFF METODOLÓGICO:** se o
      colapso por Pe valer, sims com `dt` maior (muito mais rápidas) podem ser reescaladas →
      ataca diretamente a lentidão das sims em N grande. (Refinamento possível: hoje sensing,
      mensagens, controle e atuação compartilham o mesmo `dt`; desacoplar a taxa de comunicação
      da taxa de atuação seria um estudo mais rico — registrar como sub-item.)

## Fase 3 — Robustez: comunicação degradada e churn (Cap. 7)  [coração de sistemas distribuídos]
- [ ] Comm degradada: varrer `COMMUNICATION_DELAY` e `COMMUNICATION_FAILURE_RATE` (hoje = 0).
      Como B2 degrada; sensibilidade do feedforward (malha aberta) e do broadcast de `dp_shift`
      sob perda; papel de `BROADCAST_REPEATS`.
- [ ] Churn / falhas concorrentes: Poisson denso (`FAILURE_MEAN_FAILURES_PER_MIN` alto) + recovery;
      e saída+entrada simultâneas. CONSERTAR a detecção: gate robusto a mudança líquida-zero de
      `alive_count`, carimbo de N na injeção, fallback do sucessor para ENTRADA-com-originador-falho.
      Métrica: cobertura de falhas (fração reconfigurada corretamente).
- [ ] Assincronia (ordem de firing intra-tick): robustez do `dp_shift` e dos pulsos.

## Fase 4 — Formalização teórica (Caps. 3, 4 e 5)  [backbone fino, ~20%]
- [ ] Cap. 3: trilema linearizado — gap espectral do anel, o fator N da normalização, a margem
      de estabilidade dos modos altos, e o O(N^2) sob ganho estável (medições já dão os expoentes;
      falta o argumento de 1-2 páginas).
- [ ] Cap. 4: corretude + complexidade do hop-count (O(N) rounds, O(N) mensagens, diâmetro-ótimo).
- [ ] Cap. 5: o argumento 2-DOF — por que B2 dá tau ~ T_FF (malha aberta) independente de N, e por
      que o viés cancelador COMPLETO zera o duplo-drive.

## Fase 5 — Ponte para o hardware (Cap. 8)
- [ ] SITL (ex.: Gazebo/ArduPilot) com poucos agentes: medir `tau_a` real da plataforma, calcular
      Pe, PREVER o ganho do overlay, e CONFIRMAR voando.
- [ ] (Se possível) demo com 3-5 drones reais — mesmo pequeno, blinda muito a defesa.
- [ ] Discussão honesta: idealização sem-colisão (drone falho congelado; vivos redistribuem em
      2pi) — evasão de colisão ao redor do nó falho é trabalho futuro.

## Riscos / o que pode complicar (manter à vista)
- O colapso adimensional pode NÃO ser limpo (grupos errados) -> iterar (Buckingham-pi).
- Feedforward malha-aberta é tão bom quanto o `delta_D`; sob churn/perda o delta_D degrada ->
  a Fase 3 pode revelar onde o B2 deixa de ser "plano".
- "tau plano" foi provado no regime LIMPO; em regimes sujos a dependência de N pode reaparecer.
