# Plano de execução — Colapso adimensional (Fase 2 / Cap. 6)

> Plano acionável (PT). Objetivo: provar a caracterização **adimensional** — que a vantagem
> do overlay (e o τ normalizado) **colapsa numa curva única** quando plotada contra um número
> de Péclet, ou seja, **vale para qualquer drone**, independente de N, agilidade e período de
> controle. NÃO disparar sweeps longos ainda; este doc fixa grupos, dados que faltam e scripts.

## 1. Grandezas físicas e parâmetros reais

| Símbolo | Significado | Parâmetro no código | Default | Env-overridable? |
|---|---|---|---|---|
| `N` | nº de agentes | `NUM_AGENTS` | — | ✅ |
| `dt` | **período de controle** (round) | `CONTROL_PERIOD` | 0.01 s | ✅ (feito 2026-06; override validado em 0.05) |
| `τ_a` | constante de atuação (agilidade) | `VM_TAU_XY` | 1.0 s | ✅ |
| `T_FF` | constante do feedforward | `DUAL_PULSE_T_FF` | 1.0 s | ✅ (piso ~ τ_a) |
| `scale` | escala do δ_D | `DUAL_PULSE_DELTA_SCALE` | 0.5 | ✅ |
| `K` | ganho do controlador | `K_E_TAU` | 25 (regime estável: **250/N**) | ✅ |
| `τ` | **saída**: tempo de estabilização (modo lento) | medido (`tau_fit`) | — | — |

Config B2 canônica (de `run_largeN_confirm.py`): `DUAL_PULSE_INTEGRATION=B2`,
`DUAL_PULSE_DELTA_SCALE=1.0`, `K_E_TAU=250/N` (ganho estável), `DUAL_PULSE_TTL_HOPS=3N`.

## 2. Grupos adimensionais (π) — hipótese a testar

- **Péclet de coordenação:** `Pe = (latência de informação)/(tempo de atuação) = N·dt / τ_a`.
  (Latência de disseminação do overlay ≈ atravessar o anel = O(N) rounds = N·dt segundos;
  tempo de atuação = τ_a.) **É o eixo central.**
- Grupos secundários: `dt/τ_a` (resolução de amostragem) e `T_FF/τ_a` (folga do feedforward).
- **Saídas adimensionais a colapsar:**
  - **Vantagem** `A = τ_base / τ_B2` (a métrica de "quando o overlay paga") vs `Pe`.
  - `τ_B2 / T_FF` (deve ficar ~O(1), plano) e `τ_base / (N²·dt)` (deve ficar ~O(1) — Θ(N²)).
- **Ponte CS:** `rounds = τ / dt`. Baseline ~Θ(N²) rounds; B2 ~Θ(N) rumo ao limite Ω(N).
  *Previsão-chave do colapso:* em **rounds**, o resultado deve ser **invariante a `dt`**
  (até o limite de estabilidade amostrada) — é isso que justifica rodar com `dt` maior.

> **A prova "vale para qualquer drone":** rodar combinações DIFERENTES de (N, τ_a, dt) e
> mostrar que os pontos caem na MESMA curva `A(Pe)`. Se colapsarem, a caracterização está feita.

## 3. O que JÁ temos (inventário no disco)

- **Eixo N** (τ_a=1.0, dt=0.01, B2 estável): `baseline_long_results.csv` (Θ(N²), N=24..100) +
  `optionB_results_B2scale1.csv` / `largeN_results.csv` (B2 plano até N=100). ✅ usável.
- **Eixo agilidade** (N=24, τ_a∈{0.2,0.5,1,2}): `agility_results.csv` — ⚠️ **config ERRADA**
  (rodado em **Option A**, ganho fixo; `run_agility_sweep.py` não seta B2/250-N/T_FF). **Descartar
  para o colapso; re-rodar em B2.**
- **Eixo scale** (N=24, τ fixo): `deltascale_results.csv` — deu a lei `scale*(τ_a)` decrescente.
  Útil como ponto de partida para a adaptação dos knobs, mas idem: confirmar config.
- **Eixo dt:** ❌ inexistente (CONTROL_PERIOD nem é env-overridable).
- Análise: `analyze_relaxation.py` + `tau_fit_from_csv` (exp-fit da cauda de E_gap, já validado).

## 4. O que FALTA (trabalho novo)

1. ✅ **[código] FEITO (2026-06):** `CONTROL_PERIOD` agora é env-overridable e `VM_UPDATE_RATE`
   foi amarrado a ele → um `dt` único governa controle + broadcasts + mobilidade. Override
   validado (sim a `CONTROL_PERIOD=0.05` deu timestamps espaçados 0.05). Default 0.01 inalterado
   (B2 N=24 reproduziu 2.17 s). `RAMP_TICKS` em *ticks* (escala com dt); timeouts escalam com dt.
2. **[sim] Varredura 2D (N × τ_a) em B2, ganho estável, knobs ADAPTADOS.** O ponto sutil: ao
   varrer τ_a é preciso **adaptar `T_FF`(τ_a) e `scale`(τ_a)** (T_FF tem piso ~τ_a; a lei
   `scale*(τ_a)` é decrescente). Sem adaptar, o `T_FF=1.0` fixo contamina o eixo τ_a e a corcova
   reaparece (artefato já diagnosticado). Medir `τ_base` e `τ_B2` → `A`. Grade inicial barata:
   `N∈{8,16,24}`, `τ_a∈{0.2,0.5,1,2}` (12 células × 2 métodos).
3. **[sim] Eixo `dt`** em 1–2 células (N,τ_a): varrer `dt∈{0.01,0.02,0.05,0.1}` (+`0.005`
   opcional como âncora de convergência) e testar
   (a) **invariância em rounds** (τ/dt constante a Pe fixo) e (b) onde a **estabilidade amostrada
   quebra** (dt grande → ciclo-limite; gancho Cap. 3). Define a janela de dt utilizável.
4. **[análise] `analyze_collapse.py`** (novo): para cada run calcula `Pe=N·dt/τ_a`, plota
   `A` vs `Pe` (log-x) sobrepondo TODAS as combinações (N, τ_a, dt); ajusta a fronteira `Pe*`
   onde `A=1` (info-limitado vs atuação-limitado) e checa o colapso (dispersão em torno da curva).

## 5. Estratégia BARATA (a alavanca do `dt`)

Para fugir da lentidão em N grande, **mapear o colapso em N pequeno**:
- Fazer a varredura 2D acima em `N∈{8,16,24}` (rápido), cobrindo o eixo `Pe` via τ_a **e** dt.
- Estabelecida a curva `A(Pe)`, **validar só alguns pontos de N grande** (já temos N=50..100 em
  τ_a=1.0) e mostrar que caem na curva.
- Se a invariância-em-rounds valer, **rodar N grande com `dt` maior** (menos ticks → sims muito
  mais rápidas) e reescalar — transformando a preocupação do `dt` em ferramenta.

## 6. Riscos / decisões

- **Colapso pode não ser limpo** → grupos errados; iterar Buckingham-π (talvez incluir `T_FF/τ_a`
  ou `K·dt` como grupo extra).
- **Piso do T_FF** (≥ τ_a): para τ_a grande o overlay é atuação-limitado → `A→1`. Isso é a
  **saturação esperada** do mapa de fases, não falha.
- **Instabilidade amostrada** em `dt` grande limita a alavanca (é o gancho do trilema, Cap. 3) —
  e por si só já é um resultado citável (existe um `dt` máximo).
- **Não reutilizar** `agility_results.csv` (Option A) — re-rodar em B2.

## 7. Ordem de execução proposta (quando autorizar)

1. ✅ (código) `CONTROL_PERIOD` env-overridable — FEITO e validado (2026-06).
2. ✅ (smoke) 1 célula B2 estável a τ_a=1.0/N=24 — FEITO: tau_B2=2.17 s, R²=0.93, settled,
   vantagem 8.97× — bate com `optionB_results_B2scale1.csv`. Config B2 do sweep travada.
3. ⏭️ (sim) **PRÓXIMO:** varredura 2D N∈{8,16,24} × τ_a∈{0.2,0.5,1,2} em B2 + knobs ADAPTADOS
   (T_FF(τ_a), scale(τ_a)) → `collapse_results.csv`. (Estender `run_optionB_test.py`/
   `run_agility_sweep.py` para B2 + ganho estável + adaptação.)
4. (sim) eixo dt em (N=16, τ_a=0.5): dt∈{0.01,0.02,0.05,0.1}.
5. (análise) `analyze_collapse.py` → figura `A` vs `Pe` + fronteira `Pe*` + teste de colapso.
6. validar pontos de N grande existentes na curva; se colapsar, declarar a caracterização.

> **Estado:** itens 1–2 ✅ feitos (config travada). **Próximo passo = item 3** (a varredura 2D),
> mas isso JÁ é sweep — pedir aval do usuário antes de disparar. Regra de adaptação definida ↓.

## 8. Regra de adaptação dos knobs à agilidade τ_a (proposta 2026-06 — aprovar antes de rodar)

Ao varrer τ_a (`VM_TAU_XY`), adaptar os knobs do overlay para isolar a dependência em Pe
(senão um `T_FF` fixo contamina o eixo e a "corcova" artefato reaparece). **Para o B2, quem
adapta é o `T_FF`, NÃO o `scale`:**

1. **`T_FF = c_FF · τ_a`, com `c_FF = 1.0`** (`DUAL_PULSE_T_FF = VM_TAU_XY`). Motivo: o feedforward
   `v_ff=(s_i/T_FF)·r` não pode superar o atuador (piso T_FF ~ τ_a); amarrar a τ_a faz
   `τ_B2/T_FF ≈ const` → colapsa o eixo de agilidade; e mantém o ponto validado (τ_a=1→T_FF=1→
   2.17 s) como caso particular. Se células ágeis (τ_a=0.2) derem overshoot, subir c_FF p/ 1.5–2.
2. **`DELTA_SCALE = 1.0` FIXO (não adaptar).** A lei decrescente `scale*(τ_a)` era do **B-mínimo**
   (double-drive). No **B2** o viés cancelador COMPLETO zera o double-drive → scale=1.0 é correto
   e τ_a-independente. Só checar overshoot na célula mais ágil; introduzir scale<1 apenas se a
   medição exigir.
3. **`K_E_TAU = 250/N` fixo no eixo τ_a** (feedback só limpa resíduo). No eixo `dt`, o produto
   **`K·dt`** é o que governa a estabilidade amostrada → esperar instabilidade em `dt` grande
   (é o RESULTADO/fronteira, não um knob a consertar).
4. **Mantidos:** `DUAL_PULSE_INTEGRATION=B2`, `TTL=3N`, `RAMP_TICKS=4` (em ticks → escala com dt).

**Saturação de velocidade — DECIDIDO 2026-06: opção (A).** Manter `VM_MAX_SPEED_XY=10`/
`VM_MAX_ACC_XY` FIXOS (realista) e **declarar a saturação como piso de atuação legítimo** no
mapa de fases (a ponta snappy pode ficar limitada por velocidade máxima, não por τ_a — isso é
resultado, não bug). [Opção (B) descartada: escalar max-speed ~1/τ_a daria "agilidade pura".]
**Regra aprovada pelo usuário.**

**Grade do sweep 2D (barata):** N∈{8,16,24} × τ_a∈{0.2,0.5,1.0,2.0}, métodos {baseline, B2},
T_FF=τ_a, scale=1.0, K=250/N, dt=0.01. → `collapse_results.csv`. Depois eixo dt em 1 célula.
