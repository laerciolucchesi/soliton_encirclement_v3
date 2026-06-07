# Plano de execução — Fase 3 / Cap. 7, Track C: alvo MÓVEL

> Plano acionável (PT). A Track C é o **cenário-mãe**: encircle de um alvo que se MOVE, com a
> formação acompanhando. Re-testa os estresses das Tracks A/B sob movimento. Fazer **depois** de
> fechar+documentar a Track B. Disciplina herdada: isolar; medir antes de concluir; multi-seed
> genuíno (movimento é estocástico).

## 1. Pergunta central
**O movimento do alvo MUDA as conclusões das Tracks A/B?** (não "qual é o egap", mas o Δ vs o
caso estacionário). O overlay é um acelerador de espaçamento *tangencial*; o alvo móvel adiciona
um modo comum de movimento (tracking) — será que ele interfere no overlay?

## 2. Código verificado (suficiente — só expor o knob)
- Movimento em [protocol_target.py:463-505](../protocol_target.py#L463): a cada `TARGET_MOTION_PERIOD`
  (1 s) escolhe direção aleatória e anda a `TARGET_MOTION_SPEED_XY` (env, default 0); fronteira
  `±TARGET_MOTION_BOUNDARY_XY=20 m` aponta de volta ao centro. **Velocity-driven**; os agentes
  encerram em torno de `target_state.position` → a formação **acompanha**. Spin **off** por default.
- **Características:** (a) é **manobra** (direção nova a cada 1 s), não velocidade constante — para
  ~constante usar `TARGET_MOTION_PERIOD` ≫ duração; (b) trajetória usa `random` semeado por
  `EXPERIMENT_SEED` → **multi-seed genuíno**.
- **Ação de código:** expor `TARGET_MOTION_SPEED_XY` (e `TARGET_MOTION_PERIOD`) nos scripts de
  experimento (hoje fixam `"0.0"`). Nenhum código de movimento novo é necessário.

## 3. Métricas — SEPARAR espaçamento × tracking
O overlay só mexe no espaçamento tangencial. Reportar os dois:
- **Espaçamento** (trabalho do overlay): `E_gap` → `t_settle`/`egap_settle` (evento único) ou
  `egap_avg` (churn contínuo) via `metrics_util`.
- **Tracking radial** (controlador radial + mobilidade): `E_r`, `E_vr` (já no `target_telemetry`).
- **Checar:** o overlay **não piora** o tracking (`E_r`), só ajuda/atrapalha o espaçamento.

## 4. Hipóteses NOVAS a caçar (o que a Track C revela de diferente)
1. **`consume_motion` come o shift sob manobra (PRINCIPAL).** O `consume_motion(delta_theta)` abate
   o `shift_remaining` pela rotação angular realizada. Há DUAS fontes de rotação de `theta_rel`
   não-redistributiva: (a) **spin** comandado — rotação grande e deliberada (o que o `CLAUDE.md`
   alerta); **OFF por default → essa fonte some**; (b) **atraso de tracking** — acompanhamento
   imperfeito do alvo.
   **Previsões explícitas (sem spin):**
   - **Velocidade CONSTANTE → efeito NULO esperado.** Em regime a formação translada **rigidamente**
     com o alvo: `theta_rel` fica **constante** (há um *offset* de atraso, mas constante → sem
     *taxa* → o `consume_motion` reage à variação, não ao offset). **Sanity check: se o constante
     degradar, é bug em outro lugar, NÃO o `consume_motion`.**
   - **MANOBRA → efeito modesto, transiente.** Cada mudança de direção gera re-aceleração em que a
     formação atrasa/reorienta → `theta_rel` varia transitoriamente → algum `consume_motion`
     espúrio (só quando coincide com uma redistribuição em curso). Magnitude ~ velocidade·τ_a/R.
   Logo o teste **constante vs manobra** isola o efeito: nulo no constante, modesto na manobra (ou
   nulo nos dois = overlay robusto ao movimento, plausível sem spin).
2. **Reordenação espúria sob movimento.** Manobra pode reordenar vizinhos transitoriamente →
   `succ_changed` → **injeções espúrias** (como os falsos eventos do churn). O **cenário 1**
   (movimento, sem falha) testa isto: *o movimento sozinho dispara o overlay à toa?*

## 5. Cenários (baseline vs overlay, sob movimento)
Sempre com os **fixes das Tracks A/B ligados** (FD-timeout dimensionado p/ perda; gate p/ churn).
| # | Cenário | Testa | Prioridade |
|---|---|---|---|
| 1 | só movimento (sem restrição) | reordenação espúria; overlay dormente atrapalha tracking? | **alta** |
| 2 | + 1 falha em t0 | função central + `consume_motion` (vs k1 estacionário, t_settle~6,6 s) | **alta** |
| 6 | + churn denso | pior caso sob movimento (gate ligado) | **alta** |
| 3 | + perda de comunicação | Track A (loss) sob movimento (FD-timeout ligado) | média |
| 4 | + atraso de comunicação | Track A (delay) sob movimento | média |
| 5 | + churn esparso | Track B (esparso) sob movimento | média |

## 6. Variáveis de varredura
- **Velocidade:** ≥2 (lenta ≈ quase-parado; rápida ≈ perto do limite de atuação `VM_MAX_SPEED_XY`).
  Sweep simples para achar onde o movimento começa a importar.
- **Tipo de movimento:** começar **constante** (`PERIOD` grande — common-mode, fácil de fatorar),
  depois **manobra** (default, onde o `consume_motion` morde).
- **Multi-seed** (≥3): caminho de movimento é estocástico → genuíno.

## 7. Disciplina
- **Δ vs estacionário:** cada cenário lido contra sua versão parada das Tracks A/B (a conclusão
  mudou?).
- **Isolar:** o movimento é o CONTEXTO da Track C; variar UM estresse por vez dentro dele.
- **Negativo-vira-dado:** se a manobra degradar o overlay via `consume_motion`, caracterizar
  honestamente (e considerar: descontar a rotação de tracking do `consume_motion` como fix futuro).

## 8. Próximo passo concreto (depois da Track B)
1. (código) expor `TARGET_MOTION_SPEED_XY`/`TARGET_MOTION_PERIOD` nos scripts (`run_comm_sweep`,
   `run_churn_sweep`, `diag_*`) ou um `run_trackC.py` dedicado.
2. (sim) cenário 1 (movimento puro, constante + manobra, 1 velocidade) baseline vs B2 → checar
   espúrios e tracking. **Medir antes de concluir.**
3. cenário 2 (falha sob movimento) → o teste-chave do `consume_motion` (t_settle vs k1).
4. cenário 6 (churn denso sob movimento, gate ligado) → pior caso.
> Fixes possíveis se aparecer dano: descontar rotação de tracking no `consume_motion`; estender o
> gate para "manobra detectada". Caracterizar primeiro.
