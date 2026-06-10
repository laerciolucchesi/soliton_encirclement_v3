# Overlay robusto v2 — diagnóstico + redesign sob churn denso e alvo em manobra

> ## ⚠️ RESOLUÇÃO (2026-06) — leia antes
> **A causa-raiz do "churn problem" NÃO era o overlay — era o GATILHO, que detectava eventos pelo
> `alive_count` GLOBAL (violando a premissa vizinho-apenas).** Consertado o gatilho (detecção de
> direção por **frescor local do succ**; sem contagem global), o **`dual_pulse` original (add) passou
> a AJUDAR sob churn** (vantagem 1,02–1,42; o desastre de 0,48 sumiu) e a se sustentar vizinho-apenas
> (provado por alcance curto de 25 m). Veredito das propostas deste doc, **após validação**:
> - **M8** (consumir só a rotação de redistribuição no `consume_motion`) → **IMPLEMENTADO + DEFAULT**
>   (conserta a manobra; churn+manobra sobe p/ 1,16–1,20). ✅
> - **gate, M2 (estampa/topologia-N), M5 (idempotente), acumulação condicional** → **EXPLORADOS e
>   DESCARTADOS** (eram remédios para o bug do gatilho; com o gatilho consertado, atrapalham ou são
>   inúteis — gate piora, M2 viola a premissa, M5 perde quedas simultâneas, condicional é a pior).
> - **M1 (topologia percebida graduada)** → **NÃO necessária**.
>
> Detalhes da resolução em **`docs/draft/cap7_robustez.md` §7.2.7**. O conteúdo abaixo é mantido como
> **registro do diagnóstico/exploração** que levou à descoberta da causa-raiz.

> Análise orientada à implementação (PT). Ancorada nos resultados das Tracks B (churn) e C (alvo
> móvel) e no código atual (`dual_pulse_layer.py`, `protocol_agent.py`, `protocol_target.py`,
> `config_param.py`). Referências: só trabalhos reais; onde houver dúvida de ano/edição, marco
> **(verificar)**. Status: proposta — nenhuma modificação aplicada ainda.

## 0. Ponto de partida (o que JÁ sabemos, medido)
- **Overlay = dual_pulse/B2:** *event-triggered* — na falha/recuperação, o predecessor canônico
  injeta 2 pulsos hop-count; receptores inferem `N_new = h_CCW+h_CW+1` e um shift `δ_D` que
  **enviesa os gaps** (Option A) ou alimenta um *feedforward* que consome o shift em `T_FF` (B2).
  `consume_motion(Δθ)` abate o shift pela rotação angular realizada ([dual_pulse_layer.py:263](../dual_pulse_layer.py#L263)).
- **Baseline:** controlador local (PD radial + tangencial 2-canais) que mira **uniforme `2π/M`**
  sobre os vizinhos vivos atuais (liveness binária via `AGENT_STATE_TIMEOUT`). Auto-estabilizante.
- **Resultados-chave desta campanha:**
  - **Perda:** o overlay *parecia* frágil → era **falso-positivo do detector de falhas**
    (`AGENT_STATE_TIMEOUT=5·dt` curto) → vizinho vivo "pisca morto" → tempestade de eventos falsos.
    Corrigido escalando o timeout (robusto ≥40%).
  - **Churn denso:** vantagem evapora + dano ocasional. Diagnóstico (`diag_outlier.py`): **NÃO** é
    viés preso (`e_tau` virtual ≈ físico, shift pequeno) nem `N_new` impossível (clipe não cura);
    é **feedforward incoerente** — o anel muda *durante* o voo do pulso → `N_new` plausível-mas-velho
    → muitas correções pequenas inconsistentes que agitam o enxame. **Regime mismatch**, não bug de
    álgebra. Mitigado por **gate** (degrada p/ baseline).
  - **Alvo em manobra:** o `consume_motion` **come o shift** com a rotação de *tracking* → o overlay
    **sub-redistribui**; benefício se dilui. **Constante** (translação rígida) → efeito **nulo**.
  - **Transversal:** o overlay é **puramente tangencial** → **nunca** degrada o `E_r` (tracking radial).

---

## PARTE 1 — Diagnóstico (hipóteses; mecanismo / assinatura / detecção / indicador)

| # | Hipótese | Mecanismo | Assinatura na métrica | Detecção (log/dado) | Indicador a monitorar |
|---|---|---|---|---|---|
| H1 | **Topologia assumida ≠ real** | `N_new` inferido por hop durante o voo; o anel muda no meio → δ_D com N errado | egap_avg alto sob churn; `N_new` espalhado | `events.csv`: dist de `N_new` (impossíveis/incoerentes) | `N_new` vs alive real; % impossível |
| H2 | **Visões inconsistentes entre agentes** | cada agente tem alive-set diferente (perda/atraso) | reconfiguração não converge | comparar alive_count por agente no tick do evento | variância de `alive_count` entre agentes |
| H3 | **Identificação de vizinho defasada** | `succ/pred` apontam p/ id velho (perda) | injeções espúrias; δ_D no alvo errado | `events.csv`: nº de eventos ≫ falhas reais | nº injeções / nº falhas reais |
| H4 | **Correção com base em msg antiga** | pulso circula > 1 volta (TTL) ou chega atrasado | δ_D obsoleto aplicado | idade do pulso vs instante de aplicação | `now − t_inject` do pulso aplicado |
| H5 | **FF antecipa dinâmica inválida (manobra)** | extrapolação linear do alvo durante aceleração | E_r/egap pioram em manobra; pico no instante da curva | `target_telemetry`: E_vr cresp. à mudança de direção | `|a_alvo|` estimado vs erro |
| H6 | **FF amplifica ruído de estimação** | derivada da posição do alvo sem filtro | jitter em E_r; esforço alto | `u`/`v_ff` com alta variância | std de `v_ff` |
| H7 | **Comando excessivo / satura** | δ_D grande / T_FF pequeno → v_ff > VM_MAX_SPEED | clipping; overshoot; late_std alto | `velocity_norm` no teto `VM_MAX_SPEED_XY` | fração de ticks saturados |
| H8 | **Conflito overlay × baseline** | viés (overlay) e erro real (baseline) em sentidos opostos | egap não assenta; oscila | `e_tau` (virtual) ≪ `e_tau_real` (físico) | gap `|e_tau − e_tau_real|` |
| H9 | **Churn reordena agentes** | manobra/churn troca a ordem angular → vizinho errado | δ_D com `h_CCW` errado | ordem angular (`theta_rel`) vs ordem de id | nº de trocas de `succ_id`/min |
| H10 | **Entrada/saída muda gaps esperados** | M muda → `2π/M` muda; overlay usa M velho | resíduo após cada evento | `N_new` aplicado vs M corrente | Δ(M) por janela |
| H11 | **Overlay lento p/ estabilizar pós-topologia** | shift acumulado não decai antes do próximo evento | egap_avg cresce com a taxa | shift_remaining persistente entre eventos | `mean|dual_pulse_shift|` em janelas "quietas" |
| H12 | **Não distingue erro real de transiente de churn** | dispara em flutuação temporária | injeções espúrias (como H3) | ENTRADA sem recuperação real | nº ENTRADA / nº recuperações |
| H13 | **Manobra invalida "movimento suave"** | hipótese de v≈const do FF quebra | E_r sobe nas curvas | correlação `|a_alvo|` × E_r | `|a_alvo|` |
| H14 | **FF aplicado com confiança que deveria ser baixa** | sem gating por confiança → atua sempre | dano nos regimes ruins | egap_B2 > egap_base nos cenários ruins | confiança do FF (a definir) |

**Resumo do diagnóstico (alinhado às medições):** sob **churn** dominam **H1, H4, H10, H11, H12**
(feedforward incoerente por topologia/N defasados, correções que não decaem); sob **manobra**
dominam **H5, H13, e o `consume_motion`** (não listado acima como hipótese genérica, mas é a causa
medida: a rotação de tracking é descontada do shift). H7/H8 aparecem secundariamente. O **tracking
nunca é prejudicado** (o overlay é tangencial), então H "overlay piora E_r" é **descartada** por dado.

---

## PARTE 2 — Churn denso: topologia real percebida

A ideia do usuário (estimar dinamicamente a topologia a partir das mensagens) **faz sentido** e é o
caminho certo — alinha com **detectores de falhas** (Chandra & Toueg, *verificar*) e **consenso sob
topologia comutante** (Olfati-Saber & Murray 2004, real). O `AGENT_STATE_TIMEOUT` atual já é um
detector de falhas **binário** (vivo/morto por timeout); a proposta o **gradua** em confiança.

### 2.1 Dados por mensagem (`AgentState` — estender)
Hoje `AgentState` carrega `agent_id`, `seq`, `position`, `velocity`, `u`, `prop_state`. Adicionar:
- **`seq`** (já existe) — para idade/ordem.
- **`alive_view`** *(opcional)*: bitmap/lista resumida dos ids que o emissor considera ativos
  (permite cross-check e consenso de presença).
- *(o timestamp de recepção é local — não precisa ir na mensagem.)*

### 2.2 Idade da informação
Por vizinho `j`, local: `age_j = now − rx_time_j` (rx_time já existe via `_prune_expired_states`).

### 2.3 Confiança por agente (graduada, decai com idade)
```
conf_j = clip( exp(-age_j / TAU_CONF), 0, 1 )           # decai suave (vs corte abrupto)
```
Boost por consistência: se as últimas K mensagens de `j` chegaram regularmente (intervalo ~período
de broadcast), `conf_j` sobe mais rápido; rajada de perdas só o derruba após várias ausências.
`TAU_CONF` dimensionado à taxa de perda (como o fix do `AGENT_STATE_TIMEOUT`: `TAU_CONF ≫
(perdas consecutivas)·dt`).

### 2.4 Classificação ativo / incerto / inativo (com histerese)
Dois limiares (evita flapping):
```
se conf_j >= C_HI:        estado = ATIVO
senão se conf_j <= C_LO:  estado = INATIVO
senão:                    estado = INCERTO (mantém estado anterior — histerese)
```

### 2.5 Vizinhos relevantes
`pred/succ` = vizinhos angulares (por `theta_rel`) **dentro do conjunto ATIVO** (não por id, não
por INCERTO/INATIVO). Recalcular só quando o conjunto ATIVO muda além da histerese.

### 2.6 Formação desejada quando M muda
`M_eff = |ATIVO|`; `gap_desejado = 2π / M_eff` (ponderado por λ se `PROTECTION_ANGLE`). Suavizar:
`M_suave ← (1−β)·M_suave + β·M_eff` e usar `2π/M_suave` (evita salto de gap a cada entrada/saída).

### 2.7 Evitar oscilação da topologia
Histerese (2.4) + suavização de M (2.6) + **cooldown** (após uma mudança de M, segura novas
mudanças por ~`T_COOLDOWN`). (Mesma filosofia do `HYSTERESIS_RAD` já existente para vizinhos.)

### 2.8 Integração ao overlay
- **Confiança da topologia** `Θ ∈ [0,1]` = fração média de confiança dos vizinhos relevantes (ou
  `1 − entropia` da distribuição ativo/incerto).
- **Ganho/blend do overlay** `w_overlay = g(Θ)` (ver Parte 4, M4): Θ alto → overlay pleno; Θ baixo →
  cai p/ baseline. **Generaliza o gate atual** (que é `w∈{0,1}` por taxa de eventos) para `w∈[0,1]`
  por confiança.
- **δ_D usa `M_eff`/`M_suave`** (Parte 4, M2/M5), não o `N_new` inferido por hop sob voo.

---

## PARTE 3 — Alvo em manobra: por que o FF piora e como consertar

### 3.1 Causas (medidas + prováveis)
- **`consume_motion` desconta a rotação de tracking** (CAUSA MEDIDA): o shift do overlay é abatido
  pela rotação de `theta_rel` vinda do acompanhamento, não só da redistribuição → sub-redistribuição.
  Só morde sob **manobra** (translação pura não muda `theta_rel`).
- **v_alvo defasada / aceleração não estimada / extrapolação linear:** o tracking radial usa a
  posição/velocidade do alvo; em aceleração, a estimativa fica atrás → E_r sobe nas curvas.
- **Confiança excessiva do FF:** o FF atua igual em cruzeiro e em manobra (sem gating).
- **Saturação / conflito** radial×tangencial: secundário (E_r não é prejudicado pelo overlay nos
  nossos dados, mas em manobra forte + redistribuição simultânea pode haver disputa de banda).

### 3.2 Correções propostas (detalhadas na Parte 4)
- **Corrigir o `consume_motion`** para descontar **apenas a rotação de redistribuição** (M8 — a mais
  importante para manobra).
- **Estimador de v/a do alvo** filtrado (α-β ou KF pequeno) (M6).
- **FF condicionado por confiança + horizonte adaptativo** (M7): reduzir ganho/horizonte quando
  `|a_alvo|` alta.
- **Detector de manobra → gating do FF** (M9).
- **Safety filter / command limiter** (M10).

---

## PARTE 4 — Modificações concretas (10)

> Formato: nome · problema · ideia · implementação · variáveis novas · pseudocódigo · efeito ·
> risco · custo · prioridade. Custo: B(aixo)/M(édio)/A(lto). Prioridade: 1 (primeiro) … 5.

**M1 — Detector de falhas graduado (confiança por vizinho).**
- Problema: H1/H3/H12; falso-positivo sob perda; liveness binária.
- Ideia: substituir o corte `AGENT_STATE_TIMEOUT` por `conf_j = exp(-age/TAU_CONF)` + histerese.
- Implementação: em `protocol_agent`, manter `conf[j]` no tick; classificar ATIVO/INCERTO/INATIVO.
- Variáveis: `conf[j]`, `age[j]`, estado[j].
- Pseudocódigo: `conf[j]=exp(-(now-rx[j])/TAU_CONF); estado[j]=hysteresis(conf[j])`.
- Efeito: menos eventos espúrios; base p/ M4.
- Risco: detecção de falha REAL mais lenta (trade-off latência×falso-positivo).
- Custo: M. Prioridade: **1**.

**M2 — Recálculo dos gaps a partir do conjunto ATIVO-confiável.**
- Problema: H10; M errado → `2π/M` errado.
- Ideia: `M_eff=|ATIVO|`; `gap_desejado=2π/M_eff`. δ_D usa `M_eff`, não `N_new` por hop.
- Implementação: passar `M_eff` ao layer; nas fórmulas de δ_D, `n_new←M_eff` (em vez de hop-sum).
- Variáveis: `M_eff`.
- Pseudocódigo: `n_new = M_eff; gap_new = 2π/M_eff; ... (resto igual)`.
- Efeito: elimina os `N_new` incoerentes (causa do churn). 
- Risco: `M_eff` local pode divergir entre agentes (mitigado por M1+`alive_view`).
- Custo: M. Prioridade: **1**.

**M3 — Histerese + suavização da topologia.**
- Problema: H9/oscilação; flapping de M e de vizinhos.
- Ideia: 2 limiares (C_HI/C_LO), `M_suave` (EMA), cooldown pós-mudança.
- Implementação: estado[j] só muda fora da banda; `M_suave←(1−β)M_suave+βM_eff`.
- Variáveis: `M_suave`, `t_last_topo_change`.
- Pseudocódigo: `if now-t_last_topo_change>T_COOLDOWN: aplicar mudanças de ATIVO`.
- Efeito: topologia estável; menos saltos de gap.
- Risco: resposta a mudança real um pouco mais lenta.
- Custo: B. Prioridade: **2**.

**M4 — Blending adaptativo baseline↔overlay por confiança (Θ).**
- Problema: H14; o gate atual é binário.
- Ideia: `w_overlay = g(Θ)` contínuo; `u = (1−w)·u_baseline + w·u_overlay` (ou `shift *= w`).
- Implementação: `Θ=mean(conf vizinhos relevantes)`; `w=clip((Θ−Θ_lo)/(Θ_hi−Θ_lo),0,1)`; aplicar
  `w` ao shift/feedforward. Substitui/Generaliza `set_churn_suppress` (que é `w∈{0,1}`).
- Variáveis: `Θ`, `w_overlay`.
- Pseudocódigo: `shift_eff = w_overlay * shift_remaining`.
- Efeito: degrada **graciosa e continuamente** p/ baseline (em vez de liga/desliga).
- Risco: tuning de Θ_lo/Θ_hi.
- Custo: B. Prioridade: **2**.

**M5 — N-stamp + alvo idempotente (correção absoluta, não incremental).**
- Problema: H1/H4/H11; `shift_target += δ_D` acumula correções inconsistentes sob churn.
- Ideia: (a) **estampar** `N` observado pelo originador na injeção (número consistente, não inferido
  no voo); (b) tornar a correção **idempotente** — o evento mais novo **define** o alvo de
  redistribuição (sobrescreve), em vez de **somar** (cada evento antigo deixa de "empilhar").
- Implementação: pulso carrega `N_stamp=alive_count_originador`; receptor usa `N_stamp`; mudar
  `shift_target = δ_D_corrente` (ancorado na topologia atual) em vez de `+=`.
- Variáveis: `N_stamp` no pulso.
- Pseudocódigo: `shift_target = compute_delta(M_eff, h_CCW)  # absoluto, não acumula`.
- Efeito: mata os 14% impossíveis **e** a incoerência por empilhamento.
- Risco: mudar a semântica de acumulação pode afetar o caso de eventos genuinamente múltiplos
  (validar com os testes determinísticos adj/non da Track B).
- Custo: M. Prioridade: **2**.

**M6 — Estimador de velocidade/aceleração do alvo (filtrado).**
- Problema: H5/H6/H13; FF usa v defasada, a não estimada.
- Ideia: filtro **α-β** (ou KF de 2ª ordem) sobre a posição do alvo broadcast → `v_hat`, `a_hat`.
- Implementação: no agente (ou no target, broadcastado): `v_hat,a_hat = alpha_beta(pos_alvo, dt)`.
- Variáveis: `v_hat`, `a_hat`.
- Pseudocódigo: `r=pos-pos_pred; pos_pred+=v_hat*dt+0.5*a_hat*dt²; v_hat+=β*r/dt; a_hat+=γ*r/dt²`.
- Efeito: tracking melhor em manobra; insumo p/ M7/M9.
- Risco: ruído amplificado se α/β mal sintonizados (filtrar).
- Custo: M. Prioridade: **3**.

**M7 — Feedforward condicionado por confiança + horizonte adaptativo.**
- Problema: H5/H14; FF atua igual em cruzeiro e manobra.
- Ideia: `conf_ff = exp(-|a_hat|/A_REF)`; ganho/horizonte do FF `∝ conf_ff` (manobra forte →
  FF curto/atenuado, deixa o feedback do baseline assumir).
- Implementação: escalar `v_ff` por `conf_ff`; horizonte de predição `H = H0·conf_ff`.
- Variáveis: `conf_ff`, `H`.
- Pseudocódigo: `v_ff *= conf_ff`.
- Efeito: FF para de amplificar erro em manobra.
- Risco: perde antecipação em manobra (mas evita dano — alinhado ao "não piorar").
- Custo: B. Prioridade: **3**.

**M8 — Correção do `consume_motion` (descontar só a rotação de redistribuição).**
- Problema: CAUSA MEDIDA da degradação em manobra.
- Ideia: o `consume_motion` deve abater **apenas** a rotação devida ao próprio FF de
  redistribuição, **não** a rotação comum de tracking/manobra.
- Implementação (duas opções):
  - (a) **comandado:** consumir a rotação **comandada pelo overlay** `Δθ_ff = (v_ff/r)·dt`
    (clipada pelo Δθ medido), em vez do `Δθ` total medido.
  - (b) **common-mode dos vizinhos:** estimar `ω_comum` = mediana da taxa de `theta_rel` dos
    vizinhos ATIVOS; consumir `Δθ − ω_comum·dt`.
- Variáveis: `Δθ_ff` (a) ou `ω_comum` (b).
- Pseudocódigo: `consume_motion(min(Δθ_medido, Δθ_ff))`  # opção (a).
- Efeito: **recupera a vantagem do overlay sob manobra** (o shift deixa de ser comido pelo tracking).
- Risco: se a actuação satura, o comandado superestima — por isso o `min(.)` com o medido.
- Custo: B. Prioridade: **1** (alto valor, baixo custo, ataca a causa medida).

**M9 — Detector de manobra → gating do FF.**
- Problema: H5/H13.
- Ideia: `manobra = |a_hat| > A_THR  ou  Δdireção > D_THR`; sob manobra, suprime/decai o FF
  (análogo ao gate de churn).
- Implementação: reusar a estrutura `set_churn_suppress` → `set_suppress(manobra OR churn)`.
- Variáveis: flag `manobra`.
- Pseudocódigo: `if manobra: layer.set_suppress(True)`.
- Efeito: garante "não piorar" em manobra forte.
- Risco: pode desligar o FF cedo demais (tunar A_THR); M7 (atenuação contínua) é preferível a M9
  (liga/desliga) — implementar M7 primeiro, M9 como backstop.
- Custo: B. Prioridade: **3**.

**M10 — Safety filter / command limiter no overlay.**
- Problema: H7; comando do overlay excessivo/saturando.
- Ideia: limitar a **contribuição do overlay** a uma fração de `VM_MAX_SPEED_XY` e *rate-limit*
  (limite de aceleração), preservando o comando do baseline.
- Implementação: `v_overlay = clip(v_overlay, ±κ·VM_MAX_SPEED); |Δv_overlay/dt| ≤ a_max`.
- Variáveis: `κ`, `a_max`.
- Pseudocódigo: `v_cmd = v_baseline + clip(v_overlay, limites)`.
- Efeito: impede que o overlay sozinho sature/oscile; rede de segurança final.
- Risco: limitar demais reduz a aceleração da redistribuição.
- Custo: B. Prioridade: **2**.

---

## PARTE 5 — Arquitetura recomendada

```
                    mensagens (AgentState/TargetState)
                                 │
              ┌──────────────────┼───────────────────────┐
              ▼                  ▼                        ▼
   [3] Estimador de        [1] Detector de falhas    [4] Estimador de
   topologia percebida      graduado (conf_j)          movimento do alvo
   (ATIVO/INCERTO/INATIVO,  + histerese (M1,M3)        (v_hat,a_hat — M6)
    M_eff/M_suave — M2)            │                        │
        │   │                      ▼                        ▼
        │   │              [5] Confiança topologia   [6] Confiança FF
        │   │                  Θ ∈ [0,1]                conf_ff=exp(-|a|/Aref)
        │   │                      │                        │
        │   │              [7] Detector churn        [8] Detector manobra
        │   │                  (taxa eventos)            (|a_hat|,Δdir)
        ▼   ▼                      │                        │
   [B] Baseline controller   ┌─────┴──────────┬─────────────┘
   (PD radial + tangencial)  ▼                ▼
        │              [O] Overlay 2-DOF   [9] Blending adaptativo
        │              (δ_D idempotente,   w_overlay=g(Θ,conf_ff,churn,manobra)
        │               N-stamp — M5;       (M4 — generaliza o gate)
        │               consume_motion        │
        │               corrigido — M8)       │
        └───────────────┬───────────────┬─────┘
                        ▼               ▼
                 u = (1−w)·u_base + w·u_overlay
                        │
                 [10] Safety filter / limiter (M10)
                        ▼
                 comando final ao UAV  ──► [11] Política de reset
                                          (on_reset já existe; estender:
                                           reset parcial quando ΔM grande
                                           ou manobra brusca)
```
**Interações:** o **baseline está sempre ativo** (rede de segurança). A topologia percebida (3) e o
estimador de alvo (4) alimentam as confianças (5,6) e os detectores (7,8). O **blending (9, M4)** —
não um liga/desliga — pondera o overlay por `w=g(Θ, conf_ff, churn, manobra)`. O overlay usa a
topologia (M_eff) e o N-stamp (M5) e o `consume_motion` corrigido (M8). O **safety filter (10)** é a
última barreira. O **reset (11)** zera o shift em mudanças grandes.

---

## PARTE 6 — Pseudocódigo (alto nível, por tick de cada agente)

```python
on_tick(now):
    # --- 1. recebimento + topologia percebida ---
    for msg in inbox:                      # AgentState/TargetState recebidos
        rx_time[msg.id] = now; states[msg.id] = msg
    for j in known_ids:
        age = now - rx_time.get(j, -inf)
        conf[j] = exp(-age / TAU_CONF)                         # M1
        estado[j] = hysteresis(conf[j], estado[j], C_LO, C_HI) # M1,M3
    ATIVO = {j: estado[j]==ATIVO}
    if topo_changed(ATIVO) and now - t_last_topo > T_COOLDOWN:  # M3
        M_eff = len(ATIVO); M_suave = (1-β)*M_suave + β*M_eff
        pred,succ = angular_neighbors(ATIVO, theta_rel)        # vizinhos no ATIVO
        t_last_topo = now
    Θ = mean(conf[j] for j in relevantes)                      # M4

    # --- 2. estimador do alvo + manobra ---
    v_hat, a_hat = alpha_beta(target.pos, dt)                  # M6
    conf_ff = exp(-norm(a_hat)/A_REF)                          # M7
    churn   = recent_event_rate() > R_THR                      # detector churn
    manobra = norm(a_hat) > A_THR                              # M9

    # --- 3. evento / overlay (event-triggered) ---
    if topo_event_detected(ATIVO) and confiavel(Θ):
        delta = compute_delta(M_eff, h_CCW, N_stamp)           # M2,M5 (absoluto)
        shift_target = delta                                   # idempotente (M5), NÃO +=
    shift_remaining = ramp(shift_remaining, shift_target)
    consume_motion(min(Δθ_medido, (v_ff/r)*dt))                # M8 (só redistribuição)

    # --- 4. comandos + blending + safety ---
    u_base    = baseline_controller(e_tau_real, e_r, v_hat)    # sempre
    u_overlay = overlay_cmd(shift_remaining, T_FF, r) * conf_ff# M7
    w = g(Θ, conf_ff, churn, manobra)                          # M4 (0..1)
    u = (1-w)*u_base + w*u_overlay
    u = safety_filter(u_base, u, kappa*VM_MAX_SPEED, a_max)    # M10
    send(u)
```

---

## PARTE 7 — Plano de experimentos

**Comparar:** (1) baseline, (2) overlay atual (B2), (3) overlay v2 (modificado). Usar
`run_trackC.py`/`run_churn_sweep.py`/`run_comm_sweep.py` estendidos; **multi-seed ≥5** (estocástico);
sempre **Δ vs baseline** e Δ vs overlay atual.

**Cenários (matriz):** churn ∈ {nenhum, leve 6/min, médio 12/min, denso 24–48/min} × alvo ∈
{parado, constante, circular, oito, zigue-zague, mudança brusca, aceleração} × comm ∈ {limpo, atraso,
perda}. Priorizar primeiro: {denso × {constante, manobra}} e {médio × manobra} (onde o overlay atual
piora) — é onde a v2 precisa provar "≥ baseline".

**Métricas:** espaçamento (`E_gap`→`egap_avg`/`t_settle` via `metrics_util`), **tracking** (`E_r`,
`E_vr`), tempo de recuperação pós-churn, estabilidade pós-reentrada, **esforço** (∫|u| ou std de
`v`), **saturação** (% ticks no teto), overshoot/oscilações (late_std), e **Δ% vs baseline**.
Critério de sucesso da v2: **egap_avg(v2) ≤ egap_avg(baseline)** em TODOS os cenários (não piorar) e
`< baseline` nos cenários de evento discreto/cruzeiro (ainda ajudar). Confirmar `E_r(v2) ≈ baseline`.

**Trajetórias:** hoje o alvo faz random-roaming (manobra a cada 1 s). Para "circular/oito/zigue-zague"
seria preciso estender `protocol_target` (modos de trajetória paramétricos) — **item de código**
(pequeno) antes desses cenários.

---

## PARTE 8 — Respostas objetivas

1. **Causas mais prováveis do overlay piorar sob churn denso:** **feedforward incoerente** por
   topologia/N defasados — o anel muda durante o voo do pulso, gerando muitas correções pequenas
   inconsistentes que **agitam** o enxame (H1/H4/H10/H11/H12). *Não* é viés preso nem N impossível
   (medido: clipe não cura). É **incompatibilidade de regime** (mecanismo de evento-único aplicado a
   fluxo contínuo).
2. **Causas mais prováveis sob alvo em manobra:** o **`consume_motion` desconta a rotação de
   tracking** (causa medida) → sub-redistribuição; somado a v/a do alvo defasadas e FF sem gating de
   confiança (H5/H13). **Sob movimento constante o efeito é nulo** (translação rígida).
3. **Estimar a topologia real percebida faz sentido?** **Sim.** Gradua o detector de falhas binário
   atual em confiança — ataca diretamente o falso-positivo (perda) e a incoerência (churn). É a
   base da robustez v2.
4. **Como estimar:** confiança por vizinho `conf_j=exp(-age/TAU_CONF)` (+ boost por consistência),
   classificação ATIVO/INCERTO/INATIVO com **histerese** (C_HI/C_LO), `M_eff=|ATIVO|` suavizado
   (EMA) + cooldown. `TAU_CONF` dimensionado à perda. (Partes 2.1–2.7.)
5. **Como o overlay usa a estimativa:** δ_D calculado com `M_eff` (não hop-sum no voo) e **N-stamp**
   (M2/M5); correção **idempotente** (sobrescreve, não acumula); peso `w_overlay=g(Θ)` (blending
   contínuo, M4) — generaliza o gate atual.
6. **Como modificar o feedforward p/ não piorar em manobra:** (i) **corrigir o `consume_motion`**
   (consumir só a rotação de redistribuição — M8, prioridade 1); (ii) estimador filtrado de v/a do
   alvo (M6); (iii) **FF condicionado por confiança** `conf_ff=exp(-|a|/A_REF)` + horizonte
   adaptativo (M7); (iv) safety limiter (M10).
7. **Melhor arquitetura:** baseline sempre ativo + overlay ponderado por **blending adaptativo**
   `w=g(Θ, conf_ff, churn, manobra)`, alimentado por topologia percebida (M1–M3) e estimador de alvo
   (M6), com δ_D idempotente/N-stamp (M5), `consume_motion` corrigido (M8) e safety filter (M10).
   (Parte 5.)
8. **Implementar primeiro (ordem):** **M8** (consume_motion — baixo custo, ataca a causa medida da
   manobra) → **M1+M2** (topologia graduada + gaps por M_eff — ataca a causa do churn) → **M4**
   (blending contínuo, substitui o gate) → **M5** (N-stamp/idempotente) → depois M3/M6/M7/M10 →
   M9 como backstop. Validar cada um com **Δ vs baseline** nos cenários {denso, manobra} e `pytest`.

---

## Relação com a literatura (só conceitos reais; anos a **verificar** na biblioteca do projeto)
- **Detectores de falhas não-confiáveis** (Chandra & Toueg) — fundamenta a confiança graduada/
  histerese (M1). *(verificar)*
- **Consenso sob topologia comutante** (Olfati-Saber & Murray 2004 — real) — fundamenta tratar a
  topologia como variável e o blending.
- **Consenso resiliente / W-MSR** (LeBlanc, Zhang, Sundaram, Koutsoukos 2013 — real, *verificar
  detalhes*) — relevante se houver agentes maliciosos/outliers na `alive_view`.
- **Dynamic average consensus** (survey Kia et al. — *verificar*) — estimar M_eff distribuído.
- **Disturbance observer / 2-DOF feedforward** (controle clássico; Horowitz; Ohishi DOB —
  *verificar*) — fundamenta separar feedforward de feedback e o `conf_ff`.
- **Event-triggered control** (Tabuada 2007 — real, *verificar*) — o overlay já é event-triggered; a
  v2 adiciona gating por confiança.
- **Fault-tolerant / resilient formation control** — família ampla; *verificar* refs específicas
  antes de citar.

> Próximo passo sugerido: implementar **M8** (isolado, barato) e re-rodar a Track C cenário 2
> (falha+manobra) — se o `egap_avg(B2)` em manobra cair para ≈ o de constante, confirma a causa e o
> fix antes de investir no estimador de topologia (M1/M2).

---

## STATUS de implementação

- **M8 — IMPLEMENTADO + VALIDADO (2026-06).** Flag `DUAL_PULSE_CONSUME_FF_ONLY` (default off; só
  B/B2). O `consume_motion` passa a abater `(v_ff_clipado/r)·dt` (rotação comandada pelo FF) em vez
  do Δθ total medido. **Resultado (Track C, falha):** manobra → B2 egap_avg 0,0546 → **0,0485**
  (≤ baseline 0,0499 nos 3 seeds; antes era pior em 2/3); constante → sem regressão (0,0013→0,0014);
  `E_r` inalterado. Confirma o mecanismo (consume_motion comendo a rotação de tracking) e atinge o
  "não piorar" sob manobra. `pytest` 97/97. Efeito absoluto modesto (manobra dominada pelo erro de
  perseguição), mas o sinal inverteu. **Decisão pendente:** manter gated vs tornar default p/ B2.
- **M1–M7, M9, M10 — pendentes.** Próximo de maior valor: **M1+M2** (topologia percebida graduada +
  δ_D por `M_eff`) — ataca o churn (onde o overlay mais falha). Depois M4 (blending contínuo) e M5
  (N-stamp/idempotente).
