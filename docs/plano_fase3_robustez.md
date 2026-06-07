# Plano de execução — Fase 3 / Cap. 7: Robustez (comunicação degradada + churn)

> Plano acionável (PT). Diferente da Fase 2 (só sweeps), a Fase 3 tem DUAS trilhas: a Track A
> (comunicação degradada) é medição quase pura; a Track B (churn) exige **correções de código**
> no protocolo + medição. NÃO rodar/editar nada ainda; este doc fixa knobs, fixes e ordem.

## 1. As duas perguntas

- **Track A — Comunicação degradada:** como o B2 (feedforward é **malha aberta**, "tão bom
  quanto o δ_D") degrada sob **atraso** e **perda de pacotes**? Qual o papel do
  `BROADCAST_REPEATS` (redundância) na entrega dos pulsos e do broadcast de `dp_shift`?
- **Track B — Churn / falhas concorrentes:** sob Poisson denso (falhas + recuperações,
  possivelmente simultâneas), o protocolo dual_pulse **detecta e reconfigura** corretamente?
  Métrica: **cobertura de falhas** (fração de eventos reconfigurados certo).

## 2. Aterramento no código (estado real)

| Knob | Onde | Estado | Ação |
|---|---|---|---|
| `COMMUNICATION_DELAY` | config:64 → `CommunicationMedium(delay=)` (main:206-211) | ligado ao GrADyS ✅; **não** env | tornar env |
| `COMMUNICATION_FAILURE_RATE` | config:65 → `CommunicationMedium(failure_rate=)` | ligado ✅ (perda real); **não** env | tornar env |
| `FAILURE_MEAN_FAILURES_PER_MIN` | config:143 | constante 2.0; **não** env | tornar env (churn denso) |
| `FAILURE_OFF_TIME` | config:144 | constante 8.0 (recovery) | tornar env |
| `FAILURE_CHECK_PERIOD` | config:142 | 0.1 s | manter |
| `EXPERIMENT_SEED` | main:157, protocol_agent:85-89 | ✅ semeia Poisson por-agente | **usar p/ multi-seed (aqui FAZ sentido)** |
| `BROADCAST_REPEATS` | constante em `dual_pulse_layer.py` (=2) | não env | tornar env p/ varrer redundância |
| `FAST_CHANNEL_WARMUP_SEC` | config:379 | 1.0 s | manter |

**Lógica de gatilho (protocol_agent.py:1102-1126) — confirmada:**
- SAIDA = `alive_decreased`; ENTRADA = `alive_increased` (sinal do delta de `alive_count`),
  ambos com gate `succ_changed AND not in_warmup`. Só o originador canônico (predecessor)
  injeta; ENTRADA usa `recovered_id = neighbor_succ_id`.

## 3. Track A — Comunicação degradada (medição; ~sem fix)

1. **[código pequeno]** tornar `COMMUNICATION_DELAY`, `COMMUNICATION_FAILURE_RATE` e
   `BROADCAST_REPEATS` env-overridable (como já fiz p/ `CONTROL_PERIOD`).
2. **[sim] Sweep de perda:** `COMMUNICATION_FAILURE_RATE ∈ {0; 0,1; 0,2; 0,4}` em B2 (e baseline
   p/ referência), N=24, τ_a=1, evento único, **multi-seed** (a perda é estocástica → seeds
   genuinamente diferentes). Medir: τ_B2, `egap_final`, e **cobertura do δ_D** (fração de agentes
   que receberam o shift; via `diag_coverage.py`/eventos). Hipótese: B2 degrada graciosamente até
   um ponto, depois o δ_D fica incompleto → resíduo limpo pelo feedback lento.
3. **[sim] Sweep de atraso:** `COMMUNICATION_DELAY ∈ {0; 1·dt; 5·dt; 10·dt}`. Atraso desloca a
   disseminação no tempo (não a destrói) → esperado degradar menos que perda.
4. **[sim] Papel do `BROADCAST_REPEATS`:** repetir o sweep de perda com repeats ∈ {1,2,3} →
   mostrar que a redundância recupera a cobertura sob perda (trade-off banda × robustez).
   Conectar ao Cap. 4 (custo de mensagens O(N) × repeats).

## 4. Track B — Churn (REORIENTADO: caracterizar, depois consertar só o necessário)

**Reorientação (discussão 2026-06):** o **baseline é sempre a rede de segurança** — ele corrige
espaçamento não-uniforme independentemente de eventos. Logo as "fragilidades" não quebram o
enxame; elas são, no máximo, **"o overlay não acelera esse caso"**. Distinguir:
- **Falha passiva** = overlay fica mudo → baseline cobre → **benigna**.
- **Dano ativo** = overlay aplica `δ_D` **errado** → atrapalha o baseline (como era o bug da perda).
Track B = **caracterizar onde o overlay acelera × cede ao baseline, e consertar só o dano ativo
(ou o que for frequente e custoso).** ISOLAR: churn **sem perda e sem atraso** (combinado fica
para o fim).

**Reclassificação das 3 fragilidades:**
1. **Líquido-zero (dep+arr no mesmo tick):** o `N` não muda; geometria sim, mas o `dual_pulse` é
   de **evento único** e não casa com evento composto → **não disparar e deixar o baseline é
   defensável**. **Falha passiva, provavelmente rara.** → **MEDIR frequência+impacto, não
   consertar a priori.** (protocol_agent:1102-1109.)
2. **Carimbo de N (eventos sobrepostos no tempo):** se a topologia muda durante a circulação, o
   `N_new` inferido pode sair errado → `δ_D` errado. **ÚNICO candidato a DANO ATIVO.** → medir se
   ocorre; se sim, estampar `N`/usar `n_new` consistente.
3. **ENTRADA com originador falho** (CLAUDE.md ~3/24): ninguém injeta → **falha passiva**;
   baseline cobre. → medir frequência; fix (fallback do sucessor) só se valer a pena.
(4. ✅ RESOLVIDO na Track A — `repeats≥2` sob perda era `AGENT_STATE_TIMEOUT` curto, não dedup.)

**Cenários de teste (loss=0, delay=0):**
- **Poisson denso:** 10–30 falhas/min + recovery finito, N=24, **multi-seed ≥5** (estocástico →
  multi-seed é genuíno, ≠ Fase 2). Mede o regime "natural".
- **Determinístico A — k vizinhos adjacentes** (k=2,3) em t₀: previsão = **sub-correção** (só 1
  evento dispara; 2º originador morto) → resíduo maior → baseline limpa.
- **Determinístico B — 2 (e 3) não-vizinhos** em t₀: previsão = `δ_D` **aditivo** (`shift_target`
  acumula por evento, dual_pulse_layer:582) → **~correto com leve sobre-correção ~1/N** (cada
  evento usa `gap_old=2π/(N−1)` em vez de `2π/N`; soma ≈ necessário × N/(N−1)). Caracteriza a
  superposição.

**Métrica (do `events.csv`, como no diag da Track A):** eventos reais (`failure_start/end`) ×
injeções × completados; **`N_new` correto?** (detecta dano ativo); **espúrios** (ENTRADA sem
recuperação); **cobertura global** = parar o churn → `egap_final` assenta em ~0 para o N final?

**Ordem:** (1) env knobs; (2) `diag_churn.py` (Poisson + A + B), **medir antes de mexer**;
(3) decidir quais fixes valem; (4) fix-and-measure + `pytest` 97/97 a cada passo + casos novos em
`tests/test_dual_pulse.py` (fixture `no_hop_alpha`).

## 5. Item herdado: histerese adimensional (adiado da Fase 0)

Tornar `HYSTERESIS_RAD` adimensional (fração do gap local, ~c·2π/N_alive). Só morde sob
**reordenação de vizinhos** (churn/passagem) — portanto é um item da Track B. Testar em N≥75
sob churn que a seleção de vizinho não trava (a histerese fixa excede o gap em N~126).

## 6. Riscos / ordem

- **Track A primeiro** (mais limpa: medição + 1 fix de env pequeno; não mexe no protocolo).
- **Track B depois** (mexe no protocolo → risco de regressão; rodar `pytest` a cada fix; os
  testes de controle/dual_pulse travam a dinâmica — não quebrá-los).
- Multi-seed AQUI é genuíno (perda e Poisson são estocásticos) — diferente do caso simétrico
  determinístico da Fase 2.
- Resultado pode ser negativo-vira-dado: o feedforward malha-aberta pode degradar cedo sob perda
  → caracterizar honestamente *onde* o B2 deixa de ganhar (fecha o mapa de fases do Cap. 6).

## 7. Progresso

1. ✅ **FEITO (2026-06):** `COMMUNICATION_DELAY`, `COMMUNICATION_FAILURE_RATE` (config_param) e
   `DUAL_PULSE_BROADCAST_REPEATS` (config + import no dual_pulse_layer) agora env-overridable;
   defaults preservados. (Perda já estava ligada ao `CommunicationMedium` do GrADyS.)
2. ✅ **SMOKE FEITO:** B2 (N=24, τ_a=1) com **perda 20%**, 3 seeds → τ_B2 saltou de **2,17 s
   (perda 0)** para **34–140 s**, `egap_final` 0,08–0,25 (NÃO assenta), R² 0,02–0,21. Confirma:
   (a) env-wiring funciona; (b) **20% de perda quase destrói a vantagem do B2** (feedforward
   malha-aberta depende dos pulsos chegarem); (c) **seed-sensível** (139/34/37) → multi-seed é
   genuíno aqui; (d) **budget precisa escalar sob perda** (B2 fica lento/instável → 15 s é curto).
3. ✅ **SWEEP DE PERDA FEITO (2026-06):** perda ∈ {0;0,05;0,1;0,2;0,4} × {baseline,B2}, N=24,
   τ_a=1, 3 seeds, budget 150 s. Métrica confiável = **egap_final** (o τ_fit quebra sob perda:
   decaimento vira não-exponencial → NaN/absurdo). RESULTADO (egf mediano / assentou):
   - perda 0:   base 0,0001/✅ | B2 0,0000/✅ (adv 9×)
   - perda 0,05: base 0,0001/✅ | B2 0,0001/✅
   - **perda 0,1: base 0,0001/✅ | B2 0,16/❌  ← CROSSOVER**
   - perda 0,2:  base 0,0012/✅ | B2 0,64/❌
   - perda 0,4:  base 0,008/❌  | B2 0,72/❌
   **ACHADO:** baseline (malha FECHADA, auto-corretivo) robusto à perda até ~0,2; overlay
   (feedforward malha ABERTA) quebra em ~0,1 → ~2–4× mais frágil. Dados `comm_results.csv`,
   análise `analyze_comm.py`, fig `comm_loss_robustness.png`.
4. ✅ **SWEEP BROADCAST_REPEATS FEITO (2026-06) — INVERTE o item 3.** repeats ∈ {1,2,3,5} sob
   perda 0,1/0,2 (+ repeats=1 em perda 0/0,05/0,4), B2, 3 seeds. egap_final mediano:
   - perda 0,1: r1=**0,0001✅** | r2=0,16❌ | r3=0,19❌ | r5=0,21❌
   - perda 0,2: r1=**0,0011✅** | r2=0,64❌ | r3=0,91❌ | r5=0,65❌
   - perda 0/0,05: r1 ✅ (limpo OK!); perda 0,4: r1 ~0,007 (2/3 seeds)
   **ACHADO:** a "fragilidade do overlay à perda" (item 3) era ARTEFATO do `BROADCAST_REPEATS=2`.
   Com **repeats=1 o overlay é robusto à perda até ~0,2 (≈ baseline) E funciona limpo**. repeats≥2
   CORROMPE sob perda (mais redundância = pior) -> **BUG de interação perda×reenvio** no dedup
   (chave (event_id,direction), set permanente). Resultado VIRA POSITIVO: overlay sobrevive a
   comm degradada. Dados comm_results_repeats.csv. Cap.7 §7.2.1 corrigido.
5. ✅ **BUG CONSERTADO (2026-06).** Diagnóstico (`diag_repeats.py`): 1 falha gerava 64 injeções /
   91 eventos (incl. 366 ENTRADA espúrias) sob perda 0,2 — causa = **AGENT_STATE_TIMEOUT=5·dt curto**
   (vizinho vivo "pisca" morto sob perda; falso-positivo de detector de falhas). FIX:
   AGENT_STATE_TIMEOUT env-overridable, regra timeout≫(perdas consecutivas)·dt (ex. 20·dt=0,2s).
   Confirmado: B2 (repeats=2 default) assenta em TODAS as perdas até 0,4 (egf≤0,0004); caso limpo
   não regride; baseline também melhora em 0,4. `pytest` 97/97. NÃO era bug de dedup. Dados
   comm_results_fix.csv. Cap.7 §7.2.1 reescrito (versão final).
6. ✅ **SWEEP DE ATRASO FEITO (2026-06).** delay {0,1,5,10}·dt, loss=0, 1 seed. Baseline ~imune
   (τ 19,5→21,7; egap~0). Overlay degrada a partir de ~5·dt, quebra em 10·dt (egap 0,0156→0,108).
   **Mecanismo DISTINTO da perda** (verificado: timeout 0,3 não muda → 0,108): atraso entrega
   pacotes (rxtime recente, sem expiração) → não é falso-positivo de FD; é ESTADO DEFASADO (FF
   precisa de posições atuais). Track A fecha com 2 eixos: perda (timeout do FD) + atraso (<~5·dt);
   fora disso, gating p/ baseline. Cap.7 §7.2.2/§7.2.3. Dados comm_results_delay/delaytmo.csv.
7. ⏭️ **PRÓXIMO (aguarda ok):** Track B churn (§4 itens 1-3) e/ou commit das edições.
> Track A COMPLETA. Resta Track B (churn) — possivelmente menor após o fix do FD timeout.
