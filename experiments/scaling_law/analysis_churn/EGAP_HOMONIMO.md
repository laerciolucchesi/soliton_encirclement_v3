# `egap_avg` é um homônimo: duas definições, duas janelas, o mesmo nome de coluna

Mapa factual. **Nada foi alterado** — este arquivo só registra qual definição está em uso
onde. Varredura de `experiments/scaling_law/*.py` e dos cabeçalhos reais de todos os
`*.csv` do diretório (não só do código).

---

## 1. As duas definições

### DEF-A — "churn": helper local do runner

`experiments/scaling_law/run_churn_sweep.py:47-56`

```python
T0 = 5.0            # :41
WARMUP_AVG = 15.0   # :42
steady = df[df["timestamp"] >= T0 + WARMUP_AVG]["E_gap"].to_numpy(float)   # :51
return {"egap_avg": float(np.mean(steady)),
        "egap_p90": float(np.percentile(steady, 90)),
        "egap_max": float(np.max(steady))}
```

* **Janela:** `t >= 20 s`, até o fim da rodada (`SIM_DURATION = T0 + BUDGET = 155 s`) →
  **135 s** de sinal.
* **Estímulo:** fluxo de Poisson contínuo com recuperação (churn). Não há evento único.
* **Produz também:** `egap_p90`, `egap_max` — que **só existem nesta definição**.
* **Não passa por `metrics_util.py`.**

### DEF-B — "evento": `metrics_util.event_metrics`

`experiments/scaling_law/metrics_util.py:104-127`

```python
def event_metrics(df, t0):
    sub = df[df["timestamp"] >= t0].reset_index(drop=True)          # :107
    ...
    steady = sub[sub["timestamp"] >= t0 + 10.0]["E_gap"].to_numpy(float)   # :119
    egap_avg = float(np.mean(steady))                                       # :120
```

* **Janela:** `t >= t0 + 10 s`. Todos os runners passam `T0 = 5.0`, logo **`t >= 15 s`**.
  A exceção é `analyze_breach_window.py:166`, que passa `t_fail` → `t >= t_fail + 10`.
* **Estímulo:** falha determinística única em `t0`; `egap_avg` é o **resíduo pós-evento**.
* **Produz também:** `t_settle`, `tau_fit`, `egap_peak`, `egap_settle`, `overshoot_frac`.
* **Horizonte muito mais curto:** as rodadas de falha única duram ~35–60 s, não 155 s.

### DEF-C — "final": `egap_final`, a ÚLTIMA amostra (não uma média)

Duas origens independentes, mesma semântica:

* `metrics_util.event_metrics` também devolve `egap_final = float(e[-1])`
  (`metrics_util.py:124`);
* `run_comm_sweep.py:91-94` tem um helper **próprio**, que não passa por `metrics_util`
  e devolve `{tau_fit, tau_fit_r2, egap_final, egap_late_std, settled}` — **sem
  `egap_avg` nenhum**.

Não é uma média: é o valor do último instante da rodada. Aparece em
`comm_results*.csv`, `optionB_results.csv`, `figure_data.csv`,
`mmult_adjacent_results.csv` e no `collapse_results.csv` antigo (que **não** tem
`egap_avg` nem `t_settle`, ao contrário dos `collapse_results_c1A_*`).

### DEF-D — recomputação local em `probe_gmax_floor.py`

`experiments/scaling_law/probe_gmax_floor.py:56,67` — média de `E_gap` sobre
`t >= GMAX_T0` (default **20 s**, `:41`, com o comentário "mesma janela do
run_churn_sweep"). Janela idêntica à DEF-A, **código diferente**, e aplicada à telemetria
sobrevivente de `churn_sweep_runs_stamp/`.

> A diferença **não é só a janela** (20 s vs 15 s). É o estímulo e o horizonte: DEF-A é
> **erro de regime sob churn contínuo por 135 s**; DEF-B é **resíduo após um evento
> único**, num horizonte de dezenas de segundos. Os dois números não são comparáveis mesmo
> que as janelas coincidissem.

---

## 2. Qual CSV usa qual — os 41 arquivos que carregam a coluna `egap_avg`

Marcador infalível no cabeçalho: **`egap_p90` presente ⇒ DEF-A**; **`t_settle`/`tau_fit`
presentes ⇒ DEF-B**; nenhum dos dois ⇒ DEF-D.

| definição | janela | arquivos | n |
|---|---|---|---:|
| **DEF-A** (`run_churn_sweep`) | `t ≥ 20 s`, 135 s, churn contínuo | `churn_sweep_results*.csv` | **10** |
| **DEF-B** (`event_metrics`) | `t ≥ 15 s`, pós-evento único | `collapse_results*.csv` (5), `dt_scaling_results*.csv` (7), `ladder_results*.csv` (11), `trackC_results*.csv` (7) | **30** |
| **DEF-D** (`probe_gmax_floor`) | `t ≥ 20 s`, telemetria `stamp` | `gmax_probe_results.csv` | **1** |

### DEF-A — 10 arquivos
`churn_sweep_results.csv`¹, `churn_sweep_results_add_clean.csv`,
`churn_sweep_results_c1B_m8on_dt01.csv`, `churn_sweep_results_c1C_dt05.csv`,
`churn_sweep_results_c2mmult_churn.csv`, **`churn_sweep_results_c3_churn8_dt05.csv`**,
`churn_sweep_results_c4_snappy_tau02.csv`, `churn_sweep_results_gate_clean.csv`,
`churn_sweep_results_m8off_ablation8seed.csv`, `churn_sweep_results_over_clean.csv`

¹ **byte-idêntico** a `churn_sweep_results_c3_churn8_dt05.csv` (mesmo sha256
`c92a002cb14f319a`). Não é campanha distinta; o nome canônico é o segundo.

### DEF-B — 30 arquivos
`collapse_results_c1A_dt01/…_dt05/…_c1Along_dt01/…_dt05/…_c1recheck2` ·
`dt_scaling_results_A_n40/n50/n75/n100a/n100b/A_small/CB` ·
`ladder_results_d001_s0a/s0b/s1a/s1b/s2a/s2b`, `ladder_results_d05_N24/N40s0/N40s12/N50s0/N50s1/N50s2` ·
`trackC_results`, `_churnclean`, `_churnm8`, `_m8clean`, `_recover`, `_srange`, `_stress`

---

## 3. Qual script usa qual

| script | definição | janela efetiva | escreve em |
|---|---|---|---|
| `run_churn_sweep.py:47-56` | **A** | t ≥ 20 s | `churn_sweep_results*.csv` |
| `run_collapse_sweep.py:58` | **B** (`T0 = SCALING_T0`, default 5) | t ≥ 15 s | `collapse_results*.csv` |
| `run_trackC.py:157,169` | **B** (`T0 = 5`) | t ≥ 15 s | `trackC_results*.csv` |
| `run_dt_scaling.py:226` | **B** (`T0 = 5`) | t ≥ 15 s | `dt_scaling_results*.csv` |
| `run_ladder.py:190` | **B** (`T0 = 5`) | t ≥ 15 s | `ladder_results_*.csv` |
| `analyze_breach_window.py:166` | **B** (`t0 = t_fail`) | t ≥ `t_fail` + 10 s | CSV de análise de brecha |
| `probe_gmax_floor.py:56,67` | **D** | t ≥ 20 s | `gmax_probe_results.csv` |
| `analyze_churn_why.py:112` | **A** (recomputa: `T0=5`, `WARMUP=15`) | t ≥ 20 s | só PNG (`churn_why.png`) |
| `diag_churn.py:50` | **B** (`T0 = 5`) | t ≥ 15 s | só stdout |
| `analyze_churn_paired.py` | consome A | herda | `churn_paired_results.csv` |
| `make_figures_robustez.py` | consome A **e** B | herda | figuras 14–24 |
| `analysis_churn/analyze_pi2.py` | consome A | herda | esta pasta |

---

## 3b. DÍVIDA TÉCNICA — pagar antes do Capítulo 6

**Nada foi corrigido.** `make_figures_robustez.py` está intocado. Esta é a lista de
conserto, pronta para quando chegar a hora.

Prioridade rebaixada porque a varredura confirmou que **`fig15` e `tab1` não estão
embutidas em nenhum capítulo nem na proposta** — as 12 imagens realmente incluídas nos
drafts são fig3, fig8, fig14, fig16, fig17, fig18, fig20, fig22, fig23, fig24,
`esquema_pulsos` e `desync_fig1`. `FIGURAS.md:66` as **planeja** para o Cap. 7; é aí que
a dívida vence.

| # | artefato | o que mistura | o que precisa ser feito |
|---|---|---|---|
| D1 | `figures/fig15_mapa_robustez.png` (`make_figures_robustez.py:129-159` + `:162`) | DEF-A (churn 1,31/1,23/1,15/1,14×) · DEF-B (ENTRADA 1,88×; estresse 1,10–1,15×) · τ da lei de escala · DEF-C (perda/atraso) | Acrescentar uma coluna **"métrica/janela"** em `robustness_rows()`, com o valor por linha (`egap_avg t≥20s`, `egap_avg t≥15s`, `t_settle`, `egap_final`). Alternativa mais barata: separar as linhas de razão em dois blocos rotulados ("churn contínuo" vs "evento único") e nunca deixar `1,31×` e `1,88×` na mesma coluna sem rótulo. **Não** basta nota de rodapé. |
| D2 | `figures/tab1_mapa_robustez.png` (`:396`) | idem — lê a **mesma** `robustness_rows()` | Consertar D1 conserta este automaticamente. Verificar depois que a coluna nova aparece na tabela (ela tem `colWidths` fixo em `:381`, precisa de reajuste). |
| D3 | `figures/tabelas_novas.md` §"Tabela 1" (`:446`) | idem — mesma fonte | Idem D1. Confirmar a regeneração do `.md`. |
| D4 | `docs/thesis/draft/relatorio_avancos_orientadores.md:307-318` | tabela **manuscrita** com a mesma mistura; já foi entregue aos orientadores | Editar a coluna "Resultado" para nomear a métrica de cada linha. É o único artefato desta lista que **já circulou**; se houver nova versão do relatório, corrigir antes. |
| D5 | `make_figures_robustez.py:741` | lê `churn_sweep_results.csv`, nome não-canônico e byte-idêntico a `c3_churn8_dt05` | Trocar por `churn_sweep_results_c3_churn8_dt05.csv`. Uma linha. O número não muda; o que muda é a rastreabilidade da campanha. Vale para `:769` (`fig24_esforco`), que reusa a mesma variável. |
| D6 | `diag_churn.py:34,50` | diagnostica **churn** com a janela de **evento** (t≥15 s) | Ou passar `T0+15` na chamada, ou imprimir um aviso de uma linha dizendo que o `egap_avg` dele **não** é o do `churn_sweep_results*.csv`. Só escreve em stdout, então o risco é de cópia manual. |
| D7 | `metrics_util.py:104-127` e `run_churn_sweep.py:47-56` | dois `egap_avg` com o mesmo nome | Renomear é caro (quebra 41 CSVs). O barato: docstring em ambos apontando um para o outro, e um comentário em `metrics_util.py:119` dizendo que a janela é `t0+10`, não `t0`. |

**Ordem sugerida:** D5 (uma linha, zero risco) → D6 (aviso) → D1/D2/D3 (mesma origem) →
D4 (documento externo) → D7 (documentação).

**Regressão a evitar:** qualquer conserto que **recompute** um número existente. Os
valores estão certos; o que falta é o rótulo. Recomputar violaria a regra 2 da campanha
(nunca sobrescrever resultado).

---

## 4. Os três pontos de risco encontrados

### 4.1 `diag_churn.py` diagnostica CHURN com a janela de EVENTO

`diag_churn.py:34` fixa `T0 = 5.0` e `:50` chama `event_metrics(pd.read_csv(tgt), T0)` →
`t ≥ 15 s`. Mas as rodadas que ele diagnostica são de churn, e a campanha oficial as mede
em `t ≥ 20 s`. Um `egap_avg` impresso por `diag_churn.py` **não é** o `egap_avg` do
`churn_sweep_results*.csv` da mesma configuração: inclui 5 s a mais de transiente inicial.
Ele só imprime (não grava CSV), então nenhum resultado publicado depende disso — mas
qualquer número copiado do terminal dele para o texto seria de outra definição.

### 4.2 `make_figures_robustez.py` mistura as duas famílias na mesma figura

`fig15_mapa()` (`:162`) desenha `robustness_rows()` (`:129-159`), que enfileira, num único
painel:

* `"Churn (Poisson, 24 nós, 8 sementes)" → 1.31× / 1.23× / 1.15× / 1.14×` — razões de
  `egap_avg` **DEF-A**;
* `"Recuperação de nó (ENTRADA)" → 0.0094 vs 0.0050` — valores de `egap_avg` **DEF-B**
  (`trackC_results_recover.csv`);
* `"Falha permanente de 1 nó" → τ = 19.5 s / 2.15 s` — `tau_fit`/`t_settle`, **DEF-B**.

Os valores são **hardcoded** na função (não lidos de CSV), e a figura não diz de qual
definição cada linha veio. É o ponto de contato mais direto com o capítulo.

### 4.3 `make_figures_robustez.py:741` lê o nome não-canônico

```python
churn = pd.read_csv(_csv("churn_sweep_results.csv"))
```

É o arquivo byte-idêntico à campanha `c3_churn8_dt05`. A figura 14 e a figura 24
(`fig24_esforco`, `:769`) saem dessa leitura. O número está certo; o **nome** é o que não
deve ser citado, porque não identifica a campanha.

---

## 5. O que verificar antes de escrever qualquer frase que compare campanhas

1. Um `egap_avg` de `churn_sweep_results*.csv` **nunca** deve ser comparado com um de
   `collapse_/dt_scaling_/ladder_/trackC_results*.csv` sem dizer que as definições diferem.
2. `egap_p90` e `egap_max` **só existem na DEF-A** — não há tail statistic equivalente para
   as campanhas de evento único.
3. Ao citar um número de churn, nomear a campanha (`c3_churn8_dt05`) e não o arquivo
   `churn_sweep_results.csv`.
4. Ao citar um `t_settle`/`tau`, é sempre DEF-B — é a definição que **tem** tempo de
   assentamento. A DEF-A não mede assentamento nenhum.
