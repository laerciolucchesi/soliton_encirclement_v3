# RESULTADO — o ganho da coordenacao x Pi_2' (campanha c3_churn8_dt05)

> **Enunciado da figura.** Conforme o churn aperta, o ganho **migra do corpo da distribuicao para a cauda superior**, e o **extremo permanece intocado** em todos os regimes.

> Substitui o enunciado anterior ("a vantagem cai com Pi_2'"), que era **falso**: cai na media (1,31 -> 1,14, 8/8 sementes) e **sobe** no P90 (1,04 -> 1,13, 8/8 sementes). As duas direcoes tem o mesmo p bilateral, 0,0078.

Fonte canonica: `experiments/scaling_law/churn_sweep_results_c3_churn8_dt05.csv`, sha256[:16] = `c92a002cb14f319a`.  
`churn_sweep_results.csv` e' **byte-identico** a esta fonte (mesmo sha256) — nao e' campanha distinta e **nao deve ser citado em lugar nenhum**.  
Calculo pareado (mediana, razoes, Wilcoxon) **reusado** de `experiments/scaling_law/analyze_churn_paired.py` (`paired_values`, `ratios`, `wilcoxon_paired`), verificado 25/25 celulas a rtol=1e-9 contra `churn_paired_results.csv`.

## 4.1 Semantica de `rate_total` (FASE 0)

**`rate_total` = taxa TOTAL do anel** (falhas/min somadas sobre os N agentes). Cadeia: `experiments/scaling_law/run_churn_sweep.py:61` calcula `per_agent = rate_total / float(N)` e `:76` grava esse valor em `FAILURE_MEAN_FAILURES_PER_MIN`; `protocol_agent.py:904-908` usa essa taxa num sorteio Bernoulli `p = 1-exp(-(rate/60)*dt)` com `dt = FAILURE_CHECK_PERIOD = 0.1 s` (`config_param.py:178`), executado **por agente** (`protocol_agent.py:918-920`, RNG dedicado semeado em `:99-101`, timer proprio em `:266-267`). As duas operacoes se cancelam: taxa do anel = N x (rate_total/N) = `rate_total`.

Confirmacao independente por dado de execucao: os `runs_summary.csv` preservados em `churn_sweep_runs_stamp/` registram `failure_mean_per_min` = 0,25 / 0,5 / 1,0 / 2,0 para rate 6 / 12 / 24 / 48 com N=24 — exatamente `rate_total/N`.

Teste de coerencia: sob a hipotese 'por agente', rate_total=12 exigiria 38.4 agentes ausentes num anel de 24 — impossivel.

## 4.2 Tabela principal — `egap_avg`

| rate_total (/min) | lambda_anel (1/s) | Pi_2' | % do anel | Pi_2' renovacao* | % renovacao | mediana baseline | mediana B2 | razao pareada mediana [IQR] | p Wilcoxon | sementes a favor / contra |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.1000 | 0.80 | 3.3% | 0.77 | 3.2% | 0.0819 | 0.0626 | 1.314 [1.256, 1.321] | 0.007812 | **8/8** a favor, 0/8 contra |
| 12 | 0.2000 | 1.60 | 6.7% | 1.50 | 6.2% | 0.1293 | 0.1024 | 1.226 [1.186, 1.257] | 0.007812 | **8/8** a favor, 0/8 contra |
| 24 | 0.4000 | 3.20 | 13.3% | 2.82 | 11.8% | 0.1982 | 0.1719 | 1.155 [1.151, 1.172] | 0.007812 | **8/8** a favor, 0/8 contra |
| 48 | 0.8000 | 6.40 | 26.7% | 5.05 | 21.1% | 0.2840 | 0.2449 | 1.136 [1.128, 1.165] | 0.007812 | **8/8** a favor, 0/8 contra |

> ### Os quatro p sao o PISO do teste, nao a forca do efeito

> Os quatro p de `egap_avg` valem **0.007812 = 2/2^8**, que e' o **menor p bilateral possivel** no Wilcoxon exato com n=8. Ele so diz "as 8 sementes concordam no sinal" — e' o mesmo p para 1,31 e para 1,14. **O teste por taxa NAO distingue 1,31 de 1,14 e nao deve ser usado para sustentar a tendencia.**

> A tendencia se apoia em duas outras coisas, e so nelas:
> 1. o **teste da FASE 2b** (secao 4.3), sobre 8 observacoes **independentes** — cada semente contribui UMA diferenca;
> 2. os **IQR que nao se sobrepoem** entre os extremos da varredura: [1.256; 1.321] na taxa minima vs [1.128; 1.165] na taxa maxima.

> **Regra adotada neste documento:** todo `p` reportado vem acompanhado da contagem de sementes a favor/contra. Um `p` sozinho, com n=8, nao e' informacao suficiente.

\* `Pi_2' renovacao` = `Pi_2'/(1 + Pi_2'/N)` — numero medio exato de ausentes, dado que um agente OFF nao sorteia novas falhas (`protocol_agent.py:920-922` nao reagenda o timer; `:966` so reagenda na recuperacao). `Pi_2' = lambda_anel*T_off` e' a aproximacao de baixa densidade; o eixo x das figuras usa `Pi_2'`.

Wilcoxon pareado bilateral sobre os valores brutos, n=8 por taxa.

### As outras duas metricas — onde o ganho vai parar

**`egap_p90`** — cauda superior (P90)

| rate_total | Pi_2' | mediana baseline | mediana B2 | razao mediana [IQR] | p Wilcoxon | sementes a favor / contra |
|---|---|---|---|---|---|---|
| 6 | 0.80 | 0.1811 | 0.1708 | 1.045 [1.026, 1.055] | 0.015625 | 7/8 a favor, 1/8 contra |
| 12 | 1.60 | 0.2294 | 0.2112 | 1.053 [1.046, 1.071] | 0.007812 | 8/8 a favor, 0/8 contra |
| 24 | 3.20 | 0.2956 | 0.2737 | 1.089 [1.061, 1.100] | 0.007812 | 8/8 a favor, 0/8 contra |
| 48 | 6.40 | 0.4179 | 0.3675 | 1.129 [1.110, 1.144] | 0.007812 | 8/8 a favor, 0/8 contra |

**`egap_max`** — extremo (max no tempo)

| rate_total | Pi_2' | mediana baseline | mediana B2 | razao mediana [IQR] | p Wilcoxon | sementes a favor / contra |
|---|---|---|---|---|---|---|
| 6 | 0.80 | 0.2987 | 0.2878 | 0.988 [0.956, 1.093] | 0.945312 | 3/8 a favor, 5/8 contra |
| 12 | 1.60 | 0.3847 | 0.3270 | 1.061 [0.989, 1.312] | 0.250000 | 5/8 a favor, 3/8 contra |
| 24 | 3.20 | 0.4830 | 0.4067 | 1.155 [1.106, 1.186] | 0.015625 | 7/8 a favor, 1/8 contra |
| 48 | 6.40 | 0.6144 | 0.6296 | 0.984 [0.964, 1.162] | 0.742188 | 3/8 a favor, 5/8 contra |

### Forma da distribuicao (FASE 2c) — o mecanismo

`p90/avg` e `max/avg` calculados **por rodada** e depois mediana entre sementes. A coluna "(medianas)" e' a razao das medianas da tabela 4.2 — a que se obtem lendo a tabela de fora; as duas sao dadas porque diferem.

| rate_total | Pi_2' | metodo | P90/media [IQR] | (medianas) | max/media [IQR] | (medianas) |
|---|---|---|---|---|---|---|
| 6 | 0.80 | baseline | 2.104 [1.969, 2.260] | 2.210 | 3.627 [3.096, 4.056] | 3.645 |
| 6 | 0.80 | B2 | 2.677 [2.446, 2.761] | 2.727 | 4.341 [4.059, 5.472] | 4.597 |
| 12 | 1.60 | baseline | 1.770 [1.728, 1.813] | 1.774 | 2.978 [2.655, 3.432] | 2.975 |
| 12 | 1.60 | B2 | 2.023 [1.953, 2.120] | 2.062 | 3.273 [3.125, 3.412] | 3.192 |
| 24 | 3.20 | baseline | 1.507 [1.477, 1.546] | 1.492 | 2.535 [2.456, 2.706] | 2.437 |
| 24 | 3.20 | B2 | 1.631 [1.605, 1.642] | 1.592 | 2.547 [2.456, 2.616] | 2.367 |
| 48 | 6.40 | baseline | 1.436 [1.429, 1.470] | 1.471 | 2.156 [2.056, 2.334] | 2.164 |
| 48 | 6.40 | B2 | 1.492 [1.473, 1.509] | 1.500 | 2.418 [2.296, 2.514] | 2.570 |

### Media das razoes vs razao das medias

- `egap_avg`: 6: 1.2946 vs 1.2902; 12: 1.2239 vs 1.2177; 24: 1.1609 vs 1.1609; 48: 1.1432 vs 1.1413 — divergencia relativa maxima 0.51%
- `egap_p90`: 6: 1.0426 vs 1.0426; 12: 1.0609 vs 1.0609; 24: 1.0799 vs 1.0787; 48: 1.1236 vs 1.1227 — divergencia relativa maxima 0.11%
- `egap_max`: 6: 1.0266 vs 1.0322; 12: 1.1250 vs 1.1103; 24: 1.1530 vs 1.1535; 48: 1.0810 vs 1.0698 — divergencia relativa maxima 1.32%

## 4.3 Tendencia: a vantagem cai com o churn?

Teste **pareado por semente** (FASE 2b): para cada semente, `delta_s = razao(taxa minima) - razao(taxa maxima)`; `delta_s > 0` = a vantagem caiu. n = 8 diferencas **independentes** (cada semente entra uma vez).

| metrica | taxas comparadas | sementes com queda | sementes com alta | direcao | Wilcoxon p (exato) | sinal p (exato) | [contraste] Spearman 32 pares rho | p |
|---|---|---|---|---|---|---|---|---|
| egap_avg | 6 vs 48 | 8/8 | 0/8 | **CAI com o churn (unanime)** | 0.007812 | 0.007812 | -0.8204 | 9.0572e-09 |
| egap_p90 | 6 vs 48 | 0/8 | 8/8 | **SOBE com o churn (unanime)** | 0.007812 | 0.007812 | 0.6630 | 3.5537e-05 |
| egap_max | 6 vs 48 | 4/8 | 4/8 | **sem direcao (empate)** | 0.843750 | 1.000000 | 0.0848 | 0.644622 |

O p do Wilcoxon aqui e' **bilateral**: mede se `delta_s` difere de zero, nao a direcao. Leia a direcao na coluna correspondente — `egap_avg` e `egap_p90` tem o mesmo p (0,0078, o piso com n=8) e direcoes **opostas**.

O Spearman de 32 pares **viola independencia** (a mesma semente aparece nas 4 taxas), entao seu p e' otimista por construcao; esta na tabela apenas como contraste. O teste de referencia e' o pareado por semente.

## 4.4 ESCOPO DECLARADO (constante em toda a campanha)

- N = 24 agentes
- tau_xy = 1 s
- T_off = 8 s — recuperacao finita: os agentes VOLTAM (churn, nao morte)
- dt (`CONTROL_PERIOD`) = 0,05 s (`run_churn_sweep.py:44`)
- **regime de saturacao: `sat_frac` == 0 em 100% das 64 celulas — campanha inteiramente no regime NAO SATURADO do atuador. Isto e' DECLARACAO DE ESCOPO, nao falha.**
- metrica: `egap_avg`/`egap_p90`/`egap_max` = media / P90 / MAXIMO **no tempo** de `E_gap`, sobre t >= 20 s ate 155 s (`run_churn_sweep.py:41-42,51-56`). `E_gap` e' o **RMS espacial** do erro relativo de vao, normalizado pelo numero de agentes **VIVOS** (`protocol_target.py:707`)
- n = 8 sementes por celula, pareadas entre metodos: baseline e B2 compartilham `EXPERIMENT_SEED`, e o RNG de falha e' dedicado e independente do metodo (`protocol_agent.py:99-101`, `:918-919`), logo o fluxo de falhas e' o MESMO nos dois
- taxas varridas: [6.0, 12.0, 24.0, 48.0] falhas/min TOTAIS (Pi_2' de 0.80 a 6.40 agentes)
- canal ideal: `COMMUNICATION_FAILURE_RATE=0`, `COMMUNICATION_DELAY=0` (`run_churn_sweep.py:72`)
- alvo estacionario: `TARGET_MOTION_SPEED_XY=0` (`run_churn_sweep.py:73`)
- inicializacao equidistante, sem dispersao de raio (`run_churn_sweep.py:73`)
- ganho escalado: `K_E_TAU = 250/N` (`run_churn_sweep.py:70`)

## 4.5 O QUE ESTA FIGURA NAO MOSTRA

- **Nao mostra tempo de assentamento.** `egap_avg` e' erro de REGIME PERMANENTE (media temporal sobre t >= 20 s). Nenhum `t_settle` foi medido nesta campanha; `metrics_util.settling_time` existe mas nao e' chamado por `run_churn_sweep.py`.
- **Nao e' comparavel com a vantagem medida no evento de falha unica.** Aquela usa falha deterministica e mede o transiente pos-evento; esta usa fluxo de Poisson com recuperacao e mede media temporal de regime. Estimulo, janela e metrica diferem.
- **Nao e' o vao angular maximo.** `egap_max` e' o maximo NO TEMPO de um RMS ESPACIAL — duas agregacoes empilhadas. O vao maximo e' `G_max` (`protocol_target.py:706`), que **nao existe neste CSV**.
- **Nao mede cobertura absoluta.** `E_gap` e `G_max` sao normalizados pelo numero de vivos: um anel com metade dos agentes, perfeitamente redistribuido, pontua igual a um anel cheio. Mede QUALIDADE DE REDISTRIBUICAO.
- **Nao separa taxa de duracao.** `T_off` e' constante, entao Pi_2' e `rate_total` sao proporcionais: a figura nao distingue efeito da TAXA do efeito da DURACAO da ausencia. Seria preciso variar `T_off` com Pi_2' fixo.
- **Nao varre N nem tau_xy.** Uma unica coluna do espaco de projeto (N=24, tau_xy=1). Nada aqui sustenta extrapolacao em N.
- **Nao mede custo.** `effort_mean_v2` e `fairness_p95` estao no CSV e ficaram fora desta figura de proposito; `analyze_churn_paired.py` ja os reporta (custo B2/baseline = 2,41x mediano, 32/32 pares).
- **Sem correcao para multiplas comparacoes** entre taxas e metricas; os p sao crus.
- **Piso de resolucao do teste:** com n=8, o menor p bilateral exato possivel e 0.007812. Um p igual a esse valor significa 'o maximo que este n permite afirmar', nao 'efeito enorme'.
- **`sat_frac` e' degenerada:** todas as diferencas sao exatamente zero (S10). O Wilcoxon exato e' indefinido nesse caso e a razao seria 1,000 por construcao, entao a metrica fica fora da figura. Para `sat_frac` isto e' **declaracao de escopo** (regime nao saturado do atuador), nao falha de medicao — ver 4.4.

---
Ver tambem `EGAP_HOMONIMO.md` nesta pasta: existem **duas** definicoes de `egap_avg` no repositorio, com janelas e estimulos diferentes. Todos os numeros acima sao da definicao do `run_churn_sweep.py` (t >= 20 s, churn continuo).

Gerado por `analysis_churn/analyze_pi2.py`. Log completo: `LOG_execucao.txt`. Dados por par: `paired_ratios.csv`; por taxa: `summary_by_rate.csv`.
