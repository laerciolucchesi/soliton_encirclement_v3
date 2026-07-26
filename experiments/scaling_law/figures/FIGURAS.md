# Pacote de figuras — avanços da pesquisa

Protocolo comum a todas: **falha permanente de 1 nó** (nó `2+N//2`) em t₀=5 s,
ganho estável `K_E_TAU=250/N`, init equidistante, alvo parado. Baseline =
controlador local sozinho; **overlay 2-DOF** = dual-pulse (hop-count) +
feedforward (integração B2). Dados: `figure_data.csv` (+ telemetria reusada do
baseline, telemetria do B2 regenerada de forma determinística).

| # | Arquivo | O que mostra | Mensagem de 1 linha |
|---|---------|--------------|---------------------|
| 1 | `fig1_lei_de_escala.png` | τ vs N (log-log), baseline vs overlay | **A FIGURA**: baseline Θ(N²), overlay τ-plano |
| 2 | `fig2_speedup.png` | τ_base/τ_overlay vs N | Vantagem cresce ~N² (9×→149×) |
| 3 | `fig3_recuperacao_baseline.png` | E_gap(t) p/ todos N (baseline) | Quanto maior N, mais lenta a recuperação |
| 4 | `fig4_recuperacao_B2.png` | E_gap(t) p/ todos N (overlay) | Curvas colapsam — τ independe de N |
| 5 | `fig5_comparacao_N100.png` | baseline vs overlay em N=100 | ~149× mais rápido (5 min → 2 s) |
| 6 | `fig6_painel_resumo.png` | painel 2×2 (a–d) | Imagem única para slide/e-mail |
| 7 | `fig7_kymograph.png` | espaço-tempo dos pulsos (N=50) | O algoritmo distribuído: anel "sabe" em ~0,4 s |
| 8 | `fig8_anel_snapshots.png` | anel antes/durante/depois (N=10) | Intuição geométrica do fechamento da lacuna |
| 9 | `fig9_mensagens.png` | payloads vs N | Rápido **E** barato: O(N) total, O(1)/agente |
| 10 | `fig10_tabela.png` | τ e R² por algoritmo e N (+ aceleração) | Tabela-resumo dos números brutos |
| 11 | `fig11_decomposicao_temporal.png` | esforço overlay vs baseline ao longo do tempo (N=50) | O overlay redistribui (0–2,5 s), depois "dorme" e entrega ao baseline (handover ~2,8 s) |
| 12 | `fig12_esforco_baseline.png` | esforço de controle SÓ com baseline (a: todos N; b: vs overlay em N=50) | Baseline sozinho: empurrão fraco e longo; overlay: forte e curto |
| 13 | `fig13_equacao_Egap.png` | card da equação da métrica principal $E_{gap}$ | Definição + interpretação para slide |

## Bloco 2 — Robustez, teoria adimensional e ablações (campanha Ciclos 0–2)

Aqui o protocolo VARIA por figura (churn de Poisson, perda, atraso, alvo em
movimento, falhas múltiplas) — não é mais a falha única limpa. Mesma paleta.
Dados: CSVs canônicos de `experiments/scaling_law/` (índice em
`docs/experiments/README.md`).

| # | Arquivo | O que mostra | Mensagem de 1 linha |
|---|---------|--------------|---------------------|
| 14 | `fig14_churn_robustez.png` | vantagem pareada sob churn, 8 sementes | Ajuda em **8/8 sementes**, todas as taxas (0 perdas) |
| 15 | `fig15_mapa_robustez.png` | scorecard de todos os eixos de estresse | **Imagem única**: onde acelera / onde degrada graciosamente |
| 16 | `fig16_perda_pacotes.png` | τ vs perda (faixa entre sementes) | Degradação graciosa: vai de ~2 s ao baseline, sempre assenta |
| 17 | `fig17_atraso_m8.png` | atraso: M8 off vs on + sweep | O "limite" era artefato do M8; com M8 é suave em segundos |
| 18 | `fig18_alvo_movel.png` | E_gap e E_r por cenário (alvo móvel) | Acelera o espaçamento **sem degradar o rastreamento** |
| 19 | `fig19_lei_adimensional.png` | colapso A vs N²/τₐ vs Péclet | **A ≈ 0,017·N²/τₐ** (Péclet refutado); τ invariante a dt |
| 20 | `fig20_escada_feedforward.png` | vantagem A/B/B2 vs N | Só o B2 escala; A encolhe (1,68→1,14×) |
| 21 | `fig21_m8_ablacao.png` | vantagem churn, M8 off vs on | M8 leva o regime denso de 0,93× (prejudica) a 1,21× |
| 22 | `fig22_mmult_adjacente.png` | τ por cenário, M-mult off vs on | Falhas adjacentes 13,7/15,4→2,2 s (**6–7×**); k=1 intocado |
| 23 | `fig23_ttl_cobertura.png` | cobertura vs N, TTL fixo vs 3N | TTL≥N é requisito: TTL=50 colapsa p/ **1% em N=100** |
| 24 | `fig24_esforco_sem_windup.png` | esforço e velocidade RMS vs Vmax | Custo ~2,4× de atuação, mas sat_frac=0 ⇒ **sem windup** |
| T1 | `tab1_mapa_robustez.png` | mapa de robustez (tabela auditável) | Versão auditável do fig15 (baseline vs overlay por célula) |
| T2 | `tab2_churn_vantagem.png` | vantagem churn 8 sementes (tabela) | med/min/máx + esforço; 0 sementes perdidas |

(Markdown das tabelas para colar no texto: `tabelas_novas.md`.)

## Sugestão de ordem para os slides

1. **Problema** → fig8 (o que acontece quando um nó falha) + fig3 (e por que escala mal).
2. **Resultado** → fig1 (a lei de escala) + fig5 (o contraste dramático) + fig2 (a vantagem).
3. **Como** (o algoritmo, Cap. 3) → fig7 (kymograph) + fig9 (custo de mensagens).
4. **Por que o design** → fig20 (escada A/B/B2) + fig19 (lei adimensional).
5. **Funciona no mundo real?** → fig15 (mapa) → fig14 (churn) + fig16 (perda) + fig17 (atraso) + fig18 (alvo móvel).
6. **Robustez de projeto** → fig21 (M8) + fig22 (M-mult) + fig24 (esforço sem windup).
7. **Resumo** → fig6 (painel) + fig15 (mapa) num slide só.

## Ligação com os capítulos

- **Cap. 2 (trilema):** fig1, fig3.
- **Cap. 3 (algoritmo hop-count):** fig7, fig8, fig9, fig22, fig23.
- **Cap. 4 (escape 2-DOF):** fig1, fig2, fig4, fig5, fig20.
- **Cap. 6 (lei adimensional):** fig19.
- **Cap. 7 (robustez):** fig14, fig15, fig16, fig17, fig18, fig21, fig24, tab1, tab2.

## Reproduzir

```powershell
python experiments/scaling_law/gen_figure_data.py    # baseline reusado + B2 regenerado
python experiments/scaling_law/make_figures.py        # fig1..6
python experiments/scaling_law/make_figures_extra.py  # fig7..9
# Bloco 2 (robustez/teoria/ablações):
python experiments/scaling_law/run_mmult_adjacent.py  # gera mmult_adjacent_results.csv (fig22)
python experiments/scaling_law/run_ttl_coverage.py    # gera ttl_coverage_results.csv (fig23)
python experiments/scaling_law/make_figures_robustez.py  # fig14..24 + tab1/tab2 (+ tabelas_novas.md)
```

> fig10..13 vêm de `make_table.py`, `make_decomposition.py`, `make_baseline_effort.py`,
> `make_eq_card.py` (um por figura). O `make_figures_robustez.py` lê os CSVs canônicos;
> só fig22/fig23 dependem dos dois runners acima (rodada curta).
