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

## Sugestão de ordem para os slides

1. **Problema** → fig8 (o que acontece quando um nó falha) + fig3 (e por que escala mal).
2. **Resultado** → fig1 (a lei de escala) + fig5 (o contraste dramático) + fig2 (a vantagem).
3. **Como** (o algoritmo, Cap. 3) → fig7 (kymograph) + fig9 (custo de mensagens).
4. **Resumo** → fig6 (painel) num slide só.

## Ligação com os capítulos

- **Cap. 2 (trilema):** fig1, fig3.
- **Cap. 3 (algoritmo hop-count):** fig7, fig8, fig9.
- **Cap. 4 (escape 2-DOF):** fig1, fig2, fig4, fig5.

## Reproduzir

```powershell
python experiments/scaling_law/gen_figure_data.py    # baseline reusado + B2 regenerado
python experiments/scaling_law/make_figures.py        # fig1..6
python experiments/scaling_law/make_figures_extra.py  # fig7..9
```
