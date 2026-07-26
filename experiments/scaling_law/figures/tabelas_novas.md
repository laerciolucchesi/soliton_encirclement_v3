# Tabelas novas (campanha de robustez)

## Tabela 1 — Mapa de robustez

| Família | Cenário | Baseline | Overlay 2-DOF | Resultado / veredito |
|---|---|---|---|---|
| Falha permanente de 1 nó | N = 24 | τ = 19.5 s | τ = 2.15 s | 9×  mais rápido |
|  | N = 100 | τ = 311 s | τ = 2.09 s | 149×  mais rápido |
| Churn (Poisson, 24 nós, 8 sementes) | 6 falhas/min | — | — | 1.31×  (min 1.24) |
|  | 12 falhas/min | — | — | 1.23×  (min 1.14) |
|  | 24 falhas/min | — | — | 1.15×  (min 1.11) |
|  | 48 falhas/min | — | — | 1.14×  (min 1.11) |
| Comunicação imperfeita | Perda ≤ 20% | assenta | assenta | speedup encolhe (gracioso) |
|  | Perda 40% | assenta | assenta | inerte (= baseline) |
|  | Atraso 0.1 s | τ = 22 s | τ = 3.1 s | 7×  (assenta) |
|  | Atraso 0.5 s | τ = 31 s | τ = 8.9 s | 3.5×  (assenta, sem cliff) |
|  | Fora de ordem / duplicado | — | rejeitado | seq# por emissor (testado) |
| Alvo em movimento | Velocidade constante | — | — | ajuda · rastreio $E_r$ intacto |
|  | Manobra | — | — | ajuda (diluído) · $E_r$ intacto |
|  | Recuperação de nó (ENTRADA) | 0.0094 | 0.0050 | 1.88× |
| Coordenação / casos difíceis | Falhas adjacentes (M-mult) | τ = 13.6 s | τ = 2.2 s | 6–7×  (corrigido) |
|  | Estresse combinado | — | — | 1.10–1.15× |
|  | ENTRADA c/ canônico morto | — | ~3/24 denso | não coberto (futuro) |

## Tabela 2 — Vantagem sob churn (pareada, 8 sementes, dt=0.05)

| Taxa [falhas/min] | Vant. mediana | Vant. mínima | Vant. máxima | Sementes ajudadas | Esforço B2/base |
|---|---|---|---|---|---|
| 6 | 1.31× | 1.24× | 1.34× | 8/8 | 2.5× |
| 12 | 1.23× | 1.14× | 1.30× | 8/8 | 2.4× |
| 24 | 1.15× | 1.11× | 1.20× | 8/8 | 2.4× |
| 48 | 1.14× | 1.11× | 1.18× | 8/8 | 2.4× |

_Lei adimensional (fig19):_ A ≈ 0.0170·N²/τ_a  (colapso CV ≈ 17% vs Péclet CV ≈ 62%).
