# Capítulo 3 — O trilema do controlador local

> Rascunho (PT). Status: `[provado]` (resultado negativo central + backbone teórico fino).
> Resultados: `experiments/scaling_law/` (experimento de normalização de ganho).

## 3.1 O controlador local é um Laplaciano normalizado

O controlador de espaçamento que cada agente executa — corrigir o próprio ângulo na direção
do ponto médio entre predecessor e sucessor — é, em forma linearizada, um **Laplaciano
discreto normalizado** sobre o anel, ou seja, uma **difusão**. Isso o coloca exatamente na
família cuja taxa de convergência é governada pela *algebraic connectivity* $\lambda_2$ do
grafo (Cap. 2, §2.5): no anel $C_N$, $\lambda_2 = 2(1-\cos(2\pi/N)) \approx (2\pi/N)^2 =
\Theta(1/N^2)$, e a relaxação por difusão custa $\Theta(1/\lambda_2) = \Theta(N^2)$ rounds.
O *backbone* teórico fino do capítulo é, portanto: (i) o **limite inferior $\Omega(N)$**
(diâmetro do anel — a informação precisa atravessá-lo); e (ii) o **baseline difusivo
$O(N^2)$** sob ganho estável.

## 3.2 A descoberta: normalização compra velocidade ao preço da estabilidade

O ponto central — e o resultado **negativo** que ancora a tese — é que **não há sintonia que
dê os três simultaneamente** (estável + rápido + escalável):

- A normalização que dá velocidade $O(N)$ injeta um **ganho efetivo $\sim N$** (porque divide
  pelo gap $2\pi/N$). Esse ganho desestabiliza os **modos rápidos** em $N$ grande — surge um
  **ciclo-limite** (observado em $N=50$).
- Forçar estabilidade reduzindo o ganho a $\sim 1/N$ recupera a estabilidade mas devolve o
  baseline difusivo: **$O(N^2)$**.

Não é, portanto, uma questão de ajustar parâmetros: é **estrutural**. O controlador local
está preso ao trade-off velocidade↔estabilidade, e ambos pioram com $N$.

## 3.3 Evidência empírica

O experimento de normalização de ganho confirma os dois ramos:

- **P1 (estabiliza):** com ganho estável $\sim 1/N$, o sistema converge sem ciclo-limite.
- **P2 (escala):** o tempo de estabilização do baseline com ganho estável cresce como
  $\tau \sim N^{2.02}$ — i.e., $O(N^2)$, o preço da estabilidade.

Tabela-resumo (tempo $\tau$ do modo lento, falha única controlada):

```
                                 N=24     N=40     N=50
baseline (ganho fixo alto)       7.08    12.26    140.1 (INSTÁVEL)   -> O(N) até ~N=40, depois estoura
baseline (ganho estável ~1/N)   19.48    54.79     85.35            -> O(N^2.02) (preço da estabilidade)
```

## 3.4 Conclusão

O controlador local **não tem os três ao mesmo tempo**. Esse é o resultado negativo que
motiva o overlay: para escapar do trilema é preciso **desacoplar a coordenação da atuação** —
computar o alvo de reconfiguração por um canal discreto rápido (Cap. 4) e injetá-lo sem
realimentar o ganho instável (Cap. 5).

> **A formalizar (Fase 4):** o argumento linearizado de 1–2 páginas — gap espectral do anel,
> o fator $N$ da normalização, a margem de estabilidade dos modos altos e o $O(N^2)$ sob ganho
> estável. As medições já fornecem os expoentes; falta o argumento fechado.

> **Nota — período de controle ($dt$) e estabilidade amostrada.** A margem de estabilidade do
> trilema não depende só do ganho: depende também do **período de controle $dt$** (controle
> amostrado). $dt$ menor melhora a fidelidade/estabilidade mas custa caro; $dt$ maior barateia
> mas **encolhe a margem** e pode estreitar ainda mais o trilema. Como $dt$ entra no número de
> Péclet ($\mathrm{Pe}=N\,dt/\tau_a$), o estudo quantitativo desse efeito é feito no **Cap. 6**
> (caracterização adimensional); aqui fica registrado o gancho.
