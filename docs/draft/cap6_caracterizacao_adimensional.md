# Capítulo 6 — Quando e quanto compensa? Caracterização adimensional

> Rascunho (PT). Status: `[núcleo medido; redação inicial]`. Dados:
> `experiments/scaling_law/collapse_results.csv`; análise `analyze_collapse.py`; figura
> `collapse_advantage.png`. Plano de execução: [`docs/plano_colapso_fase2.md`](../plano_colapso_fase2.md).

## 6.1 A pergunta e a tese do capítulo

Os Caps. 3–5 estabeleceram que o controlador local enfrenta um trilema e que o overlay
*feedforward* 2-DOF (B2) o quebra. Resta a pergunta de **princípio + caracterização**: *quando*
e *quanto* o overlay compensa, de forma **adimensional e válida para qualquer plataforma**?

A hipótese de partida era um **número de Péclet de coordenação**, $\mathrm{Pe}=N\,dt/\tau_a$
(latência de informação sobre tempo de atuação): o overlay valeria quando a coordenação fosse
limitada por informação. **Os experimentos refutaram essa hipótese** — e revelaram algo mais
limpo. A tese deste capítulo passa a ser:

> O ganho do overlay é governado pelo grupo adimensional **$N^2/\tau_a$** — a razão entre a
> relaxação difusiva do baseline ($\Theta(N^2)$) e o tempo de atuação do UAV ($\tau_a$) — e
> **não** por um número de Péclet de informação. Além disso, o resultado é **robusto ao período
> de controle** $dt$ (a malha pode rodar com frequência muito menor sem perder o ganho).

## 6.2 Montagem experimental

Regime limpo, evento único controlado: anel uniforme; no instante $t_0$ um nó falha
permanentemente; mede-se o **tempo de estabilização do modo lento** $\tau$ (constante de tempo
do ajuste exponencial da cauda de $E_{\text{gap}}$). Compara-se **baseline** (controlador local,
ganho estável $K_{E\tau}=250/N$) com **B2** (overlay *feedforward* completo). Os knobs do overlay
são **adaptados à agilidade** (regra do Cap. 5): $T_\text{FF}=\tau_a$, $\texttt{DELTA\_SCALE}=1{,}0$,
$\texttt{TTL}=3N$; velocidade máxima **fixa** (sem reescalar com $\tau_a$ — escolha "opção A",
ver §6.6). Varre-se $N\in\{8,16,24\}$, $\tau_a\in\{0{,}2;0{,}5;1{,}0;2{,}0\}$ e, num eixo
dedicado, $dt\in\{0{,}01;0{,}02;0{,}05;0{,}1\}$. O caso simétrico de falha única é
essencialmente **determinístico** (três *seeds* com perturbação de condição inicial deram
$\tau$ idêntico a $\sim 0{,}1\%$), de modo que cada célula é reportada como um valor único.

## 6.3 Resultado central: duas leis de escala

Os dados decompõem-se em **duas leis universais**, ambas em segundos e **invariantes ao período
de controle**:

$$\tau_{\text{base}} \approx 0{,}033\,N^2 \qquad\text{(CV 8\%)}, \qquad
  \tau_{B2} \approx 2{,}3\,T_\text{FF} = 2{,}3\,\tau_a \qquad\text{(CV 16\%).}$$

A primeira é a relaxação difusiva $\Theta(N^2)$ do baseline (expoente medido $2{,}0\!-\!2{,}2$ em
todos os $\tau_a$, e $1{,}97$ no intervalo $N{=}24\!-\!100$ dos dados de larga escala); ela
**não depende de $\tau_a$**. A segunda é o tempo do *feedforward*, **linear em $\tau_a$**
(confirmado sobre um intervalo de $4\times$, $\tau_a\in\{0{,}5;1{,}0;2{,}0\}$) e **independente
de $N$**. A razão das duas dá a **vantagem**:

$$A \;=\; \frac{\tau_{\text{base}}}{\tau_{B2}} \;\approx\; 0{,}014\,\frac{N^2}{\tau_a}.$$

Ou seja: enxames maiores e UAVs mais ágeis (menor $\tau_a$) ampliam o ganho; um enxame pequeno
e ágil não precisa do overlay (em $N{=}8$, $A\approx 1$, porque o baseline já é rápido).

## 6.4 O grupo adimensional é $N^2/\tau_a$ — não um número de Péclet

A vantagem **colapsa** numa curva única quando plotada contra $N^2/\tau_a$ (coeficiente de
variação $\sim 20\%$, mais apertado excluindo o regime saturado da §6.6), e **não** colapsa
contra $\mathrm{Pe}=N\,dt/\tau_a$ (CV $\sim 64\%$). A razão é estrutural: o baseline é
$\Theta(N^2)$ e o overlay é $\propto \tau_a$ (plano em $N$), então a razão escala com
$N^2/\tau_a$, que é **linear em $N$ no Péclet** — grupos diferentes.

Por que a latência de informação não governa? Porque, na faixa realista testada, a disseminação
($O(N\,dt)$) **nunca se torna o gargalo**: o tempo do overlay é dominado pela atuação
($\tau_{B2}\propto\tau_a$), e variar o Péclet até $\mathrm{Pe}=3{,}2$ (via $dt$) **não** fez
$\tau_{B2}$ crescer. A hipótese de uma transição "atuação-limitada → informação-limitada" em
$\mathrm{Pe}\sim 1$ foi, portanto, **refutada** nessa faixa. (Uma eventual quebra em
$\mathrm{Pe}\gg 3$ fica como questão em aberto, §6.7.)

Vale registrar a honestidade do caminho: sob **sintonia fixa** (T_FF constante), a vantagem ao
longo do eixo de agilidade exibia uma "corcova" não-monotônica que sugeria uma janela ótima de
agilidade. Com os knobs **adaptados** ($T_\text{FF}=\tau_a$), essa corcova desaparece e a
vantagem vira monotônica em $N^2/\tau_a$ — a janela era artefato de sintonia, não um fenômeno.

## 6.5 Robustez ao período de controle ($dt$)

Num eixo dedicado ($N{=}16$, $\tau_a{=}0{,}5$, $dt$ de $0{,}01$ a $0{,}1$, ou seja $\mathrm{Pe}$
de $0{,}32$ a $3{,}2$), tanto $\tau_{\text{base}}$ quanto $\tau_{B2}$ (em segundos) e a vantagem
permanecem **essencialmente inalterados** ($A=8{,}3\!-\!9{,}6$, CV $7\%$). Isto é: **reduzir a
frequência de controle em $10\times$ não degrada o ganho**, e não houve instabilidade de
controle amostrado (*sampled-data*) nessa faixa — a margem do trilema (Cap. 3) ainda não é
atingida em $dt=0{,}1$ para $\tau_a=0{,}5$. É um resultado de **deployability**: a coordenação
tolera *hardware* com sensoriamento/comunicação/atuação muito mais lentos, e — em simulação —
permite varrer o eixo de Péclet de forma barata via $dt$.

## 6.6 Caveat: saturação no regime ágil ($\tau_a$ pequeno)

Em $\tau_a=0{,}2$ (UAV muito ágil) o *feedforward* comanda velocidades altas que **saturam o
limite de velocidade** (`VM_MAX_SPEED_XY`, mantido fixo — opção A). Isso introduz um piso de
atuação legítimo: o ajuste de $\tau_{B2}$ fica ruidoso ($R^2<0{,}9$, $\tau_{B2}/\tau_a$ inflado)
e a vantagem aparente é parcialmente artefato. O regime limpo da lei é $\tau_a\ge 0{,}5$; a
saturação é **declarada** como parte do mapa de fases (a alternativa — reescalar a velocidade
máxima com $1/\tau_a$ para uma "agilidade pura" — foi descartada por ser menos realista).

## 6.7 Síntese e o que falta

A caracterização central está medida: **$A\approx 0{,}014\,N^2/\tau_a$**, dt-invariante, com as
duas leis-componente. Em vocabulário de sistemas distribuídos: o baseline relaxa em
$\Theta(N^2)$ rounds, o overlay em tempo ligado à atuação ($\propto\tau_a$, plano em $N$), e o
ganho é exatamente a razão. **A surpresa defensável** é negativa-vira-dado: a coordenação aqui é
limitada por $N^2/\tau_a$ (escala $\times$ atuação) e **não** por latência de informação — o
oposto da intuição "soliton/Péclet" de origem.

Pendências (não bloqueantes): **(i)** robustez à condição inicial genuína (ângulos
não-equidistantes + $t_0$ grande) — adiada por custo; **(ii)** procurar se existe quebra de
Péclet em $\mathrm{Pe}\gg 3$ (para afirmar "sem gargalo de informação até $X$"); **(iii)**
multi-seed só agrega valor fora do regime simétrico determinístico (ver §6.2). O mapa de fases
completo ($N\times$ agilidade $\times$ comunicação) é o fechamento natural, já com a comunicação
degradada do Cap. 7.
