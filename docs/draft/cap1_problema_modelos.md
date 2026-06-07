# Capítulo 1 — Problema, modelos e enquadramento de sistemas distribuídos

> Rascunho (PT). Status: `[base pronta]`. Fonte dos resultados: `experiments/scaling_law/`;
> decisões e histórico na memória do assistente.

## 1.1 Motivação e frase-tese

Manter um espaçamento angular uniforme num **anel de UAVs que cerca um alvo**, sob falha e
recuperação de membros, é — antes de ser um problema de controle — um problema de
**coordenação distribuída auto-estabilizável**. O controlador local que cada agente executa
enfrenta um **trilema fundamental** — estabilidade × velocidade × tamanho do enxame — porque
a informação só se propaga "montada" na própria dinâmica física (difusão de ganho). Esta tese
propõe um **overlay** que computa o alvo de reconfiguração por **hop-count distribuído** e o
injeta por **feedforward, por fora do ganho** do controlador, com um **feedback ciente do
plano (2 graus de liberdade, 2-DOF)**. Esse overlay **quebra o trilema** (estável + rápido +
escalável) e — esta é a contribuição central — **caracteriza, de forma adimensional e válida
para qualquer plataforma, quando e quanto compensa**.

A contribuição é, portanto, do tipo **princípio + caracterização**, não "meu método é mais
rápido". A área é Ciência da Computação / sistemas distribuídos / enxames de UAVs / controle
de formação, e a validação combina simulação em larga escala com voo real / SITL em pequena
escala.

## 1.2 O problema como auto-estabilização distribuída

Formalizamos o cerco em anel como auto-estabilização (cf. Cap. 2, §2.4): o **estado legítimo**
é o espaçamento angular uniforme (cada arco igual a $2\pi/N$); o **evento** é uma falha —
*crash*, *recovery* ou *churn* — que perturba o anel para um estado arbitrário; e cada agente
observa apenas seus **dois vizinhos** (predecessor e sucessor no anel). Um protocolo correto
deve, a partir de qualquer estado pós-evento, reconvergir ao estado legítimo (convergência) e
nele permanecer (clausura). Esse enquadramento substitui o vocabulário de controle (margens,
autovalores) pelo de sistemas distribuídos, que é o que torna a contribuição comparável a
resultados de algoritmos distribuídos.

## 1.3 Os três modelos que governam tudo

1. **Modelo de falha.** Cada agente pode falhar (crash), recuperar-se e re-entrar (recovery),
   possivelmente de forma concorrente (churn). O alvo nunca falha. As falhas são o gatilho de
   toda a coordenação.
2. **Modelo de comunicação.** Variando de síncrono e confiável até com **perda** e **atraso**
   de mensagens e **assincronia** (ordem de disparo intra-tick). A robustez do overlay a esses
   regimes é o objeto do Cap. 7.
3. **Modelo de atuação.** O UAV tem dinâmica física com **velocidade**, **aceleração** e uma
   **constante de tempo de atuação** $\tau_a$. A agilidade da plataforma (eixo $\tau_a$) é uma
   variável primária: define se o regime é limitado por informação ou por atuação (Cap. 6).

## 1.4 Métricas (de CS, não de controle)

Avaliamos por métricas de sistemas distribuídos, não de controle clássico:

- **Tempo de estabilização** — a relaxação do modo lento após um evento ($\tau$ ajustado).
- **Complexidade de mensagens** — payload por agente e total por falha (modelo CONGEST).
- **Cobertura de falhas** — fração dos eventos reconfigurados corretamente.
- **Escalabilidade** — como tempo e mensagens crescem com $N$, contra o limite inferior
  $\Omega(N)$ do diâmetro do anel (Cap. 2, §2.4).

## 1.5 Visão geral da contribuição e organização da tese

O Cap. 3 prova o **trilema** do controlador local (resultado negativo central). O Cap. 4
apresenta o **algoritmo distribuído de redistribuição por hop-count** (o núcleo de
algoritmos distribuídos). O Cap. 5 mostra como **quebrar o trilema** integrando o resultado
discreto por **feedforward 2-DOF** (Option B/B2). O Cap. 6 dá a **caracterização adimensional**
(Péclet/agilidade) de quando o desacoplamento compensa. O Cap. 7 trata da **robustez**
(comunicação degradada e churn) e o Cap. 8 da **validação** em larga escala e da ponte para o
hardware. O Cap. 2 (Trabalhos Relacionados) posiciona a contribuição na interseção de três
tradições até aqui desconexas.

> **Estado.** Capítulos 1–5 têm base/resultados; o Cap. 6 é parcial; os Caps. 7–8 e a
> formalização teórica são o trabalho restante, detalhado no plano de campanha
> (`docs/tese_estrutura.md`, Fases 0–5).
