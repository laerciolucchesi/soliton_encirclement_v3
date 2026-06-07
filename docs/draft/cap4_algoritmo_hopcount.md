# Capítulo 4 — O algoritmo distribuído de redistribuição por hop-count

> Rascunho (PT). Status: `[implementado; análise a formalizar]`. Implementação:
> `dual_pulse_layer.py`; testes em `tests/test_dual_pulse.py`.

Este é o **núcleo de Ciência da Computação / algoritmos distribuídos** da tese e merece
capítulo próprio. Ele responde: dado um evento de topologia (saída/entrada de um agente),
como cada nó descobre — só com informação local — qual deslocamento angular deve aplicar para
restaurar o espaçamento uniforme no novo tamanho do anel?

## 4.1 Descoberta de topologia por pulsos contra-propagantes

Quando ocorre um evento, o **originador canônico** (o predecessor do nó que saiu/voltou)
injeta **dois pulsos contra-propagantes** (sentido horário, CW, e anti-horário, CCW),
marcados com um `event_id`, o `event_type` (SAÍDA/ENTRADA), um `hop_count` e — para ENTRADA —
o `recovered_id`. Cada receptor registra o hop-count de cada direção e o reencaminha. Quando
um nó já viu **ambas** as direções, ele conhece sua **posição relativa** ao nó que
saiu/voltou e o **tamanho do anel** $N$ — tudo **sem conhecimento global**. O encaminhamento
é *stateless* com **cache refratário** (filtra duplicatas), na linhagem do *flooding* amnésico
(Cap. 2, §2.5).

## 4.2 Derivação do alvo de deslocamento $\delta_D$

A partir dos hop-counts das duas direções, cada nó deriva o deslocamento angular alvo
$\delta_D$ que o levaria à posição equidistante no novo anel. As fórmulas distinguem **SAÍDA**
de **ENTRADA** e o papel de **originador** vs **receptor** (o nó recuperado, na ENTRADA, opera
em modo *passthrough*, pois já está na posição de equilíbrio). O originador detecta o retorno
do próprio pulso (traversal completo do anel) para ler o novo $N$ e aplicar sua fórmula
específica.

## 4.3 Complexidade e modelo CONGEST

O algoritmo atinge os limites desejados:

- **Tempo:** $O(N)$ rounds — **ótimo** pelo limite inferior $\Omega(N)$ do diâmetro (Cap. 2).
- **Mensagens:** $O(N)$ por falha.
- **Payload:** $O(1)$ por nó (modelo CONGEST) — cada pulso carrega apenas os campos acima.

O custo de banda foi medido por um contador de pulsos (`diag_messages.py`): a disseminação é
$O(N)$, com $\approx 3.9$ payloads/agente em $N \le 50$ (= 2 direções × `BROADCAST_REPEATS=2`)
e payload por-agente $O(1)$. Em $N=75/100$ houve inflação transitória (8,6/6,0 payloads) por
pulsos espúrios de *flapping* de vizinho no transiente (efeito histerese-vs-gap; ver Cap. 7 e
o item de higiene da Fase 0).

## 4.4 Corretude e modelo de falhas do próprio protocolo

A corretude exige tratar o modelo de falhas **do protocolo em si**: a coordenação canônica
(só o originador canônico injeta), o cache refratário (evita re-injeção) e os casos de borda
sob churn (saída+entrada simultâneas, mascaramento de `alive_count`, inferência de $N$, e
ENTRADA-com-originador-falho), com os respectivos consertos detalhados no Cap. 7.

## 4.5 Limite de configuração descoberto e corrigido (TTL)

Um achado de validação relevante: o `DUAL_PULSE_TTL_HOPS=50` era pequeno demais para
$N \gtrsim 51$ (um receptor precisa de pulsos de até $N-1$ hops; o originador, de $N$ hops
para o retorno), truncando o $\delta_D$ e degradando o feedforward (Cap. 5). O diagnóstico
confirmou (cobertura de shift caindo 96% → 36% → 1% e `max_hop` travado em 50); tornando o TTL
*env-overridable* com $\text{TTL}=3N$, a cobertura volta a ~100%. Era **limite de
configuração, não do mecanismo**.

> **A formalizar (Fase 4):** prova de corretude + complexidade do hop-count (O(N) rounds,
> O(N) mensagens, diâmetro-ótimo) escrita formalmente; e a derivação fechada de $\delta_D$ nos
> quatro casos (SAÍDA/ENTRADA × originador/receptor).
