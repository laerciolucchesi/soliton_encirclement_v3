# RASCUNHO — churn: vantagem da coordenação e o pico do vão

> **RASCUNHO.** Números conferidos e reprodutíveis; a redação ainda não é de tese.
> Documento curado à mão. Os números da parte Π₂′ são gerados por `analyze_pi2.py` →
> [`RESULTADO_PI2.md`](RESULTADO_PI2.md); os da parte G_max por `aggregate_gmax.py` →
> [`LOG_gmax.txt`](LOG_gmax.txt).

**Fonte canônica:** `experiments/scaling_law/churn_sweep_results_c3_churn8_dt05.csv`,
sha256[:16] = `c92a002cb14f319a`. `churn_sweep_results.csv` é byte-idêntico e **não deve ser
citado** — não identifica a campanha. Telemetria por evento: 64 rodadas re-executadas nesta
sessão, em `rerun_runs/`.

---

## 0. Reprodutibilidade — determinística, não apenas numérica

As 64 células foram re-executadas com a mesma configuração e comparadas célula a célula
contra o CSV original em `egap_avg`/`egap_p90`/`egap_max`.

**192 de 192 valores com diferença relativa exatamente `0.0e+00`.** Zero valores meramente
dentro de `rtol = 1e-9`. É reprodução **bit a bit**, não concordância numérica — a distinção
está registrada aqui de propósito.

Sentinelas de integridade, com o portão de reprodutibilidade ativo:

| # | o que verifica | resultado |
|---|---|---|
| S11 | os dois métodos veem o mesmo fluxo de falhas | 32/32 pares idênticos |
| S15 | `alive_count(t)` idêntico entre braços, amostra a amostra | 100% em 64/64 células |
| S16 | `gmax_peak ≥ gmax_pre·(M−1)/M` (valida a extração) | 0 violações nas taxas 6/12/24 |
| S17 | `gap_prev` idêntico entre braços (seletor exógeno) | OK |

S11 e S15 juntas estabelecem, **por dado de execução e não por leitura de código**, que
baseline e B2 enfrentam o mesmo fluxo de falhas e o mesmo padrão de ausência a cada
instante. É o que dá validade ao pareamento por evento.

As 7 violações de S16 no conjunto total (0,26%) estão **todas na taxa 48** — 0/226, 0/404,
0/726 nas taxas 6/12/24 —, com magnitude entre 0,08% e 1,55%. Balde único, e é o balde já
excluído por outro motivo (§4).

---

## 1. Dois vereditos, e são perguntas diferentes

### 1.1 O teste de falseamento pré-registrado — **PASSOU**

Critério escrito **antes** de ver o dado: *"se o coeficiente do método for distinguível de
zero DEPOIS de controlar pelo estado pré-evento, o enunciado do piso geométrico está
errado."*

Desenho: diferenças pareadas por evento (efeitos fixos de evento). Cada evento existe duas
vezes, diferindo só no método e no estado que aquele método produziu; tudo que pertence ao
evento — nó, instante, eventos concorrentes, fase da rodada — sai na diferença.

```
d_pico = alpha + b1·d_gmax_pre + b2·d_egap_pre + erro
```

**alpha (efeito DIRETO) = −0,0016, IC95 [−0,0350; +0,0318], p = 0,922.**

Por taxa, mesmo veredito nas três:

| taxa | pares | alpha | IC95 | p |
|---|---|---|---|---|
| 6 | 113 | +0,0021 | [−0,0231; +0,0274] | 0,848 |
| 12 | 202 | +0,0023 | [−0,0541; +0,0586] | 0,927 |
| 24 | 363 | +0,0027 | [−0,0615; +0,0668] | 0,925 |

> **O enunciado SOBREVIVEU ao teste de falseamento.** Dado o mesmo estado pré-evento, o
> método não tem poder explicativo sobre o pico.

### 1.2 H1 vs H2 — **INCONCLUSIVO**

Pergunta **separada** e secundária: existe efeito *total* do método sobre o pico, mediado
pelo estado que o próximo evento encontra?

| coeficiente | valor | IC95 | p |
|---|---|---|---|
| TOTAL (sem controlar) | +0,0247 | [−0,0016; +0,0510] | **0,064** |
| DIRETO (controlado) | −0,0016 | [−0,0350; +0,0318] | 0,922 |
| MEDIADO = TOTAL − DIRETO | +0,0263 | — | — |

O padrão — total pequeno e positivo, direto exatamente zero — é o que H2 prevê, **mas o
efeito total não se separa de zero**. Não é possível afirmar H2. Rótulo recusado.

**Tendência registrada como tendência, não como alegação:** `alpha_total` cresce
monotonicamente com o churn — **0,0152 → 0,0164 → 0,0322** nas taxas 6/12/24 —, que é a
direção que H2 prevê. Três pontos, IC largos: sugestivo, **não afirmável**.

---

## 1.3 Regra de apresentação adotada nesta sessão

> **Nenhuma diferença pareada entra em documento sem MEDIANA, IQR, FRAÇÃO DE SINAL e
> MÉDIA-SEM-DECIL-SUPERIOR ao lado da média. As quatro juntas, sempre.**
>
> A regra nasceu de um erro concreto: o `d_arco = +0,178 m` foi apresentado como "o número
> mais comunicável do dia" antes de alguém olhar a distribuição. A mediana era +0,052 m,
> 47% dos eventos iam na direção oposta, e remover o decil superior invertia o sinal.

---

## 2. O pico: DUAS análises independentes, MESMO nulo

Este é o resultado, e ele é uma **confirmação**, não uma ausência.

| análise | desfecho | resultado |
|---|---|---|
| efeitos fixos de evento (§1.1) | `alpha` direto | −0,0016 [−0,0350; +0,0318], p = 0,922 → **nulo** |
| diferença pareada bruta (§2.1) | `d_arco` mediano | +0,052 m; 52,5% / 47,5% de sinal → **nulo** |

**Duas análises independentes — efeitos fixos de evento e diferença pareada bruta —
concordam que o pico não distingue os métodos.** É exatamente o que o invariante geométrico
do §3 prevê: `E[pico] = 2(M−1)/M` não depende de protocolo.

Se os 18 cm tivessem sobrevivido ao escrutínio, **isso seria um problema**: um efeito do
método sobre o pico contradiria o invariante. O nulo é a previsão da teoria confirmada por
dois caminhos.

### 2.0 Recalibração contra o nulo sintético — os "efeitos marginais" somem

O nulo sintético (§6) constrói um mundo onde o método **não** afeta o pico dado o anel, mas
**afeta o anel** (dispersão diferente entre braços, calibrada no dado real). Ele mede,
portanto, quanto de efeito surge **só da geometria**. O `p` honesto é o percentil do valor
real na distribuição nula:

| coeficiente | real | p nominal | mediana da nula | percentil do real | pipeline rejeita no nulo |
|---|---|---|---|---|---|
| `alpha` DIRETO | −0,0016 | 0,922 | −0,0061 | **0,585** | 8,0% |
| `alpha` TOTAL | +0,0247 | 0,064 | +0,0342 | **0,205** | 79,0% |
| `d_arco` [m] | +0,178 | 0,040 | +0,220 | **0,300** | 79,5% |
| `b3` | +1,110 | <0,001 | +0,984 | **0,720** | 90,0% |

Três leituras:

1. **Os "efeitos marginais" não são marginais — são inexistentes.** `alpha_total` e `d_arco`
   ficam **abaixo da mediana** do que a geometria pura produz. O efeito total observado é
   *menor* que o que a diferença de dispersão entre os anéis já implica mecanicamente.
2. **O `p` nominal estava inflado ~16×** para esses coeficientes: o pipeline rejeita em
   79–80% das réplicas de um mundo sem efeito. `p = 0,040` não significava nada.
3. **O nulo de `alpha_direto` é mais forte, não mais fraco.** Para esse coeficiente o
   pipeline rejeita em apenas 8,0% das réplicas nulas — perto do nominal — e mesmo assim
   não rejeitou no dado real, com percentil 0,585 (centro da nula).

*Limite desta calibração:* o nulo foi calibrado por `gmax_pre` mediano (1,595/1,450
sintético vs 1,502/1,406 real), então a diferença de dispersão do nulo é um pouco **maior**
que a real — o que empurra a nula para cima. Uma calibração mais apertada aproximaria a nula
do valor real, sem mudar a conclusão de direção.

---

## 2. O vão absoluto — mesmo efeito, duas normalizações, ambas marginais

| normalização | efeito | IC95 | p |
|---|---|---|---|
| metros (arco no raio de 20 m) | **+0,178 m** | [+0,008; +0,348] | **0,040** |
| adimensional (`alpha_total`, por `2π/M`) | +0,0247 | [−0,0016; +0,0510] | **0,064** |

São **o mesmo efeito** em duas normalizações. **Os dois são marginais.** Reportar só o de
`p` menor seria seleção de normalização; os dois aparecem juntos aqui e na figura
[`fig_vao_absoluto.png`](fig_vao_absoluto.png).

### 2.1 A média não é o evento típico — quatro qualificações obrigatórias

Ao desenhar a figura apareceu o que a tabela escondia. O `+0,178 m` é uma **média**, e a
distribuição pareada não a sustenta como efeito típico:

| estatística | valor |
|---|---|
| média pareada | **+0,178 m** |
| **mediana pareada** | **+0,052 m** |
| IQR de `d_arco` | [−0,461; +0,694] m |
| fração de eventos com baseline pior | **52,5%** (moeda = 50%) |
| **média removendo o decil superior** | **−0,136 m** — *inverte de sinal* |
| mediana por rodada (n = 24, sem pseudo-replicação) | +0,080 m, 16/24 rodadas positivas, p = 0,055 |
| medianas **marginais** (não pareadas) | 11,19 m baseline vs 11,30 m B2 — **direção oposta** |

O decil superior sozinho contribui +0,301 dos +0,178 da média. Sem ele, o efeito muda de
sinal. **O efeito existe na cauda, não no evento típico**: em ~47% dos eventos o overlay abre
a brecha maior.

Redação defensável: *"num anel de raio 20 m, sob churn, a brecha do baseline é em média
~18 cm mais larga que a do overlay no mesmo evento — mas a mediana é de 5 cm, 47% dos
eventos vão na direção oposta, e o efeito médio desaparece ao remover o decil superior. É um
efeito de cauda, marginal (p = 0,040 em metros; p = 0,064 normalizado)."*

Redação **não** defensável, e que era a minha primeira versão: *"a brecha do baseline abre
18 cm a mais"*, sem nada ao lado.

E, com a recalibração de §2.0, nem a versão qualificada se sustenta como efeito: o valor real
fica no percentil 0,300 da nula geométrica. **O `d_arco` é, ele próprio, uma segunda medida
do nulo do pico** — é a leitura correta, e é o que §2 afirma.

---

## 2b. A segunda metade: a coordenação compra a INCLINAÇÃO, não o piso

A decomposição da brecha só fica completa com os dois lados:

| quantidade | previsão da teoria | medido |
|---|---|---|
| **pico** de `G_max` | invariante geométrico ⇒ **nulo** | nulo em duas análises (§2), e dentro da nula sintética |
| **área/tempo em brecha** | coordenação compra a inclinação ⇒ **efeito** | 1,28–1,42× na campanha de falha única; sob churn, abaixo |

Um nulo onde a teoria prevê nulo **e** um efeito onde a teoria prevê efeito. As duas juntas
são a afirmação inteira, e o par é mais difícil de derrubar que um efeito solto.

### 2b.1 `t_close` NÃO é identificável sob churn — e isso é do regime

| limiar | pares | censurados | fração |
|---|---|---|---|
| `G_max` > 1,25 | 678 | 676 | **99,7%** |
| `G_max` > 1,5 | 678 | 645 | **95,1%** |

Censura = a brecha não fechou dentro da janela adaptativa. Sob churn o intervalo entre
eventos é **menor que o tempo de fechar**, então `t_close` completo não cabe na janela. É
propriedade do **regime**, não defeito da medida — a campanha de falha única mede `t_close`
porque lá há um evento só. **Declarado não identificável**, não estimado.

### 2b.2 A ÁREA de brecha dentro da janela — essa é medível

A área não exige que a brecha feche, e é comparável entre braços porque `W_e` é idêntico nos
dois (S11). Limiar `G_max > 1,25`, pareado por evento, taxas 6/12/24 (n = 678):

| desfecho | média | IC95 | p | mediana | frac > 0 | média s/ decil sup. |
|---|---|---|---|---|---|---|
| área [adim·s] | +0,0352 | [+0,0173; +0,0531] | 0,0005 | +0,0095 | 0,544 | **−0,0104** |
| área [m·s] | +0,2227 | [+0,1129; +0,3325] | 0,0003 | +0,0524 | 0,544 | **−0,0624** |
| **fração da janela em brecha** | **+0,0440** | [+0,0346; +0,0534] | <0,0001 | +0,0000 | 0,205 | **+0,0091** |

Por taxa (área): +0,0230 (p = 0,051) · +0,0242 (p = 0,008) · +0,0451 (p = 0,022) — mesma
direção nas três.

**Das três, só a fração da janela em brecha sobrevive à regra das quatro estatísticas**: é a
única cuja média **não inverte de sinal** ao remover o decil superior. As medianas marginais
são 1,000 (baseline) vs 0,867 (B2) — o baseline passa a janela inteira em brecha na mediana,
o overlay não. A mediana das diferenças é 0 porque em ~80% dos eventos **ambos** ficam a
janela toda acima do limiar; o efeito vive nos ~20% em que o overlay sai antes.

**Ressalva que não pode faltar:** esses `p` **não** foram recalibrados contra nulo sintético.
O nulo de §6 é geometria estática, sem dinâmica, e não gera área nem tempo. Enquanto não
houver um nulo dinâmico, esses `p` são nominais — e o pipeline mostrou inflar `p` nominal em
~16× nos desfechos em que a calibração foi possível. **Tratar como sugestivo.**

---

## 3. O pico é uma MÉDIA EXATA, não um piso

Teorema, sem hipótese sobre a configuração. Anel com `M` vivos, vãos `g_1..g_M`,
`Σ g_k = 2π`. O agente `i` é ladeado por `g_{i−1}` e `g_i`; morrendo, forma `g_{i−1}+g_i`.
Somando sobre todos os agentes, cada vão é contado duas vezes:

```
Σ_i (g_{i−1} + g_i) = 2·Σ_k g_k = 4π
```

Com a vítima uniforme entre os vivos — que é o caso (`protocol_agent.py:918-920`, um sorteio
independente por agente à mesma taxa):

```
E[vão de fusão] = 4π/M   →   E[pico] = 2(M−1)/M   EXATO, para QUALQUER configuração
```

**Não é cota. É a média, e é exata.** A dispersão observada é variância em torno de uma média
exata, não um piso violado; o anel uniforme é a configuração de variância zero. No escopo
declarado em §4 (taxas 6/12/24, n = 1356 eventos): **29,6% dos picos abaixo de `2(M−1)/M`,
24,6% nele, 45,8% acima**. Incluindo a taxa 48, que §4 exclui da análise por evento
(n = 2690): 28,3% / 18,2% / 53,5%.

A dispersão **não é homogênea**, e o trio agregado esconde dois gradientes:

| taxa | n | abaixo | no piso | acima |
|---|---|---|---|---|
| 6 | 226 | 23,5% | 48,2% | 28,3% |
| 12 | 404 | 32,9% | 23,0% | 44,1% |
| 24 | 726 | 29,8% | 18,0% | 52,2% |
| 48 | 1334 | 26,9% | 11,8% | 61,3% |

Por método, no escopo 6/12/24: o overlay fica **no piso** em 31,3% dos eventos contra 17,8%
do baseline (n = 678 cada) — a variância em torno da média exata é menor com o anel mais
uniforme, que é o que o teorema prevê.

**Verificação empírica (D21), sem modelo.** À medida que o recorte se aproxima do anel
uniforme, a razão converge:

| recorte por `gmax_pre` | n | `gap_rad` no pico | `2·(2π/M)` | razão | `gmax_peak` | `2(M−1)/M` |
|---|---|---|---|---|---|---|
| todos | 2690 | 0,6139 | 0,5984 | 1,0120 | 1,9562 | 1,9048 |
| ≤ P50 | 1345 | 0,5702 | 0,5712 | 1,0071 | 1,9316 | 1,9091 |
| ≤ P25 | 673 | 0,5486 | 0,5464 | 1,0047 | 1,9262 | 1,9130 |
| ≤ P10 | 269 | 0,5288 | 0,5236 | **1,0009** | 1,9190 | 1,9167 |

O desvio **encolhe monotonicamente com a uniformidade do pré-estado**. Viés fixo de
instrumento não faria isso — seria constante em todos os recortes. E bate com a calibração
independente: campanha de falha única (anel uniforme por construção) +0,037%; recorte P10
(quase uniforme) +0,09%; conjunto todo +1,2%. **O instrumento está limpo; o desvio é do
regime.**

**Redação final, e o assunto está encerrado:**

> A razão converge para **1,0009** no decil mais uniforme, contra **1,9167** previsto —
> consistente com a calibração independente de **+0,037%** na campanha de anel uniforme. O
> resíduo agregado de **+1,2%** é atribuível ao registro de `G_max` como máximo sobre
> **todos** os vãos, e não sobre a fusão; fechá-lo exatamente exigiria instrumentação por
> agente, não realizada.

O teorema está **provado matematicamente** — três linhas, sem dado. A confirmação empírica é
a **convergência**, e ela já está feita. Instrumentar para ganhar 1,2% não vale o tempo.

Candidato investigado e **descartado**, registrado para não ser re-proposto: a hipótese de
que a referência usasse um `M` defasado (`alive_count` conta quem já morreu durante os
~0,30 s de latência de detecção). A defasagem **existe** e cresce com a taxa (2,8% / 5,1% /
9,1% / 15,7% das amostras), mas recalcular com o `M` verdadeiro reconstruído de `events.csv`
dá **1,0628 [1,0272; 1,0984]** — não fecha, e não melhora.

### 3.1 Correções aplicadas na documentação de campanha

Quatro formulações de classe (b) — *"piso"*, *"cota"*, *"nenhum protocolo consegue menor"* —
corrigidas em `docs/experiments/`: `BREACH_WINDOW.md:9`, `:50-53`, `CHURN_PAIRED.md:405-409`
e `:464`. A demonstração do teorema entrou como `BREACH_WINDOW.md §1.1`. As de classe (a)
ficaram, ganhando a qualificação "no anel uniforme". `CHURN_PAIRED.md:409` continha uma
**instrução** para levar a formulação falsa ao Cap. 7 — era o item mais urgente e foi
neutralizado.

Nenhuma ocorrência em `docs/thesis/`: a formulação errada **não chegou aos capítulos**.

Registro: a documentação **já sabia** que sob churn o pico sobe acima de 1,92 (2,11 → 3,49,
`BREACH_WINDOW.md:153-155`); o que nunca se considerou foi que pudesse ficar **abaixo**.
Ultrapassado por cima e furado por baixo — não é piso em direção nenhuma.

---

## 4. ESCOPO DECLARADO — e são DOIS escopos diferentes

> **Atenção:** as duas afirmações desta análise cobrem faixas diferentes de Π₂′. Declarado
> nos dois lugares para não virar contradição aparente.

| análise | taxas | Π₂′ | por quê |
|---|---|---|---|
| **regime permanente** (`egap_*` × Π₂′) | 6, 12, 24, 48 | **0,80 – 6,40** | métrica é média temporal; não precisa isolar evento |
| **por evento** (G_max, vão absoluto) | 6, 12, 24 | **0,80 – 3,20** | taxa 48 **declarada não identificável**: mediana de `W_e` no piso de 0,6 s e 48% dos eventos truncados — o pico de um evento não é separável do próximo |

Constante em toda a campanha, nas duas análises:

- N = 24 agentes · `tau_xy` = 1,0 s · `T_off` = 8,0 s (recuperação finita: os agentes voltam)
- dt (`CONTROL_PERIOD`) = 0,05 s · `K_E_TAU` = 250/N
- canal ideal: `COMMUNICATION_FAILURE_RATE=0`, `COMMUNICATION_DELAY=0`
- alvo estacionário · inicialização equidistante, sem dispersão de raio
- **regime NÃO saturado do atuador: `sat_frac` = 0 em 100% das 64 células** — declaração de
  escopo, não falha
- n = 8 sementes por célula, pareadas (mesmo `EXPERIMENT_SEED` ⇒ mesmo fluxo de falhas)
- métrica de regime: `E_gap` = **RMS espacial** do erro relativo de vão, normalizado pelo
  número de agentes **VIVOS** — mede qualidade de redistribuição, não cobertura absoluta
- janela de regime: t ≥ 20 s até 155 s

---

## 5. O que esta análise NÃO mostra

- **Não mostra tempo de assentamento.** `egap_avg` é erro de regime permanente. Nenhum
  `t_settle` foi medido nesta campanha.
- **Não é comparável com a campanha de falha única.** Estímulo, janela e métrica diferem.
- **Não cobre a taxa 48 na análise por evento** (§4).
- **Não mede cobertura absoluta** no regime permanente: `E_gap` e `G_max` são normalizados
  pelo número de vivos.
- **Não separa taxa de duração**: `T_off` é constante, logo Π₂′ e `rate_total` são
  proporcionais.
- **Não varre N nem `tau_xy`**: uma única coluna do espaço de projeto.
- **Não mede custo**: `effort_mean_v2` e `fairness_p95` ficaram fora; `analyze_churn_paired.py`
  já reporta o custo (2,41× mediano, 32/32 pares).
- **Sem correção para múltiplas comparações.**
- **Piso de resolução do teste de regime:** com n = 8, o menor p bilateral exato possível é
  0,007812. Os quatro p de `egap_avg` **são** esse valor — o teste por taxa não distingue
  1,31 de 1,14, e a tendência se apoia no teste da FASE 2b e nos IQR disjuntos, nunca nos p.

---

## 6. Descartado nesta sessão, com registro

| o que | por que caiu |
|---|---|
| `b3` (interação método × estado pré) | **Nulo sintético**: num mundo sem efeito de método por construção, o mesmo pipeline produz `b3` com IC excluindo zero em **83,5%** (aditivo) e **93,5%** (log) das réplicas. O `b3` real (1,110 / 1,610) fica no **percentil 69 / 62** da distribuição nula. Artefato de especificação. **Não entra como número.** |
| `b1 < 0` "sete vezes a predição composicional" | Era o coeficiente **não centrado** — a inclinação em `pre_bar = 0`, ponto que nunca ocorre. Centrado na média: −0,119 [−0,437; +0,198], que **contém** a predição composicional −2/M = −0,095. Não há magnitude em excesso. O nulo sintético reproduz o mesmo sinal invertido. |
| "censura pelo piso geométrico" | Argumento retirado: comparava correlações **dentro de faixas estreitas da variável dependente** — condicionar em y atenua ou inverte qualquer correlação, haja censura ou não. E era incoerente com ~30% dos picos abaixo do piso. |
| janela de pico fixa | Substituída por `W_e = clip(t_próximo_evento − t_f, 0,6, 1,5)` s. O diagnóstico mostrou **contaminação**, não truncamento: a fração de picos tardios *cresce* com W e o lag p99 *acompanha* W (razão p99/W ≈ 0,98–1,00) em vez de estacionar. Entre eventos isolados a fração tardia é 0,0000 nas três janelas. |
| `M` defasado como causa do resíduo do teorema | Testado com o `M` verdadeiro de `events.csv`: não fecha (§3). |

---

## 7. Pendências

1. **Nulo sintético DINÂMICO** para os desfechos de área/tempo (§2b.2). O nulo atual é
   geometria estática e não gera duração; enquanto não existir, os `p` da área são nominais,
   num pipeline que inflou `p` nominal em ~16× onde a calibração foi possível.
2. **Dívida técnica do homônimo `egap_avg`** — sete itens (D1–D7) em
   [`EGAP_HOMONIMO.md`](EGAP_HOMONIMO.md), a pagar antes do Capítulo 6. Nada corrigido.
3. **Taxa 48 na análise por evento** — exigiria `T_off` menor ou taxa menor para separar
   eventos, ou uma métrica que não precise isolar o evento.
4. *(baixa prioridade, 10 min, só depois de escrever)* restringir aos eventos em que a fusão
   domina estritamente e recomputar a razão do teorema. Não muda a conclusão de §3.
5. *(encerrado, não reabrir)* instrumentar `g_pred + g_succ`. Ganharia 1,2% num teorema já
   provado; não vale o tempo de tese.
