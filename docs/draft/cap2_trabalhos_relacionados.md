# Capítulo 2 — Trabalhos Relacionados

> Rascunho (PT) para a tese/proposta. Versão completa em inglês e tabela de posicionamento:
> `docs/related_work.md`. Citações em autor–ano; chaves = nomes dos PDFs em
> `11 Doc References/` e subpastas. Afirmações verificadas em
> `docs/pesquisa_literatura_encirclement.md` (5 rodadas + varredura de primazia).

O problema desta tese — manter espaçamento angular uniforme num anel de agentes que cercam
um alvo, tolerando falha e recuperação de membros — situa-se na interseção de três tradições
que permaneceram, até aqui, largamente desconexas: (i) a literatura **de teoria de controle**
sobre cerco, circumnavegação e formação circular, que resolveu a geometria de equilíbrio mas
a analisa por estabilidade de Lyapunov, não por complexidade distribuída; (ii) a literatura
**de sistemas distribuídos** sobre auto-estabilização, localidade e disseminação de
informação, que fornece o vocabulário correto (rounds, complexidade de mensagens, limites
inferiores de diâmetro, cobertura de falhas) mas nunca foi aplicada ao cerco físico; e (iii)
a literatura **de escala/espectral**, que caracteriza como o tempo de coordenação cresce com
a rede. Este capítulo percorre cada tradição apoiando-se na biblioteca de referências montada
para esta tese, identifica os vizinhos mais próximos e argumenta que a interseção ocupada por
esta tese está, ao melhor do nosso conhecimento, vazia.

## 2.1 Cerco, circumnavegação e formação circular (controle)

É a tradição mais desenvolvida. A observação fundadora de que o espaçamento equiangular pode
**emergir** de uma regra local, sem líder, vem da *cyclic pursuit*: Marshall, Broucke e
Francis (2004; 2006) mostram que um anel de veículos, cada um sentindo apenas o sucessor,
converge para polígonos regulares generalizados. Variantes enriquecem a geometria e a
convergência — *cyclic pursuit* generalizada (Mukherjee e Ghose, 2015), hierárquica, que
acelera a convergência num resultado explicitamente de **escalabilidade** (Smith et al.,
2005), desviada, produzindo polígonos rotativos (Mallik e Sinha, 2016; Rezaee e Abdollahi,
2015), movimento circular coletivo em torno de um *beacon* virtual (Ceccarelli et al., 2008),
e formação circular com controle de espaçamento (Fujioka e Hayashi, 2024; Park et al., 2024;
Zheng et al., 2023, esta última precisando só do vizinho dianteiro e tolerando alcance/atraso
limitados). Todas são **leis de controle contínuas** provadas por argumentos de estabilidade,
sem noção de complexidade de rounds ou mensagens.

Um grande conjunto projeta leis distribuídas que levam os agentes a ângulos especificados em
torno do alvo: cerco de robôs anônimos com espaçamento arbitrário (Yao et al., 2017), cerco
com espaçamento arbitrário (Sen e Sahoo, 2021; Song et al., 2019), cerco dinâmico de agentes
anônimos (Huang et al., 2024), cerco rotativo de múltiplos alvos com restrições de entrada
não-convexas (Zhang et al., 2020) e cerco denso num "tubo virtual" anular (Gao et al., 2022).
A formação circular **não-uniforme** por robôs *oblivious*/anônimos na tradição
Suzuki–Yamashita (Défago e Souissi, 2008; Wang, Xie e Cao, 2013) é o casamento mais próximo
em *problema*, mas pressupõe agentes sem comunicação e sem modelo de falhas — relevante ao
nosso espaçamento não-uniforme protegido (`PROTECTION_ANGLE_DEG`).

Quando a posição do alvo é desconhecida, os agentes precisam **localizar enquanto orbitam**.
Essa linha acopla um estimador ao controlador: circumnavegação *bearing-only* de alvo móvel
(Yu et al., 2019; Ji et al., 2025), o par bearing-only de tracking e circumnavegação com
estimação adaptativa PI + Kalman (Zhou et al., 2024) e sua extensão *fixed-time* com
espaçamento uniforme (Zhou et al., 2026), circumnavegação *range-only* de agentes
não-holonômicos (Wang et al., 2024) e de grupos de alvos não-cooperativos em ambiente sem GPS
(Huang et al., 2025), com campo vetorial robusto a vento (Muslimov, 2023) e — notável para a
nossa história de falhas — circumnavegação *bearing-only* **livre de comunicação** robusta a
perda de pacotes e *jamming* (Sui e Deghat, 2023). O arcabouço *estimator-coupled* é detalhado
em §2.3.

Trabalhos recentes acoplam "event-triggered" ao cerco: controle de cerco *event-triggered*
(Babazadeh et al., 2025), cerco multi-alvo *event-based* resiliente a ataques DoS (Zhang et
al., 2025) e cerco de alvo móvel *event-triggered* por formação de UAVs (Jia et al., 2024).
**Aqui é crucial uma distinção que esta tese adota:** nesses trabalhos "event-triggered"
denota **parcimônia de atualização de controle/comunicação** (o controlador recomputa quando
um erro de medição cruza um limiar) — **e não** a *disseminação event-triggered de um evento
de topologia* que define esta tese. O trabalho de Xu et al. (2020), que compartilha exatamente
o nosso vocabulário (*self-trigger*, formação circular, vizinhos de anel $i^{+}/i^{-}$,
espaçamento angular $\alpha^{*}$) mas nenhuma da nossa maquinaria de sistemas distribuídos, é
a ilustração mais limpa de **event-triggered control $\neq$ event-triggered dissemination**.

Uma vertente paralela e metodologicamente distinta usa aprendizado por reforço para cerco de
alvos mais rápidos ou evasivos (Ma et al., 2019; Li et al., 2024; Qu et al., 2025, 2026; Mu
et al., 2026), perseguição cooperativa formando uma armadilha angular (Fang et al., 2020) e
defesa de perímetro escalável por GNN (Lee et al., 2023; Guerrero-Bonilla et al., 2021).
Otimizam empiricamente e, como a linha de controle, não oferecem garantias de complexidade de
rounds/mensagens nem de auto-estabilização. A área é consolidada em revisões usadas aqui como
âncoras: *survey* de formação circular (Litimein et al., 2021), o *survey* canônico de
formação por capacidade de sensoriamento (Oh, Park e Ahn, 2015), *survey* de *containment*
(Thummalapeta e Tsai, 2023) e *survey* de *tracking* por UAV (Wu et al., 2025).

> **Conclusão do eixo.** Em ~60 papers da coleção local de cerco, todo método resolve a
> geometria equiangular, mas nenhum fornece caracterização de complexidade distribuída nem
> garantia de auto-estabilização no sentido de Dijkstra. A própria biblioteca da tese
> corrobora o gap.

## 2.2 Movimento coletivo, osciladores acoplados e sincronização

Uma tradição complementar modela o movimento circular por **osciladores de fase acoplados**.
Sepulchre, Paley e Leonard (2007) estabilizam o movimento coletivo planar e identificam o
*order parameter* de Kuramoto $\rho = |\tfrac{1}{N}\sum_k e^{i\theta_k}|$ como a medida
natural de quão uniformemente os agentes se distribuem; seu projeto de duas escalas de tempo
*desacopla* a coordenação de fase da atuação de espaçamento — mas por perturbação singular
dentro de **um** controlador contínuo, não por um algoritmo distribuído discreto — e os
autores declaram explicitamente a suposição all-to-all como irrealista, deixando topologias
esparsas/locais em aberto (alavanca direta para esta tese). Generalizações geométricas levam a
sincronização a grupos de Lie e variedades (Sarlette et al., 2009; Markdahl et al., 2021;
Butcher, 2025), e o *flocking* fornece o arcabouço fundador de movimento distribuído
(Olfati-Saber, 2006). De especial relevância, Liu et al. (2023) conduzem o cerco com um
movimento desejado **por oscilador acoplado** combinado a localização relativa recursiva — o
elo existente mais próximo entre a visão Kuramoto e o cerco em anel, ainda assim contínuo e
sem disseminação disparada por falha.

## 2.3 Localização e estimação do alvo para cerco

Como o alvo é frequentemente não-cooperativo e sem GPS, uma sub-literatura substancial funde
**estimação distribuída** ao controlador de cerco — o paradigma *estimator-coupled* que esta
tese contrasta. A circumnavegação seminal por distância sob *slow drift* (Shames et al., 2012)
e sua contraparte por *bearing* (Deghat et al., 2014) estabelecem garantias exponenciais de
localizar-e-orbitar. A linha de Franchi oferece localização mútua e cerco distribuídos
validados experimentalmente, estendidos a 2D/3D com prevenção de colisão garantida (Franchi et
al., 2010, 2015). A linha do KTH trata a circumnavegação coletiva de alvo desconhecido/em
deriva com provas de convergência e resiliência a entrada/saída de agentes (Boccia et al.,
2017; Swartling et al., 2014). Completam a vertente estimação *bearing-only* com cota de erro
explícita (Parayil e George, 2020), aplicação a *algal bloom* (Fonseca et al., 2019), estimação
TDOA de escala única para formações de UAV (Doostmohammadian et al., 2022) e estimação neural
do centro de múltiplos alvos sem GPS (Liu et al., 2024). A resiliência que oferecem (p.ex. a
add/removal em Boccia et al., 2017) vem de re-estimação contínua, não de um protocolo de
disseminação event-triggered com complexidade de rounds limitada.

Um bloco intimamente relacionado é o **consenso de média dinâmica** (DAC): o tutorial canônico
— cuja motivação cita explicitamente o custo $O(N^2)$ e o ponto único de falha da estimação
centralizada (Kia et al., 2019) — com DAC discreto robusto (Montijano et al., 2014), DAC
não-linear (Nosrati et al., 2012), DAC robusto a entrada/saída (Gudeta et al., 2022) e
estimadores DAC de taxa ótima (Van Scoy et al., 2015). O DAC é a contraparte contínua da
disseminação discreta desta tese, e seus limites de taxa (§2.5) enquadram nosso argumento de
escala.

## 2.4 Auto-estabilização e localidade em computação distribuída

O enquadramento em sistemas distribuídos repousa sobre dois corpos teóricos maduros mas, até
aqui, separados. A **auto-estabilização**, introduzida por Dijkstra (1974) e desenvolvida na
monografia de Dolev (2000), é a propriedade de um sistema iniciado em estado *arbitrário*
convergir e permanecer em estado legítimo (convergência + clausura) — a abstração correta para
coordenação em anel tolerante a crash/recovery e perda de mensagens. É historicamente notável
que o próprio EWD386 de Dijkstra tenha posto o problema geométrico de "distribuir $N$ pontos
igualmente num círculo", apontado por Ghosh Dastidar e Herman (2009) como o ancestral
geométrico dos algoritmos comportamentais de coordenação em anel — uma linhagem conceitual
direta da auto-estabilização ao espaçamento equiangular. A **teoria de localidade** limita o
que é computável a partir de informação de raio limitado: Naor e Stockmeyer (1995) formalizam
*locally checkable labelings* (LCL) e Linial (1992) prova os limites inferiores canônicos de
localidade (p.ex. $\Omega(\log^{*} n)$ rounds para coloração de anel no modelo LOCAL). Com o
fato textbook de que a informação leva $\Theta(\text{diâmetro})$ rounds para cruzar a rede —
$\Theta(N)$ no anel — isso fornece o limite inferior $\Omega(N)$ contra o qual nosso protocolo
é medido (o tratamento formal de $\Omega(\text{diâmetro})$ está em Peleg, 2000). Mais
diretamente, uma pequena literatura estuda **espaçamento uniforme auto-estabilizável num anel
de processos**: *deployment* uniforme de agentes móveis em anel dinâmico (Shibata et al., 2020,
2022), separação uniforme de tokens circulantes (Ghosh Dastidar e Herman, 2009) e balanceamento
de carga em anel (Gehrke, Plaxton e Rajaraman, 1999). São os análogos *computacionais* mais
próximos da redistribuição equiangular, mas puramente discretos/grafo-teóricos, sem atuação
contínua, sem disseminação disparada por crash/recovery e sem imersão espacial em torno de um
alvo móvel.

## 2.5 Disseminação de informação, escala e tolerância a falhas

Como nosso *overlay* é essencialmente um protocolo de disseminação, os *baselines* vêm dessa
literatura. Algoritmos *gossip*/epidêmicos (Demers et al., 1987) espalham uma atualização em
$O(\log n)$; Karp et al. (2000) dão a análise canônica de $O(\log n)$ rounds e
$O(n\log\log n)$ mensagens do *rumor spreading* randomizado, depois mostrado assintoticamente
ótimo (Doerr e Fouz, 2011). Um contraste definidor: o *gossip* randomizado troca a garantia de
latência proporcional ao diâmetro por uma cota de alta probabilidade, ao passo que nosso
mecanismo é determinístico e disparado por evento. *Flooding*/wave determinísticos terminam em
tempo linear no diâmetro; Hussak e Trehan (2023) mostram que mesmo o *flooding* **sem estado**
(amnésico) termina em $\Theta(\text{diâmetro})$ sem manter árvore — um precedente teórico do
encaminhamento *stateless* (cache refratário) usado aqui. *Population protocols* (Angluin et
al., 2006, 2007) formalizam computação por agentes anônimos de estado finito.

O custo que o *overlay* visa bater é a relaxação lenta da coordenação puramente local no anel,
agora ancorada por citação. Olfati-Saber e Murray (2004) provam que o desacordo do consenso
decai como $e^{-\kappa t}$ com taxa $\kappa=\lambda_2$, a *algebraic connectivity* (Fiedler,
1973), e afirmam que o anel é "uma forma relativamente lenta" de consenso por ter $\lambda_2$
pequeno; o *survey* (Olfati-Saber, Fax e Murray, 2007) consolida, e Boyd et al. (2006) ligam o
*averaging time* ao *mixing time*/spectral gap. Para o ciclo $C_N$, os autovalores do
Laplaciano são $2-2\cos(2\pi k/N)$, logo $\lambda_2 = 2(1-\cos(2\pi/N)) \approx (2\pi/N)^2 =
\Theta(1/N^2)$ (Spielman, 2009; Brouwer e Haemers, 2012), produzindo tempo de estabilização
$\Theta(N^2)$ no anel — corroborado empiricamente pelo DESYNC (Degesys et al., 2007), cuja
regra local de "pular para o ponto médio dos dois vizinhos" converge em $O(N^2)$ rounds. O
DESYNC — que atinge espaçamento equiangular **em fase** num anel lógico TDMA, é
auto-estabilizável e tolerante a churn, e se apoia nos osciladores *pulse-coupled* de Mirollo
e Strogatz (1990) — é o *baseline* mais relevante: é o análogo em espaço de fase do nosso
problema e exatamente a relaxação $O(N^2)$ que o *overlay* visa quebrar.

A tolerância a falhas é tratada no controle por **consenso resiliente**: LeBlanc et al. (2013)
introduzem *network robustness* ($r$- e $(r,s)$-robustez) e o W-MSR (condição
$(F\!+\!1,F\!+\!1)$-robusta, iff, sob adversário $F$-total); Usevitch e Panagou (2020) mostram
que determinar robustez é NP-difícil (decisão coNP-completa); Saulnier et al. (2017) levam o
W-MSR a robôs móveis via gestão de *algebraic connectivity*. Na teoria, Kuhn, Lynch e Oshman
(2010) modelam redes dinâmicas de pior caso (conectividade $T$-intervalar, $O(n^2)$ /
$O(n+n^2/T)$ rounds) e Casteigts et al. (2012) dão a taxonomia de grafos variantes no tempo,
cuja hierarquia transfere viabilidade e impossibilidade. Em enxames, Liu et al. (2024) obtêm
formação **auto-curável** robusta a ~50% de perda de pacotes e a entrada/saída de robôs — mas
por re-convergência emergente via consenso contínuo, não por disseminação event-triggered do
evento de topologia, e com comprimento de mensagem constante em vez de melhoria provada de
complexidade de rounds.

## 2.6 Controle por ondas (WBC) e o enquadramento "soliton-inspired"

A tese nasce de uma metáfora de sóliton — pulsos contra-propagantes que preservam forma e
sobrevivem a colisões. Uma revisão dedicada das coleções `soliton-inspired/` e `WBC/`
esclarece o que é genuinamente citável: **não há trabalho anterior que use um sóliton real
(KdV/sine-Gordon) como sinal de coordenação num enxame**; os itens que invocam a palavra
"sóliton" o fazem metaforicamente (otimização, mapas cognitivos), e os textos de sóliton
(Drazin e Johnson, 1989) são fundo de física. A linhagem **rigorosa e citável** é o **controle
por ondas (WBC)**: ondas contra-propagantes de "lançar/absorver" em cadeias 1-D (O'Connor e
McKeown, 2007) e controladores de pelotão veicular absorvedores de onda que curam instabilidade
de cadeia (Martinec, Šebek e Hurák, 2013) — diretamente análogos aos pulsos CCW/CW num anel —
além de métodos de equação de onda em grafos: consenso recolocado como PDE de onda amortecida
(Galbusera et al., 2012) e propagação de onda que deixa nós computarem o espectro do Laplaciano
localmente, muito mais rápido que difusão (Sahai et al., 2011). Por isso esta tese enquadra seu
*overlay* como **sinalização feedforward event-triggered**, citando WBC e consenso por onda-PDE
como precedente técnico, e mantém "soliton-inspired" apenas como rótulo de origem — posição
honesta, consistente com a ausência de qualquer precedente real de sóliton-coordenação.

## 2.7 Vizinhos mais próximos e o que os separa

**Risco principal de primazia — Gilbert, Lynch, Mitra e Nolte (2009)**, *Self-Stabilizing
Robot Formations over Unreliable Networks*. Ocupa simultaneamente vários elementos da nossa
fatia: auto-estabilização estrita de Dijkstra/Dolev (prova por TIOA + relação de simulação),
cenário genuinamente espacial/físico (robôs se distribuem uniformemente ao longo de uma curva
no plano, da qual o anel equiangular é caso especial), tolerância a crash/recovery, join/leave
e perda de mensagens, e desacoplamento discreto–contínuo via camada de *Virtual Stationary
Automata*. É a maior ameaça à novidade e deve ser citado como tal. É, contudo, separável em
três eixos decisivos: (1) **mecanismo** — a coordenação é uma **difusão periódica baseada em
rounds** sobre nós virtuais, sem pulsos hop-count contra-propagantes disparados por evento;
(2) **análise** — prova só corretude/convergência, sem escala $O(N^2)\!\to\!O(N)$, sem limite
$\Omega(N)$ de diâmetro, sem caracterização adimensional; (3) **problema** — o alvo é uma curva
fixa, não um alvo móvel cercado, e o controlador do robô é *bang-bang* trivial, não um
controlador 2-DOF com *gap-biasing*.

Os demais vizinhos não preemptam: o DESYNC (Degesys et al., 2007; §2.5) e o *deployment* em
anel CS-puro (Elor e Bruckstein, 2011; Shibata et al., 2022) estabelecem que o espaçamento
uniforme em anel é problema auto-estabilizável estabelecido com relaxação $O(N^2)$, mas são ou
de fase/tempo, ou livres de comunicação (oblivious), e nunca acoplam uma camada de disseminação
rápida a um atuador contínuo; a formação auto-curável de Liu et al. (2024) é o vizinho de
*redistribuição por falha* mais próximo, mas re-converge por consenso contínuo, não por
disseminação hop-count event-triggered; em física estatística, Geiss, Kroy e Holubec (2022)
caracterizam uma transição difusivo→balístico da propagação de informação no modelo de Vicsek
com atraso, mas sem definir um número adimensional *nomeado* de latência-vs-resposta —
confirmando que o número tipo-Péclet aqui proposto não tem, ao nosso conhecimento, contraparte
estabelecida; e o PCO em robótica (Anglea e Wang, 2019) permanece no domínio de
orientação/tempo, não de posicionamento espacial.

## 2.8 Posicionamento e gap

A tabela resume a comparação nas dimensões que definem a contribuição (✓ presente, ◐ parcial,
✗ ausente).

| Trabalho / família | Auto-estab. (Dijkstra) | Disseminação *event-triggered* | Cerco espacial/físico | Acoplamento ciber-físico 2-DOF | Escala $O(N^2)\!\to\!O(N)$ vs $\Omega(N)$ | Número tipo-Péclet |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Cyclic pursuit / mov. coletivo (Marshall 2004; Smith 2005; Sepulchre 2007) | ✗ | ✗ | ✓ | ◐ | ◐ | ✗ |
| Cerco c/ espaçamento + circumnav. SOTA (Yao 2017; Sui 2023; Zhou 2026; Jia 2024) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Estimator-coupled (Shames 2012; Deghat 2014; Franchi 2015; Boccia 2017) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| RL para cerco (Ma 2019; Qu 2026) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| ET/self-triggered *control* (Xu 2020; Babazadeh 2025; Psomiadis 2025) | ✗ | ✗ (sentido controle) | ✓ | ✗ | ✗ | ✗ |
| DESYNC / deployment em anel (Degesys 2007; Shibata 2022; Elor 2011) | ✓ | ✗ | ✗ (fase/grafo) | ✗ | ◐ (só $O(N^2)$) | ✗ |
| Formações de robôs auto-estab. (Gilbert 2009) | ✓ | ✗ (difusão) | ✓ | ◐ (2 camadas) | ✗ | ✗ |
| Controle por ondas / onda-PDE (O'Connor 2007; Martinec 2013; Sahai 2011) | ✗ | ◐ (ondas, não eventos) | ◐ (cadeias 1-D) | ◐ | ✗ | ✗ |
| Teoria resiliente / redes dinâmicas (LeBlanc 2013; Kuhn 2010; Liu 2024) | ◐ | ✗ | ◐ | ✗ | ◐ | ✗ |
| **Esta tese** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Cada ingrediente isolado tem prior art forte; o que **nenhum** trabalho combina — e que
portanto constitui a fatia defensável e provavelmente original desta tese — é a interseção de:
(1) um **protocolo de disseminação event-triggered** por pulsos hop-count contra-propagantes
disparados por crash/recovery, tolerante a perda de mensagens, conduzindo a redistribuição
angular; (2) um **acoplamento ciber-físico** que injeta a saída do protocolo discreto, por
*gap-biasing*, num controlador contínuo 2-DOF *sem desestabilizá-lo*; (3) um **resultado de
escala** que quebra a relaxação $O(N^2)$ rumo a $O(N)/O(\sqrt N)$ casando o limite $\Omega(N)$
do diâmetro do anel; e (4) uma **caracterização adimensional (tipo Péclet)** de *quando*
desacoplar coordenação de atuação compensa.

**Declaração honesta de novidade.** Propagação rápida de informação, isoladamente, não é nova
— *flooding* já atinge $O(\text{diâmetro})$ e o WBC já usa sinais contra-propagantes em cadeias
1-D. A novidade defensável não está na velocidade *per se*, mas em (a) o acoplamento
ciber-físico que deixa o resultado de um algoritmo distribuído discreto conduzir um controlador
contínuo 2-DOF sem desestabilizar, **num anel em torno de um alvo móvel**; (b) o modelo de
falhas crash/recovery com perda de mensagens; e (c) a caracterização adimensional do regime em
que o desacoplamento compensa. Como reivindicação de originalidade é prova de ausência,
limitada à cobertura de venues indexados, ela é feita "ao melhor do nosso conhecimento", com os
vizinhos mais próximos — Gilbert et al. (2009), Degesys et al. (2007), Shibata et al. (2022),
Liu et al. (2024) e Xu et al. (2020) — citados e explicitamente diferenciados.
