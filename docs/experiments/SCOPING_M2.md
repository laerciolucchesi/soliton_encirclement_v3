# Scoping — item 9: baseline externo por densificação m=2

**Escopo apenas; nada implementado.** Este documento responde às sete perguntas do pedido
com referências ao código atual (commit `d48ed62`). Vocabulário: célula = uma rodada;
condição = (alcance, timeout) com 8 sementes × métodos; grade = tudo.

**O que m=2 é aqui:** acoplamento DIRETO de cada agente aos vizinhos 1 e 2 de cada lado —
uma transmissão física, sem relé. Exige alcance ≥ corda de 2 saltos `2R·sin(2π/N)`, isto é
c ≥ `2cos(π/N)` = 1,9829 (N=24) — o mesmo limiar geométrico que a fase 8a mediu para o
overlay (`c** ∈ (1,99; 3,01]` sob churn 12/min). Objetivo: terceira linha MEDIDA nas
tabelas (baseline / m2 / overlay), com custo de mensagens ao lado.

---

## 1. Arquivos e lei

### 1.1 A lei atual (m=1), exata

[protocol_agent.py:519-538](../../protocol_agent.py#L519-L538), `compute_spacing_error`:

```
e_tau = (lp_pred·g_succ − lp_succ·g_pred) / (lp_pred·g_succ + lp_succ·g_pred)
```

onde `g_pred`, `g_succ` são os vãos angulares medidos e os pesos vêm de
`_update_neighbor_lps_from_target` ([:640-665](../../protocol_agent.py#L640-L665)):
`lp_pred = λ(pred₁)` e `lp_succ = λ(self)` — **a convenção de indexação é que cada arco
pertence ao seu nó inicial no sentido CCW**: o arco pred→self é indexado por pred, o arco
self→succ é indexado por self. Equilíbrio: `g_pred/g_succ = λ(pred)/λ(self)`, arcos
proporcionais aos lambdas. O erro é adimensional em [−1, 1].

`e_tau` alimenta `u_local` no `TangentialSpacingController.update()` com ganho `K_E_TAU`
(controllers.py); a costura no laço de controle é `compute_e_tau_used(...)` em
[protocol_agent.py:1117](../../protocol_agent.py#L1117).

### 1.2 A lei m=2 proposta — MESMA normalização por arcos locais

Para alcance de salto k ∈ {1, 2}, com `g_pred_k`, `g_succ_k` os vãos medidos até o k-ésimo
vizinho e os pesos como **somas dos lambdas dos arcos que o vão atravessa** (a mesma
convenção de indexação da 1.1):

```
Λ_pred_k = λ(pred_k) + … + λ(pred_1)          (k arcos do lado pred)
Λ_succ_k = λ(self) + λ(succ_1) + … + λ(succ_{k−1})   (k arcos do lado succ)

e_tau^(k) = (Λ_pred_k·g_succ_k − Λ_succ_k·g_pred_k) / (Λ_pred_k·g_succ_k + Λ_succ_k·g_pred_k)
```

Para k=1 isto **reduz exatamente à lei atual** (Λ_pred_1 = λ(pred₁), Λ_succ_1 = λ(self)) —
não é uma lei nova com um caso especial, é a mesma lei parametrizada por k. Combinação:

```
e_tau_m2 = (e_tau^(1) + w₂·e_tau^(2)) / (1 + w₂)        w₂ = M2_W2, default 1.0
```

**Ganho.** Para a comparação pareada valer com a mesma margem de estabilidade, o ganho do
laço é renormalizado pelo autovalor máximo do Laplaciano do anel: para N=24,
λ_max(m1) = 4,000 (k=12) e λ_max(m2) = 6,2497 (k=7), razão **1,5625**. Proposta:
`K_E_TAU_M2 = K_E_TAU / 1.5625` = (250/N)/1,5625, fixado no runner como os demais. É esta
renormalização que produz a predição de ~3,2× (§5.3): razão λ₂ = 4,930 dividida pela razão
λ_max = 1,5625 → **3,155**.

**Divergência da lei atual: nenhuma na normalização.** O único elemento novo é a média dos
dois erros adimensionais (ambos em [−1, 1]) e a renormalização de ganho — declarada, com o
divisor pinado. A comparação pareada não é perturbada porque `e_tau_real` (telemetria)
continua sendo o erro m=1 físico em TODOS os métodos (ver §2).

### 1.3 Arquivos que mudam

| arquivo | mudança |
|---|---|
| `config_param.py` | knobs `M2_W2` (1.0), `M2_GAIN_SCALE` (1/1,5625); nenhum seletor novo (§7) |
| `protocol_agent.py` | (a) generalizar a construção do anel de `get_two_neighbors` para devolver também pred₂/succ₂ — mesmo `ring` ordenado de [:790-799](../../protocol_agent.py#L790-L799), índices ±2, com a guarda de aliasing do §3; (b) `compute_spacing_error_k` (a lei 1.2); (c) ramo no laço de controle na costura já existente de `compute_e_tau_used` |
| `tests/test_m2_law.py` | novo: álgebra da lei (redução a m=1 com w₂=0; simetria; equilíbrio com lambdas não uniformes; guarda de aliasing) |
| `main.py` | uma linha em `_METHODS` (menu) — §7 |
| runner da campanha | novo `run_m2_campaign.py` no molde do `run_comm_churn_sweep.py`, herdando as três asserções |

`plot_telemetry.py`, `provenance.py`, `protocol_target.py`, `controllers.py`: **intocados.**

## 2. O que fica idêntico — com a construção apontada

| item | construção que garante a identidade |
|---|---|
| Semeadura global (posições iniciais) | `random.seed(EXPERIMENT_SEED)` em [main.py:165](../../main.py#L165), ANTES do laço `builder.add_node`; o seletor de método é lido do env e não consome RNG |
| Fluxo de falhas (pareamento por semente) | RNG por agente `random.Random(0xF00DCAFE + node_id + seed·10000)` em [protocol_agent.py:99-101](../../protocol_agent.py#L99-L101), fórmula fechada independente do método; único consumidor é o sorteio Bernoulli em [:919](../../protocol_agent.py#L919), no timer de falha, agendado em `initialize()` igual para todos os métodos |
| Telemetria | mesmas colunas do contrato em CLAUDE.md; `e_tau` = entrada do controlador (no método m2, a lei 1.2); **`e_tau_real` = erro m=1 físico sempre**, computado dos vãos de 1 salto não modificados — é a coluna que M1..M7 e as comparações entre métodos já usam, então o pipeline fica byte-compatível |
| Métricas / análise | os runners da 8a leem `target_telemetry.csv` + `events.csv`; nenhum dos dois muda de esquema |
| Proveniência | o valor do seletor já entra no manifesto por `env_overrides()` ([provenance.py:297-311](../../provenance.py#L297-L311), `PROPAGATION_METHOD` em `_EXTRA_ENV_KEYS`) |

**RNG não desloca entre métodos, por construção:** os dois pontos de semeadura acima
precedem qualquer código dependente de método, e o ramo m2 não introduz nenhum sorteio.
Verificação barata no teste de fumaça: rodar baseline e m2 com a mesma semente e comparar
`events.csv` (timestamps de `failure_start`) — devem ser idênticos byte a byte.

## 3. Segundo vizinho fora de alcance

**O que o código faria hoje** (sob `RoleAwareCommunicationHandler`, alcance < corda de 2
saltos): o broadcast do 2º vizinho **não é entregue**, logo ele não entra em
`agent_states` — ou entra e expira, porque `get_two_neighbors` filtra candidatos com
`(now − rxtime) > AGENT_STATE_TIMEOUT` ([:779](../../protocol_agent.py#L779)). Entre sair
de alcance e expirar, os vãos seriam computados de posição velha ≤ timeout — a mesma
estalidade que a lei m=1 já tolera hoje para o 1º vizinho sob flapping.

**O perigo real é outro: aliasing.** O `ring` é construído SÓ com os nós ouvidos e frescos
([:790-799](../../protocol_agent.py#L790-L799)). Com alcance curto o anel visível típico é
{pred₁, self, succ₁} — 3 membros — e a extensão ingênua `ring[(self_idx + 2) % len(ring)]`
devolveria **pred₁ como "succ₂"**: o termo k=2 agiria sobre o nó errado com sinal errado.
É o análogo m=2 da ENTRADA espúria da fase 8a.

**Proposta (vira braço do desenho):** o termo k=2 é **descartado e a lei renormaliza para
m=1** quando (a) o anel visível fresco tem < 5 membros, ou (b) o candidato a k=2 não está
fresco. Sem estado velho, sem aliasing. Duas colunas novas de telemetria de agente com o
escopo no nome: `m2_k2_dropped_frac_steady20` (fração dos ticks em regime com o termo k=2
descartado, por lado) e `m2_ring_lt5_frac_steady20`. O desenho reusa a grade da 8a:
**c ∈ {1,61 (aperto — degrada para m=1 quase sempre); 3,01 (folga — termo k=2 ativo)}**,
com a predição de que em c=1,61 o método m2 fica estatisticamente indistinguível do
baseline (é o teste de que a degradação está correta).

## 4. Mensagens por rodada

**Fato de construção que a medição deve exibir, não esconder:** todos os métodos
transmitem exatamente **1 broadcast de `AgentState` por agente vivo por tick**
([protocol_agent.py:1363-1364](../../protocol_agent.py#L1363-L1364)). m=2 **não adiciona
transmissão nenhuma** — o 2º vizinho já ouve o broadcast existente quando o alcance
permite; o que muda é o alcance EXIGIDO (energia por transmissão) e o payload do overlay
(pulsos em `prop_state`). O precedente de contagem é `dual_pulse_messages.csv`
([protocol_agent.py:1615-1627](../../protocol_agent.py#L1615-L1627)):
`get_broadcast_pulse_count()` por nó.

Colunas propostas por célula (escopo no nome, denominador na linha — regra do §5b do
README):

| coluna | definição |
|---|---|
| `tx_broadcasts_steady20` | broadcasts de AgentState no regime; deve ser ≈ igual entre métodos e a tabela IMPRIME os três para provar |
| `pulse_payloads_fullrun` | soma do `dual_pulse_messages.csv` (0 para baseline e m2) |
| `range_required_c` | o c mínimo do método (1 para baseline, 2cos(π/N) para m2; overlay: 2cos(π/N) pela 8a) — o custo REAL de m=2 é este, não mensagens |
| `run_duration_s` | denominador, em toda linha |

## 5. Custo estimado: **≤ 1 semana** — segue o plano

### 5.1 Implementação (2–3 dias)

1. Lei + generalização do anel + guarda de aliasing (protocol_agent) — 1 dia.
2. Seletor (§7), knobs, asserção anti-combinação — meio dia.
3. Testes unitários: redução exata a m=1 (w₂=0 e também anel < 5), simetria de sinal,
   equilíbrio com lambdas não uniformes (PROTECTION_ANGLE), aliasing bloqueado — meio dia.
4. Fumaça pareada: baseline vs m2, mesma semente, `failure_start` byte-idêntico;
   `e_tau_real` byte-idêntico com `M2_W2=0` — meio dia.

### 5.2 Campanha (1–2 dias de máquina + análise)

Molde do `run_comm_churn_sweep.py`, com as três asserções mantidas (A1 matriz efetiva,
A2 vivos reconstruídos por superposição, A3 censo de papéis). E1-style:
métodos {baseline, m2, dual_pulse} × N ∈ {24, 50} × regime {óbito único, churn 12/min} ×
c ∈ {1,61; 3,01} × 8 sementes pareadas — com poda das células sem pergunta (m2 em c=1,61 é
a verificação de degradação, não precisa dos dois regimes; grade final ~200–250 células,
~3–4 h a ~45 s/célula).

### 5.3 Esboço de pré-registro (entra ANTES do dado, no docstring do runner)

- **P5 (regime limpo):** com `K_E_TAU_M2 = K_E_TAU/1,5625`, o tempo de reconfiguração do
  m2 melhora sobre o baseline por **λ₂-razão/λ_max-razão = 4,930/1,5625 ≈ 3,16×** em N=24
  (recalcular e pinar os autovalores discretos para N=50 antes de rodar). Predição
  derivada, não ajustada — o mesmo estilo do `2cos(π/N)` da 8a.
- **P6 (churn, c=3,01):** ordenação prevista overlay ≥ m2 > baseline em
  `egap_mean_steady20`; a fase 8a mediu overlay 1,19× — se m2 ficar entre 1,19× e 1×, a
  vantagem do overlay sobre a densificação é o número que falta ao texto.
- **P7 (aperto, c=1,61):** m2 ≈ baseline (degradação limpa); qualquer coisa diferente é
  bug da guarda, não resultado.
- Censura por critério, n_ev/n_run/picos distintos nas tabelas por evento, sentinelas
  abortando — tudo herdado.

## 6. Alterações fora deste arquivo

Nenhuma. Este documento é o único artefato novo; nada de código, config, testes ou docs
foi tocado.

## 7. Seleção do braço por config

**Como é selecionado HOJE:** um único eixo, `PROPAGATION_METHOD` — menu interativo em
[main.py:78-128](../../main.py#L78-L128) (`_METHODS`) ou env para batch, consumido UMA vez
em [protocol_agent.py:187-193](../../protocol_agent.py#L187-L193)
(`create_propagation_layer`). "baseline" e "dual_pulse" já são valores deste eixo — não
existe seletor separado para o overlay.

**Extensão mínima proposta:** novo valor **`m2`** no MESMO eixo. `create_propagation_layer("m2")`
devolve a camada baseline (no-op), e `protocol_agent` deriva `self._m2_enabled = (método == "m2")`
no mesmo ponto [:187-193] — as duas ramificações (conjunto de vizinhos, lei de e_tau) são
keyed nesse booleano, na costura de `compute_e_tau_used`. Nenhum seletor paralelo.

Requisitos, um a um:

- **(a) manifesto:** já satisfeito — `PROPAGATION_METHOD` está em `_EXTRA_ENV_KEYS`
  ([provenance.py:299](../../provenance.py#L299)) e entra em `env_overrides()` de todo
  manifesto de rodada. Nada a fazer.
- **(b) default preserva:** o default do env é `"baseline"`
  ([protocol_agent.py:187](../../protocol_agent.py#L187)); o ramo m2 é código morto sob
  qualquer valor ≠ "m2", e o caminho m=1 não é editado — é a condição do `if`. Reprodução
  bit a bit das campanhas antigas verificável pela fumaça pareada do §5.1.
- **(c) combinações não planejadas abortam:** m2 + dual_pulse é **inexprimível** — o eixo
  tem um valor só. Asserção defensiva mesmo assim, em `initialize()`:
  `assert not (self._m2_enabled and self.dual_pulse_layer is not None)`. E `PROPAGATION_K_PROP`
  ≠ 0 com m2 aborta igual ("m2" entra em `_METHODS_WITHOUT_K_PROP`,
  [main.py:92](../../main.py#L92)).
- **(d) ramificação depois da semeadura:** a semeadura global é [main.py:165](../../main.py#L165)
  e a per-agente é [protocol_agent.py:99](../../protocol_agent.py#L99), ambas antes de
  qualquer código keyed no método; a leitura do env não consome RNG; o ramo m2 não sorteia
  nada. Verificação: fumaça do §5.1 (fluxo de falhas byte-idêntico entre métodos).


---

# ADENDO — pinos (a) e (b) da aprovação condicionada (2026-08-04)

Aprovação condicionada a dois consertos; este adendo os fixa ANTES de qualquer
implementação. Na verificação do pino (a) apareceu um TERCEIRO fator 2, da minha lei do
§1.2, que muda os números do pino sem mudar seu critério — documentado em A.2 com a
identidade que prova que o laço fechado resultante é exatamente o que o pino prescreve.

## A.1 Pino (a) — o critério de justiça, explícito

> **Mesma margem nominal amostrada `g·dt·λ_max` nas duas leis.**
> N=24: baseline (250/24)·0,05·4 = **2,0833**; m2 idem = **2,0833**.
> N=50: (250/50)·0,05·4 = **1,0000** nos dois.

Consequência que simplifica tudo: sob esse critério, o speedup previsto **depende só da
FORMA do operador**, não da sua escala —

```
speedup = (λ₂/λ_max)_m2 / (λ₂/λ_max)_m1
```

Qualquer reescala global do erro é absorvida pelo fator de ganho que restaura λ_max.

## A.2 O terceiro fator 2 — e por que o default é w₂ = 2

Linearizando a lei do §1.2 em torno do anel uniforme (todos os erros em unidades de
`e = −A·δθ/(2ḡ)`): o termo k=1 dá `A₁ = L₁` (Laplaciano de 1 salto). Mas o termo k=2,
**por ser normalizado pelo próprio vão de 2 arcos** (denominador ≈ 4ḡ, não 2ḡ), entra
como `L₂/2` — a normalização por vão penaliza o acoplamento de salto k por 1/k. A
combinação `(e⁽¹⁾ + w₂·e⁽²⁾)/(1+w₂)` tem operador:

```
A(w₂) = (L₁ + (w₂/2)·L₂) / (1+w₂)
```

O pino (a) calculou com o operador `(L₁+L₂)/2` — que a média w₂=1 NÃO produz sob a lei
do §1.2. As três prescrições, lado a lado (N=24, autovalores discretos):

| prescrição | operador | λ_max | margem | speedup |
|---|---|---:|---:|---:|
| escopo §1.2 original (w₂=1, ganho ÷1,5625) | (L₁+½L₂)/2 | 2,250 | **0,75** | **0,95 — PIOR que o baseline** |
| pino (a) literal sobre a lei §1.2 (w₂=1, ×1,2801) | (L₁+½L₂)/2 | 2,250 | **1,50** | **1,90** |
| **este adendo (w₂=2, ×1,9201)** | **(L₁+L₂)/3** | 2,08323 | **2,0833 = baseline** | **3,1565** |

O escopo original não só nascia com metade da margem, como o pino diagnosticou — nascia
com speedup **0,95**, a P5 refutada pela construção duas vezes.

**Conserto:** `M2_W2 = 2` por default. w₂ = k cancela exatamente o 1/k da normalização
por vão — peso FÍSICO igual por acoplamento, que é o anel densificado uniforme `L₁+L₂`
que a predição do pino pressupõe. A combinação continua convexa (÷(1+w₂) = ÷3), então
`e ∈ [−1,1]` é mantido, como o pino exige. E:

```
M2_GAIN_SCALE = λ_max(L₁) / λ_max(A(w₂=2))  =  4 / 2,08323  =  ×1,92010   (N=24)
                                               4 / 2,07869  =  ×1,92429   (N=50)
```

**Identidade que prova a equivalência com o pino (a):** o laço do pino (×1,2801 sobre
`(L₁+L₂)/2`) e o deste adendo (×1,9201 sobre `(L₁+L₂)/3`) são o MESMO laço fechado —
ambos restauram `escala·λ_max = 4`: 1,2801·3,1248 = 4,000 = 1,9201·2,08322. A diferença
entre os fatores (1,9201/1,2801 = 3/2) é exatamente a razão dos divisores internos. O
pino estava certo sobre o laço-alvo; o que faltava era o w₂ que faz a lei do §1.2
produzir esse laço.

**Predições derivadas DESSA fórmula, pinadas:**

| N | λ₂(L₁) | λ₂(L₁+L₂) | razão λ₂ | λ_max(L₁+L₂) | razão λ_max | **predição** |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0,0681483 | 0,3360975 | 4,93185 | 6,2496889 (k=7) | 1,5624222 | **3,1565** |
| 50 | 0,0157706 | 0,0786043 | 4,98423 | 6,2360680 (k=15) | 1,5590170 | **3,1970** |

`M2_GAIN_SCALE` é computado no config dos autovalores discretos (fórmula fechada, zero
RNG), com override numérico por env.

## A.3 Pino (b) — a guarda restaura lei E ganho

Quando o termo k=2 cai, o tick alimenta **`e⁽¹⁾` sem escala nenhuma** — nem combinação,
nem `M2_GAIN_SCALE`. Implementação que torna isso trivial e bit-exato: o caminho do erro
no controlador é LINEAR (`du_from_error = k_e_tau·e_tau`, controllers.py, laço local),
então o chaveamento de ganho é implementado **escalando o erro**, não o `k_e_tau` do
controlador. Tick degradado ⇒ o mesmo caminho aritmético do baseline, float a float — a
P7 (m2 ≈ baseline em c=1,61) segue da construção, e a fumaça S3 a verifica por
byte-igualdade.

Declarações exigidas pelo pino:

- **(i)** `e⁽²⁾` exige os DOIS lados frescos. Um lado ausente ou não fresco descarta o
  termo k=2 INTEIRO (meio termo mudaria o operador e a margem). As colunas por lado
  existem só para diagnóstico.
- **(ii)** chattering do chaveamento sob churn: medido por `m2_k2_dropped_frac_steady20`
  (fração dos ticks em regime com o termo descartado) e ganha linha própria no
  pré-registro (P8, §A.4). **Adição minha, justificada:** a fração não distingue 1
  chaveamento de 1000 — acrescento `m2_k2_toggles_per_s_steady20` (taxa de bordas do
  chaveamento), que é a medida de chattering de fato. As duas colunas com o escopo no
  nome e `run_duration_s` na linha.
- Guarda completa: o termo k=2 cai quando o anel visível fresco tem **< 5 membros** (a
  extensão ingênua aliasa `succ₂ = pred₁` num anel de 3 — §3) OU qualquer candidato k=2
  está ausente/não fresco. Sem histerese no k=2 (a histerese existente protege só
  pred₁/succ₁); qualquer flapping de identidade resultante aparece nas colunas de
  chattering — declarado, não escondido.

## A.4 Itens de pré-registro exigidos (entram no docstring do runner antes da grade)

**1. Tabela de células enumerada — 192, sem poda.** A estimativa "~200–250 com poda" do
§5.2 estava errada nas duas direções ao mesmo tempo (poda reduziria, não aumentaria; a
grade cheia é 192). Conta certa: 3 métodos × 2 N × 2 regimes × 2 c × 8 sementes:

| bloco | N | regime | c (alcance em m) | células |
|---|---:|---|---|---:|
| A | 24 | óbito único | 1,6089 (8,4) | 24 |
| B | 24 | óbito único | 3,0071 (15,7) | 24 |
| C | 24 | churn 12/min | 1,6089 (8,4) | 24 |
| D | 24 | churn 12/min | 3,0071 (15,7) | 24 |
| E | 50 | óbito único | 1,6089 (4,041) | 24 |
| F | 50 | óbito único | 3,0071 (7,553) | 24 |
| G | 50 | churn 12/min | 1,6089 (4,041) | 24 |
| H | 50 | churn 12/min | 3,0071 (7,553) | 24 |

Declarações que fecham a conta: **timeout único fd = 0,25 s** (a 8a-(ii) já mediu o eixo
do timeout; dobrá-lo aqui dobraria a grade); **churn 12/min TOTAL nos dois N** (mesmo
fluxo de eventos por rodada entre N, comparabilidade pareada); blocos C e D **re-rodam**
baseline e overlay em vez de reusar as linhas da 8a-(ii), para proveniência uniforme sob
um commit — com SENTINELA: o `target_telemetry.csv` das células baseline re-rodadas deve
sair **byte-idêntico** ao de `comm_churn_runs/` (o caminho baseline não é editado; se
divergir, a implementação não é inerte e o sweep aborta). Custo: medido por célula de
calibração em cada N na fase de fumaça, não estimado.

**2. N=50 pinado.** Autovalores discretos na tabela do A.2 (λ_max em k=15, não no
contínuo). Alcances em metros derivados de `c·2R·sin(π/N)` POR N: corda₁(50) = 2,5116 m
⇒ **4,041 m** e **7,553 m** (não os metros de N=24). Limiar: 2cos(π/50) = **1,99605**;
corda₂(50) = 5,0133 m.

**3. P6 simétrica.** Predição: overlay ≥ m2 > baseline em `egap_mean_steady20` sob churn
em c=3,01. **Desfecho adverso registrado com o significado:** se m2 superar o overlay, a
comparação de alcance igual ENFRAQUECE o argumento do §4.1 da tese — o overlay perderia
para uma densificação passiva no mesmo raio de rádio; isso é resultado publicável, não
falha. Predição cruzada no regime limpo, N=50: overlay/m2 ≈ 16,0/3,197 ≈ **5,0×**.

**4. Linha de ameaça declarada.** A margem nominal igual assume o fator de ganho efetivo
da normalização por arcos (1/(2ḡ), lambdas uniformes) COMUM às duas leis. Lambdas não
uniformes ou desvios grandes do uniforme quebram a igualdade nominal. Mitigação: a
campanha fixa `PROTECTION_ANGLE_DEG=0` explicitamente (regra 3) e a ameaça fica na seção
de ameaças do runner.

**5. Fumaças, incluindo a exigida.** S1: baseline vs m2, mesma semente ⇒ `failure_start`
byte-idêntico (RNG não desloca). S2: `M2_W2=0` + `M2_GAIN_SCALE=1` ⇒ telemetria
byte-idêntica ao baseline. S3 (a exigida): guarda forçada — alcance abaixo da corda de
2 saltos, onde o anel visível é 3 — ⇒ byte-idêntica ao baseline (P7 por construção).
S4: c=3,01 ⇒ telemetria DIFERE (termo ativo; sanidade de que a lei liga).

Sequência acordada: este adendo (commit isolado) → implementação com os testes do §5.1 →
fumaças → pré-registro completo volta para revisão ANTES de disparar a grade. Nenhuma
grade roda sem essa revisão.
