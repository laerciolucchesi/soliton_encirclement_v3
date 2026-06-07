# Capítulo 8 — Validação, escopo e a ponte para o hardware

> Rascunho/esqueleto (PT). Status: `[a fazer]`. Trabalho detalhado no plano de campanha,
> Fases 1 e 5 (`docs/tese_estrutura.md`).

## 8.1 Tese do capítulo
Transformar "prova de mecanismo" em "evidência de tese": campanha de validação em larga
escala + ponte para o hardware, prevendo pelo número de Péclet (Cap. 6) se o overlay ajuda na
plataforma e **confirmando voando**.

## 8.2 Estado e o que falta
- [x] **Campanha de escala (falha única controlada) — FEITA até N=100:** baseline Θ(N²)
      (N^1.97, R² 0.89–0.94) e B2 plano (~2.1 s, 2 seeds, R² 0.95–0.97), vantagem ~N²
      (até 149×), mensagens O(N). Pacote de 9 figuras pronto. FALTA: ponto N=150 (opcional) e
      a varredura ampla de condição inicial (`INIT_RADIUS_RANGE` + ângulos não-equidistantes).
- [ ] **SITL** (ex.: Gazebo/ArduPilot) com poucos agentes: medir $\tau_a$ real, calcular Pe,
      **prever** o ganho do overlay e **confirmar** voando.
- [ ] (Se possível) **demo com 3–5 drones reais** — mesmo pequena, blinda a defesa.

## 8.3 Escopo honesto
- Os 6 métodos PDE anteriores viram **apêndice** (tentativas anteriores).
- O rótulo "**soliton**" fica só onde tem carga estrutural (colisão de pulsos em falhas densas);
  caso contrário, o mecanismo é descrito como **sinalização feedforward event-triggered** (cf.
  Cap. 2, §2.6 — a linhagem citável é o controle por ondas / WBC).
- Foco em **evento único** isola a afirmação central; idealização sem-colisão (drone falho
  congelado, vivos redistribuem em $2\pi$) declarada — evasão de colisão ao redor do nó falho
  é **trabalho futuro**.

## 8.4 Higiene metodológica (Fase 0) — quase toda feita
- [🔁] `HYSTERESIS_RAD` adimensional: ANALISADO — não morde na falha única limpa (ordem cíclica
      preservada; só em N~126 sob reordenação) → movido para a **Fase 3 (churn)**.
- [~] **Multi-seed:** PARCIAL — 2 seeds (nó que falha variado) PASSA. FALTA `INIT_RADIUS_RANGE`
      + ângulos não-equidistantes; reportar mediana + dispersão.
- [x] **Offset de regime CONFIRMADO** — late_std ~0 e `egap_final` ~0.001 em N≤100 (B2 settled).
