#!/usr/bin/env python
"""Fase 8a (ii) -- CHURN sob localidade: alcance finito do anel + fluxo de Poisson.

Runner SEPARADO do run_comm_range_sweep (fases i / i-b) de proposito: o modelo de
falha, a assercao A2 e as metricas primarias sao outros, e ramificar aquele
arquivo arriscaria a reprodutibilidade de 112 linhas ja publicadas. Os auxiliares
comuns sao IMPORTADOS de la, nao copiados.

=======================================================================
PRE-REGISTRO -- escrito e commitado ANTES da grade rodar. Nao editar depois.
=======================================================================

P3 -- O c** efetivo SOBE acima de 2*cos(pi/N) = 1.9829.
  Racional: 2*cos(pi/N) e' a razao corda-2-saltos / corda-1-salto, isto e', o
  limiar medido nas fases (i)/(i-b) para um anel UNIFORME. Sob churn os vaos
  chegam a ~2x o ideal, entao a distancia que o pulso de SAIDA precisa cruzar
  cresce e o limiar deve subir.
  Discriminador: c = 2.0 fica 0.9% ACIMA de 1.9829. Se c** subir por pouco que
  seja, c = 2.0 vira de sucesso (fase i) para falha sob churn.
    c=2.0 falha  -> P3 confirmada
    c=2.0 aguenta -> P3 refutada, o limiar uniforme sobrevive ao churn

P4 -- A ENTRADA espuria interage com as ENTRADAS legitimas (retornos reais).
  Reportar espurias/legitimas por celula. REGRA DE CLASSIFICACAO, fixada aqui
  antes do dado:

    PRIMARIA (por IDENTIDADE). Reconstroi-se o conjunto vivo ao longo do tempo
    dos failure_start/failure_end. Para um evento ENTRADA que aterrissou, com
    originador o (prefixo do event_id) e instante t (primeira conclusao):
      candidato = primeiro no' apos o, em ordem ciclica de id, VIVO em t
                  (isto e', o sucessor nominal do originador naquele instante)
      LEGITIMA  se o candidato tem failure_end em [t-W, t]
      ESPURIA   caso contrario
    Identidade mata a coincidencia que a contagem nao mata (retorno do agente X
    coincidindo no tempo com ENTRADA espuria do agente Y). Nao exige
    instrumentacao nova: o originador esta no event_id e a ordem do anel segue a
    ordem dos ids -- premissa ja usada nas fases (i)/(i-b), onde o sucessor da
    vitima foi vitima+1 nas 8 sementes.

    SECUNDARIA (por CONTAGEM, declarada como sensibilidade). LEGITIMA se o total
    de vivos reconstruido AUMENTOU em [t-W, t]. Reportada lado a lado; a
    divergencia entre as duas mede a taxa de coincidencia.

    W = AGENT_STATE_TIMEOUT + 0.5 s. Reportado tambem em 2W como sensibilidade.
    W ACOMPANHA O TRATAMENTO (0.25 vs 1.0 s), logo a resolucao do classificador
    DIFERE entre os bracos -- isso tem de ser dito ao compara-los, e e' o motivo
    de a sensibilidade em 2W ser obrigatoria e nao opcional.

A2 SOB CHURN -- a assercao de uplink, redefinida.
  O limite >= N-1 e' do regime de obito unico e abortaria na primeira celula:
  medido na calibracao, alive_count cai a 20 (quatro agentes fora ao mesmo
  tempo). O PROPOSITO da sentinela e' pegar corrupcao de uplink, nao morte real.
  Redefinicao: reconstroi-se o numero VERDADEIRO de vivos dos eventos e exige-se
  que o alive_count do alvo o acompanhe com atraso limitado.
    TOLERANCIA = AGENT_STATE_TIMEOUT + 2*dt   (numerica, fixada aqui; 0.35 s no
    braco de 0.25 e 1.10 s no braco de 1.0)
  Viola se o alvo relatar MENOS vivos do que a verdade por mais que a
  tolerancia. Relatar mais que a verdade tambem viola (nunca deveria acontecer).

METRICAS -- primaria e' o erro medio em regime, nao t_close.
  Sob churn contínuo nao ha assentamento (precedente: run_churn_sweep) e a
  calibracao confirma: mediana de G_max = 1.427, o anel nunca volta abaixo de
  1.25, entao t_close e' censurado por construcao em praticamente toda celula.
  A censura E' reportada como medida (censura x c, nos dois criterios), nao como
  apendice.
  Os nomes de coluna carregam a definicao (janela e limiar), para nao repetir o
  homonimo de egap_avg -- ver analysis_churn/EGAP_HOMONIMO.md:
    egap_mean_steady20 / egap_p90_steady20 / egap_max_steady20
    frac_gmax_gt125_steady20 / frac_gmax_gt150_steady20
  "steady20" = t >= T0 + 15 = 20 s, a janela DEF-A do run_churn_sweep, ate o fim
  da rodada. Denominador identico entre celulas (budget igual), logo comparavel.
  Pico por evento vem de analysis_churn.aggregate_gmax.event_rows -- janela
  adaptativa e pre-registro proprios, reusados em vez de redefinidos.

Uso:
    python experiments/scaling_law/run_comm_churn_sweep.py
    # env: CCS_RANGES="8.4,10.4,15.7"  CCS_TIMEOUTS_LOW="0.25,1.0"
    #      CCS_TIMEOUTS_HIGH="0.25"    CCS_C_SPLIT="2.5"
    #      CCS_RATE="12"  CCS_OFF="8"  CCS_SEEDS="0,1,2,3,4,5,6,7"
    #      CCS_BUDGET="150"  CCS_METHODS="baseline,dual_pulse"  CCS_TAG=""
    #      CCS_DRY_RUN="1"
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_churn"))

from metrics_util import effort_metrics  # noqa: E402
from run_comm_range_sweep import (  # noqa: E402  -- auxiliares comuns, nao copiados
    AssertionFailed,
    chord,
    c_units,
    c_units_post,
    parse_comm_marker,
    provenance_with_retry,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("CCS_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "comm_churn_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "comm_churn_results" + _SUF + ".csv")
EVENTS_CSV = os.path.join(EXP_DIR, "comm_churn_events" + _SUF + ".csv")
WORK_CSV = os.path.join(RUNS_DIR, "_partial_results.csv")   # ignorado pelo *_runs/

RANGES = [float(x) for x in os.environ.get("CCS_RANGES", "8.4,10.4,15.7").split(",") if x.strip()]
TIMEOUTS_LOW = [float(x) for x in os.environ.get("CCS_TIMEOUTS_LOW", "0.25,1.0").split(",") if x.strip()]
TIMEOUTS_HIGH = [float(x) for x in os.environ.get("CCS_TIMEOUTS_HIGH", "0.25").split(",") if x.strip()]
C_SPLIT = float(os.environ.get("CCS_C_SPLIT", "2.5"))   # c <= split -> os dois timeouts
RATE_TOTAL = float(os.environ.get("CCS_RATE", "12"))    # por minuto, TOTAL (convencao da campanha)
OFF = float(os.environ.get("CCS_OFF", "8"))
SEEDS = [int(x) for x in os.environ.get("CCS_SEEDS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
METHODS = [m.strip() for m in os.environ.get("CCS_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
N = int(os.environ.get("CCS_N", "24"))
R_ENC = float(os.environ.get("CCS_RADIUS", "20"))
UPLINK = float(os.environ.get("CCS_UPLINK", "200"))
TAU = float(os.environ.get("CCS_TAU", "1.0"))
BUDGET = float(os.environ.get("CCS_BUDGET", "150"))
DT = float(os.environ.get("CONTROL_PERIOD", "0.05"))
VMAX = float(os.environ.get("CCS_VMAX", "10.0"))
T0 = 5.0
WARMUP_AVG = 15.0          # DEF-A do run_churn_sweep: regime comeca em T0+15 = 20 s
STEADY_T0 = T0 + WARMUP_AVG
THR_PRIMARY = 1.25
THR_STRICT = 1.10
THR_WIDE = 1.50
PROGRESS_EVERY = max(1, int(os.environ.get("CCS_PROGRESS_EVERY", "8")))
DRY_RUN = os.environ.get("CCS_DRY_RUN", "").strip().lower() in ("1", "true", "yes", "y")


def timeouts_for(rng):
    return TIMEOUTS_LOW if c_units(rng) <= C_SPLIT else TIMEOUTS_HIGH


# ---------------------------------------------------------------------------
# reconstrucao do conjunto vivo -- base da A2 e da classificacao do P4
# ---------------------------------------------------------------------------

def alive_intervals(ev):
    """{node_id: [(t_morte, t_volta_ou_inf), ...]} a partir dos eventos."""
    out = {}
    types = ev["event_type"].astype(str)
    starts = ev[types == "failure_start"][["timestamp", "node_id"]].to_numpy(float)
    ends = ev[types == "failure_end"][["timestamp", "node_id"]].to_numpy(float)
    by_end = {}
    for t, nid in ends:
        by_end.setdefault(int(nid), []).append(float(t))
    for lst in by_end.values():
        lst.sort()
    for t, nid in sorted(starts, key=lambda r: r[0]):
        nid = int(nid)
        cands = [e for e in by_end.get(nid, []) if e > t]
        out.setdefault(nid, []).append((float(t), float(cands[0]) if cands else float("inf")))
    return out


def dead_at(intervals, node_id, t):
    return any(a <= t < b for a, b in intervals.get(int(node_id), []))


def true_alive_series(ev, timestamps, n_agents):
    """Numero VERDADEIRO de vivos em cada instante, dos eventos."""
    iv = alive_intervals(ev)
    out = np.full(len(timestamps), float(n_agents))
    for spans in iv.values():
        for a, b in spans:
            out[(timestamps >= a) & (timestamps < b)] -= 1.0
    return out


# ---------------------------------------------------------------------------
# assercoes
# ---------------------------------------------------------------------------

def assert_cell(label, run_dir, stdout, fd_timeout):
    marker = parse_comm_marker(stdout)
    if marker is None:
        raise AssertionFailed(f"{label}: main.py nao emitiu a linha '[comm] role_aware=1'.")
    if not marker["differs"]:
        raise AssertionFailed(f"{label}: A1 violada -- matriz efetiva == default.")
    roles = marker["roles"]
    if int(roles.get("unknown", 0)):
        raise AssertionFailed(f"{label}: A3 violada -- {roles['unknown']} no(s) 'unknown'. roles={roles}")
    if int(roles.get("target", 0)) != 1 or int(roles.get("agent", 0)) != N:
        raise AssertionFailed(f"{label}: A3 violada -- esperado 1 target + {N} agents, veio {roles}")

    tgt = os.path.join(run_dir, "target_telemetry.csv")
    evp = os.path.join(run_dir, "events.csv")
    if not os.path.exists(tgt) or not os.path.exists(evp):
        raise AssertionFailed(f"{label}: A2 nao verificavel -- falta target_telemetry.csv ou events.csv.")
    tel = pd.read_csv(tgt, usecols=["timestamp", "alive_count"])
    ev = pd.read_csv(evp)
    t = tel["timestamp"].to_numpy(float)
    seen = tel["alive_count"].to_numpy(float)
    truth = true_alive_series(ev, t, N)

    # TOLERANCIA numerica, fixada no pre-registro: timeout + 2*dt.
    tol_s = float(fd_timeout) + 2.0 * DT
    # O alvo atrasa em relacao a verdade nos DOIS sentidos, e a tolerancia tem de
    # ser bilateral: depois de uma morte ele ainda conta o morto ate o timeout
    # expirar (ve MAIS que a verdade), e depois de um retorno leva um instante
    # para ouvi-lo (ve MENOS). Comparar o excesso contra a verdade INSTANTANEA,
    # como na primeira versao, aborta na primeira celula por latencia normal --
    # foi o que a fumaca pegou (alvo 22, verdade 20, com tol=0.35s).
    # Piso e teto defasados na janela [t-tol, t]:
    lo, hi = np.copy(truth), np.copy(truth)
    for i, ti in enumerate(t):
        w = (t >= ti - tol_s) & (t <= ti)
        if w.any():
            lo[i], hi[i] = truth[w].min(), truth[w].max()
    mask = t >= max(STEADY_T0, tol_s)
    dmax = emax = float("nan")
    if mask.any():
        deficit = lo[mask] - seen[mask]     # abaixo do piso defasado = perdeu vivo
        excess = seen[mask] - hi[mask]      # acima do teto defasado = conta morto ha tempo demais
        dmax, emax = float(np.nanmax(deficit)), float(np.nanmax(excess))
        if dmax > 0.5:
            i = int(np.nanargmax(deficit))
            raise AssertionFailed(
                f"{label}: A2 violada -- alvo viu {seen[mask][i]:.0f} vivos, piso da verdade na "
                f"tolerancia de {tol_s:.2f}s era {lo[mask][i]:.0f}, em t={t[mask][i]:.2f}s. "
                "Perda de uplink, nao latencia de deteccao.")
        if emax > 0.5:
            i = int(np.nanargmax(excess))
            raise AssertionFailed(
                f"{label}: A2 violada -- alvo viu {seen[mask][i]:.0f} vivos, teto da verdade na "
                f"tolerancia de {tol_s:.2f}s era {hi[mask][i]:.0f}, em t={t[mask][i]:.2f}s. "
                "Morto contado por mais tempo que o timeout permite.")
    return {"assert_alive_deficit_max": dmax, "assert_alive_excess_max": emax,
            "assert_tol_s": tol_s}


# ---------------------------------------------------------------------------
# classificacao das ENTRADAs (P4)
# ---------------------------------------------------------------------------

def classify_entradas(ev, fd_timeout, n_agents):
    """Espurias vs legitimas, por IDENTIDADE (primaria) e por CONTAGEM (sensibilidade)."""
    types = ev["event_type"].astype(str)
    landed = ev[types.str.startswith("dual_pulse_event_completed")
                | types.str.startswith("dual_pulse_self_shift")]
    out = {}
    if not len(landed) or "event_id" not in landed.columns:
        return {f"ent_{k}": 0 for k in ("total", "legit_id", "spur_id", "legit_cnt", "spur_cnt")}

    iv = alive_intervals(ev)
    ends = ev[types == "failure_end"][["timestamp", "node_id"]].to_numpy(float)
    truth_t = np.sort(np.concatenate([
        ev[types == "failure_start"]["timestamp"].to_numpy(float),
        ev[types == "failure_end"]["timestamp"].to_numpy(float)])) if len(ev) else np.array([])

    ent = landed[landed["event_type"].astype(str).str.endswith("entrada")]
    first_t, origin = {}, {}
    for _, r in ent.iterrows():
        eid = str(r["event_id"])
        t = float(r["timestamp"])
        if eid not in first_t or t < first_t[eid]:
            first_t[eid] = t
        parts = eid.split("_")
        if len(parts) == 2 and parts[0].isdigit():
            origin[eid] = int(parts[0])

    for wmul, suf in ((1.0, ""), (2.0, "_2w")):
        W = (float(fd_timeout) + 0.5) * wmul
        legit_id = spur_id = legit_cnt = spur_cnt = 0
        for eid, t in first_t.items():
            o = origin.get(eid)
            # --- PRIMARIA: identidade do sucessor nominal vivo em t
            ok_id = False
            if o is not None:
                for step in range(1, n_agents + 1):
                    cand = 2 + ((o - 2 + step) % n_agents)
                    if not dead_at(iv, cand, t):
                        ok_id = any(cand == int(nid) and t - W <= te <= t for te, nid in ends)
                        break
            if ok_id:
                legit_id += 1
            else:
                spur_id += 1
            # --- SECUNDARIA: a contagem de vivos subiu na janela?
            grew = any(t - W <= te <= t for te, _ in ends)
            if grew:
                legit_cnt += 1
            else:
                spur_cnt += 1
        out.update({f"ent_total{suf}": len(first_t),
                    f"ent_legit_id{suf}": legit_id, f"ent_spur_id{suf}": spur_id,
                    f"ent_legit_cnt{suf}": legit_cnt, f"ent_spur_cnt{suf}": spur_cnt,
                    f"ent_W{suf}": W})
    return out


# ---------------------------------------------------------------------------
# metricas por celula
# ---------------------------------------------------------------------------

def close_time_after_last_event(t, g, thr, t_last):
    """Instante, apos o ULTIMO evento, em que G_max para DE VEZ de exceder thr.

    inf = censurado (o criterio nunca e' atingido dentro do budget). Sob churn
    isso e' o caso esperado, e a taxa de censura E' o resultado, nao um defeito.
    """
    m = t >= t_last
    if not m.any():
        return float("inf")
    tt, gg = t[m], g[m]
    above = gg > thr
    if not above.any():
        return 0.0
    last = int(np.max(np.where(above)[0]))
    if last >= gg.size - 1:
        return float("inf")
    return float(tt[last + 1] - t_last)


def metrics_from_run(run_dir, fd_timeout):
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    evp = os.path.join(run_dir, "events.csv")
    if not os.path.exists(tgt):
        return {}
    tel = pd.read_csv(tgt, usecols=["timestamp", "G_max", "E_gap", "alive_count", "gap_max_rad"])
    ev = pd.read_csv(evp) if os.path.exists(evp) else pd.DataFrame(columns=["event_type"])
    t = tel["timestamp"].to_numpy(float)
    g = tel["G_max"].to_numpy(float)
    e = tel["E_gap"].to_numpy(float)

    steady = t >= STEADY_T0
    m = {}
    if steady.any():
        gs, es = g[steady], e[steady]
        m.update({
            "egap_mean_steady20": float(np.nanmean(es)),
            "egap_p90_steady20": float(np.nanpercentile(es, 90)),
            "egap_max_steady20": float(np.nanmax(es)),
            "gmax_mean_steady20": float(np.nanmean(gs)),
            "gmax_max_steady20": float(np.nanmax(gs)),
            # fracao do tempo em brecha: denominador identico entre celulas
            # (mesmo budget, mesmo dt), logo comparavel sem normalizacao extra.
            "frac_gmax_gt125_steady20": float(np.nanmean(gs > THR_PRIMARY)),
            "frac_gmax_gt150_steady20": float(np.nanmean(gs > THR_WIDE)),
            "n_samples_steady20": int(steady.sum()),
        })

    types = ev["event_type"].astype(str) if len(ev) else pd.Series(dtype=str)
    fs = ev[types == "failure_start"]["timestamp"].to_numpy(float) if len(ev) else np.array([])
    t_last = float(fs.max()) if fs.size else STEADY_T0
    m["t_last_failure"] = t_last
    m["t_close_125_after_last"] = close_time_after_last_event(t, g, THR_PRIMARY, t_last)
    m["t_close_110_after_last"] = close_time_after_last_event(t, g, THR_STRICT, t_last)
    m["censored_125"] = not np.isfinite(m["t_close_125_after_last"])
    m["censored_110"] = not np.isfinite(m["t_close_110_after_last"])

    if len(ev):
        for name in ("failure_start", "failure_end", "pulse_injected",
                     "dual_pulse_event_completed_saida", "dual_pulse_event_completed_entrada",
                     "dual_pulse_self_shift_saida", "dual_pulse_self_shift_entrada"):
            m[f"n_{name}"] = int((types == name).sum())
        landed = ev[types.str.startswith("dual_pulse_event_completed")
                    | types.str.startswith("dual_pulse_self_shift")]
        ids = landed["event_id"].dropna().astype(str) if "event_id" in landed.columns else pd.Series(dtype=str)
        m["n_landed_events"] = int(ids.nunique())
        seqs = [int(i.split("_")[1]) for i in ids.unique()
                if len(i.split("_")) == 2 and i.split("_")[1].isdigit()]
        m["dp_seq_max"] = max(seqs) if seqs else 0
        m.update(classify_entradas(ev, fd_timeout, N))
    return m


# ---------------------------------------------------------------------------
# execucao de uma celula
# ---------------------------------------------------------------------------

def run_cell(method, rng_aa, fd_timeout, seed):
    is_b2 = (method == "dual_pulse")
    tag = "B2" if is_b2 else "baseline"
    label = f"{tag} aa={rng_aa:g} fd={fd_timeout:g} s={seed}"
    run_dir = os.path.join(RUNS_DIR, f"{method}_aa{rng_aa:g}_fd{fd_timeout:g}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)

    per_agent = RATE_TOTAL / float(N)   # convencao da campanha: taxas sao TOTAIS
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": method, "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(N), "ENCIRCLEMENT_RADIUS": f"{R_ENC:g}",
        "SIM_DURATION": f"{T0 + BUDGET:g}", "CONTROL_PERIOD": f"{DT:g}",
        "AGENT_STATE_TIMEOUT": f"{fd_timeout:g}",
        "K_E_TAU": f"{250.0 / N:.6f}", "VM_MAX_SPEED_XY": f"{VMAX:g}",
        "EXPERIMENT_SEED": str(seed), "EXPERIMENT_REPRODUCIBLE": "True",
        "METRICS_T0": "0.0", "VM_TAU_XY": f"{TAU:g}",
        # --- eixo varrido
        "COMM_ROLE_AWARE_RANGES": "True",
        "COMM_RANGE_AGENT_AGENT": f"{rng_aa:g}",
        "COMM_RANGE_AGENT_TARGET": f"{UPLINK:g}",
        "COMMUNICATION_TRANSMISSION_RANGE": f"{UPLINK:g}",
        # --- churn (sem falha determinista)
        "DETERMINISTIC_FAILURE_ENABLE": "False",
        "FAILURE_ENABLE": "True",
        "FAILURE_MEAN_FAILURES_PER_MIN": f"{per_agent:.6f}",
        "FAILURE_OFF_TIME": f"{OFF:g}",
        # --- canal ideal fora do alcance, cenario estatico
        "COMMUNICATION_DELAY": "0.0", "COMMUNICATION_FAILURE_RATE": "0.0",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:g}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    else:
        env.pop("DUAL_PULSE_INTEGRATION", None)

    print(f"  -> {tag:8s} aa={rng_aa:<5g} (c={c_units(rng_aa):.2f}) fd={fd_timeout:<5g} s={seed} ...",
          end="", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")

    checks = assert_cell(label, run_dir, proc.stdout, fd_timeout)   # aborta o sweep

    m = metrics_from_run(run_dir, fd_timeout)
    if not m:
        print(f" FALHOU (rc={proc.returncode})")
        return None, None

    agent_csv = os.path.join(run_dir, "agent_telemetry.csv")
    m.update(effort_metrics(agent_csv, t0=STEADY_T0, vmax=VMAX))
    try:
        os.remove(agent_csv)
    except OSError:
        pass

    m.update(checks)
    m.update({"method": tag, "N": N, "radius": R_ENC, "range_aa": rng_aa, "range_at": UPLINK,
              "c_hops": c_units(rng_aa), "c_hops_post": c_units_post(rng_aa),
              "fd_timeout": fd_timeout, "rate_total": RATE_TOTAL, "off_time": OFF,
              "tau_xy": TAU, "seed": seed, "dt": DT, "budget": BUDGET})
    m.update(provenance_with_retry(run_dir))

    ev_rows = per_event_rows(run_dir, tag, rng_aa, fd_timeout, seed)
    nan = float("nan")
    print(f" egap={m.get('egap_mean_steady20', nan):.4f}"
          f"  fr125={m.get('frac_gmax_gt125_steady20', nan):.2f}"
          f"  cens125={m.get('censored_125')}"
          f"  ENTesp/leg={m.get('ent_spur_id', 0)}/{m.get('ent_legit_id', 0)}"
          f"  ev={len(ev_rows)}")
    return m, ev_rows


def per_event_rows(run_dir, tag, rng_aa, fd_timeout, seed):
    """Pico por evento via analysis_churn.aggregate_gmax.event_rows.

    Reuso deliberado: aquela definicao tem pre-registro, janela adaptativa e SE
    agrupado proprios. Redefinir aqui criaria uma segunda definicao de "pico por
    evento" -- exatamente o erro que EGAP_HOMONIMO.md documenta para egap_avg.
    """
    try:
        from aggregate_gmax import event_rows
    except Exception:
        return []
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    evp = os.path.join(run_dir, "events.csv")
    if not (os.path.exists(tgt) and os.path.exists(evp)):
        return []
    try:
        tel = pd.read_csv(tgt, usecols=["timestamp", "G_max", "E_gap", "alive_count", "gap_max_rad"])
        ev = pd.read_csv(evp)
        rows = event_rows(tel, ev, tag, RATE_TOTAL, seed)
    except Exception as exc:
        print(f"\n     [aviso] event_rows falhou em {run_dir}: {exc}")
        return []
    for r in rows:
        r.update({"range_aa": rng_aa, "c_hops": c_units(rng_aa), "fd_timeout": fd_timeout})
    return rows


# ---------------------------------------------------------------------------
# grade, relatorio, main
# ---------------------------------------------------------------------------

def cells():
    out = []
    for rng in RANGES:
        for fd in timeouts_for(rng):
            for seed in SEEDS:
                for meth in METHODS:
                    out.append((meth, rng, fd, seed))
    return out


def print_grid():
    grid = cells()
    thr = 2.0 * np.cos(np.pi / N)
    print(f"Fase 8a (ii): churn sob localidade. N={N}, R={R_ENC:g} m, uplink={UPLINK:g} m")
    print(f"  corda 1 salto = {chord(1, n=N, radius=R_ENC):.3f} m | "
          f"2 saltos = {chord(2, n=N, radius=R_ENC):.3f} m | limiar uniforme 2cos(pi/N) = {thr:.4f}")
    print(f"  churn = {RATE_TOTAL:g}/min TOTAL ({RATE_TOTAL / N:.3f}/min por agente), "
          f"recuperacao {OFF:g}s, budget {BUDGET:g}s")
    print(f"  metodos={METHODS}  sementes={SEEDS}  dt={DT:g}  tau_a={TAU:g}")
    print(f"\n  {'range':>7} {'c_pre':>6} {'c_pos':>6} {'timeouts':>14} {'celulas':>8}")
    for rng in RANGES:
        fds = timeouts_for(rng)
        print(f"  {rng:>7g} {c_units(rng):>6.2f} {c_units_post(rng):>6.2f} "
              f"{','.join(f'{f:g}' for f in fds):>14} {len(fds) * len(SEEDS) * len(METHODS):>8}")
    print(f"\n  TOTAL = {len(grid)} celulas")
    allfd = sorted(set(TIMEOUTS_LOW + TIMEOUTS_HIGH))
    print("  A2 tolerancia = timeout + 2*dt = "
          + ", ".join(f"{f + 2 * DT:.2f}s (fd={f:g})" for f in allfd))
    print("  P4 W = timeout + 0.5s = "
          + ", ".join(f"{f + 0.5:.2f}s (fd={f:g})" for f in allfd) + "  (+ sensibilidade em 2W)\n")


def _key(r):
    return (str(r["method"]), round(float(r["range_aa"]), 6),
            round(float(r["fd_timeout"]), 6), int(r["seed"]))


def iqr(series):
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    f = s[np.isfinite(s)]
    if f.empty:
        return None
    return float(f.median()), float(f.quantile(.25)), float(f.quantile(.75)), len(f)


def fmt(series, prec=4):
    q = iqr(series)
    return "n.a." if q is None else f"{q[0]:.{prec}f} [{q[1]:.{prec}f},{q[2]:.{prec}f}]"


def _med(g, col):
    return float(pd.to_numeric(g[col], errors="coerce").median()) if col in g else float("nan")


def _sum(g, col):
    return float(pd.to_numeric(g[col], errors="coerce").sum()) if col in g else float("nan")


def report(df, evdf):
    thr = 2.0 * np.cos(np.pi / N)
    print("\n" + "=" * 100)
    print("FASE 8a (ii) -- CHURN SOB LOCALIDADE")
    print("=" * 100)
    print(f"  limiar uniforme das fases (i)/(i-b): 2cos(pi/N) = {thr:.4f}")
    print(f"  P3: c=2.0 esta {(c_units(10.4) / thr - 1) * 100:.1f}% acima desse limiar\n")

    for metric, prec, title in (
        ("egap_mean_steady20", 4, "egap_mean_steady20 (PRIMARIA; t>=20s ate o fim)"),
        ("frac_gmax_gt125_steady20", 3, "fracao do tempo com G_max > 1.25 (t>=20s)"),
        ("frac_gmax_gt150_steady20", 3, "fracao do tempo com G_max > 1.50 (t>=20s)"),
        ("gmax_max_steady20", 3, "G_max maximo em regime"),
    ):
        print(f"=== {title}: mediana [IQR] ===")
        print(f"{'range':>7} {'c':>5} {'fd':>5} | {'baseline':>26} | {'B2':>26}")
        for rng in sorted(df.range_aa.unique()):
            for fd in sorted(df[df.range_aa == rng].fd_timeout.unique()):
                a = fmt(df[(df.range_aa == rng) & (df.fd_timeout == fd) & (df.method == "baseline")][metric], prec)
                b = fmt(df[(df.range_aa == rng) & (df.fd_timeout == fd) & (df.method == "B2")][metric], prec)
                print(f"{rng:>7g} {c_units(rng):>5.2f} {fd:>5g} | {a:>26} | {b:>26}")
        print()

    print("=== CENSURA por criterio (medida, nao apendice) ===")
    print("  censurado = o criterio nunca e' atingido apos o ULTIMO evento, dentro do budget")
    print(f"{'range':>7} {'c':>5} {'fd':>5} {'metodo':>9} | {'cens 1.25':>12} | {'cens 1.10':>12}")
    for rng in sorted(df.range_aa.unique()):
        for fd in sorted(df[df.range_aa == rng].fd_timeout.unique()):
            for meth in ("baseline", "B2"):
                g = df[(df.range_aa == rng) & (df.fd_timeout == fd) & (df.method == meth)]
                if not len(g):
                    continue
                c125 = int(g.censored_125.astype(bool).sum())
                c110 = int(g.censored_110.astype(bool).sum())
                print(f"{rng:>7g} {c_units(rng):>5.2f} {fd:>5g} {meth:>9} | "
                      f"{c125:>4}/{len(g):<7} | {c110:>4}/{len(g):<7}")

    print("\n=== eventos por tipo (mediana por celula, B2) ===")
    print(f"{'range':>7} {'c':>5} {'fd':>5} | {'mortes':>7} {'retornos':>9} | "
          f"{'SAIDAcompl':>11} {'ENTRADAcompl':>13} | {'landed':>7} {'seq_max':>8}")
    for rng in sorted(df.range_aa.unique()):
        for fd in sorted(df[df.range_aa == rng].fd_timeout.unique()):
            g = df[(df.range_aa == rng) & (df.fd_timeout == fd) & (df.method == "B2")]
            if not len(g):
                continue
            print(f"{rng:>7g} {c_units(rng):>5.2f} {fd:>5g} | {_med(g, 'n_failure_start'):>7.0f} "
                  f"{_med(g, 'n_failure_end'):>9.0f} | "
                  f"{_med(g, 'n_dual_pulse_event_completed_saida'):>11.0f} "
                  f"{_med(g, 'n_dual_pulse_event_completed_entrada'):>13.0f} | "
                  f"{_med(g, 'n_landed_events'):>7.0f} {_med(g, 'dp_seq_max'):>8.0f}")

    print("\n=== P4: ENTRADAs espurias / legitimas (B2, soma sobre as sementes) ===")
    print("  IDENT = primaria (sucessor nominal vivo do originador tem failure_end na janela)")
    print("  CONT  = sensibilidade por contagem; divergencia IDENT-CONT = taxa de coincidencia")
    print(f"{'range':>7} {'c':>5} {'fd':>5} | {'IDENT esp/leg':>16} {'razao':>7} | "
          f"{'CONT esp/leg':>15} | {'IDENT 2W esp/leg':>17}")
    for rng in sorted(df.range_aa.unique()):
        for fd in sorted(df[df.range_aa == rng].fd_timeout.unique()):
            g = df[(df.range_aa == rng) & (df.fd_timeout == fd) & (df.method == "B2")]
            if not len(g):
                continue
            esp, leg = _sum(g, "ent_spur_id"), _sum(g, "ent_legit_id")
            ratio = (esp / leg) if leg else float("inf")
            print(f"{rng:>7g} {c_units(rng):>5.2f} {fd:>5g} | {esp:>7.0f}/{leg:<8.0f} {ratio:>7.2f} | "
                  f"{_sum(g, 'ent_spur_cnt'):>6.0f}/{_sum(g, 'ent_legit_cnt'):<8.0f} | "
                  f"{_sum(g, 'ent_spur_id_2w'):>8.0f}/{_sum(g, 'ent_legit_id_2w'):<8.0f}")

    if evdf is not None and len(evdf):
        print("\n=== pico de G_max POR EVENTO (aggregate_gmax.event_rows, janela adaptativa) ===")
        print("  REGRA 'contar observacoes independentes, nao linhas': n_ev sao EVENTOS, e eventos")
        print("  da mesma rodada NAO sao independentes -- o estado pre de um e' o pos do anterior.")
        print("  O n independente e' n_run (sementes). O IQR abaixo e' sobre eventos; qualquer")
        print("  inferencia tem de agrupar por rodada (e' o que aggregate_gmax.fit ja faz).")
        print(f"{'range':>7} {'c':>5} {'fd':>5} | {'baseline':>26} | {'B2':>26} | {'n_ev':>6} {'n_run':>6}")
        for rng in sorted(evdf.range_aa.unique()):
            for fd in sorted(evdf[evdf.range_aa == rng].fd_timeout.unique()):
                sel = evdf[(evdf.range_aa == rng) & (evdf.fd_timeout == fd)]
                cb = fmt(sel[sel.method == "baseline"]["gmax_peak"], 3)
                co = fmt(sel[sel.method == "B2"]["gmax_peak"], 3)
                n_run = sel.groupby(["method", "seed"]).ngroups if "seed" in sel else 0
                print(f"{rng:>7g} {c_units(rng):>5.2f} {fd:>5g} | {cb:>26} | {co:>26} | "
                      f"{len(sel):>6} {n_run:>6}")
        # Os picos podem colapsar em poucos valores distintos: o pico precede
        # qualquer resposta e e' invariante ao metodo. Se isso acontecer, o n
        # efetivo e' o numero de valores distintos, nao o de linhas.
        nd = evdf.groupby(["range_aa", "fd_timeout"])["gmax_peak"].nunique()
        print(f"  valores DISTINTOS de gmax_peak por celula: {dict(nd)}")

    if "git_dirty" in df.columns:
        vals = sorted({"SEM MANIFESTO" if pd.isna(v) else str(v) for v in df.git_dirty})
        print(f"\ngit_dirty nas linhas: {vals}")


def main():
    print_grid()
    if DRY_RUN:
        print("CCS_DRY_RUN ligado: grade impressa, nada simulado.")
        return

    os.makedirs(RUNS_DIR, exist_ok=True)
    store, ev_all = {}, []
    for src in (RESULTS_CSV, WORK_CSV):
        if os.path.exists(src):
            try:
                for r in pd.read_csv(src).to_dict("records"):
                    store[_key(r)] = r
            except Exception:
                pass
    if os.path.exists(EVENTS_CSV):
        try:
            ev_all = pd.read_csv(EVENTS_CSV).to_dict("records")
        except Exception:
            ev_all = []
    print(f"{len(store)} celulas ja no CSV (merge incremental)\n")

    grid = cells()
    done = 0
    for (meth, rng, fd, seed) in grid:
        k = ("B2" if meth == "dual_pulse" else "baseline", round(rng, 6), round(fd, 6), seed)
        if k in store:
            continue
        r, evr = run_cell(meth, rng, fd, seed)
        done += 1
        if r:
            store[_key(r)] = r
            ev_all.extend(evr or [])
            pd.DataFrame(list(store.values())).to_csv(WORK_CSV, index=False)
        if done % PROGRESS_EVERY == 0:
            print(f"[progress] {done}/{len(grid)} celulas | ultimo: aa={rng:g} fd={fd:g}", flush=True)

    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados.")
        return
    evdf = pd.DataFrame(ev_all) if ev_all else None
    report(df, evdf)

    df.to_csv(RESULTS_CSV, index=False)
    if evdf is not None and len(evdf):
        evdf.to_csv(EVENTS_CSV, index=False)
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except AssertionFailed as exc:
        print(f"\n*** SWEEP ABORTADO -- {exc}", file=sys.stderr)
        sys.exit(2)
