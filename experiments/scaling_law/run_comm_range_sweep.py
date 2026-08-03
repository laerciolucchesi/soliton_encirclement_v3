#!/usr/bin/env python
"""Fase 8a (i) -- alcance FINITO do anel, com o uplink ao alvo preservado.

Varre COMM_RANGE_AGENT_AGENT (o link agente-agente) mantendo
COMM_RANGE_AGENT_TARGET fixo em 200 m, para baseline e B2, N e tau_a fixos,
um unico obito determinista, sementes pareadas (a mesma semente escolhe a mesma
vitima nos dois metodos).

POR QUE OS DOIS ALCANCES SAO SEPARADOS. O AgentState e' UM broadcast servindo
duas plateias com exigencias opostas: os vizinhos do anel (que queremos curtos)
e o alvo, que precisa ouvir TODO agente -- quem ele deixa de ouvir por mais de
AGENT_STATE_TIMEOUT e' podado de agent_states/alive_lambdas, ou seja, declarado
morto estando vivo. Isso corrompe alive_count, o mapa de lambdas devolvido aos
agentes e M1..M7 -- em silencio, porque G_max/E_gap normalizam pelo numero de
agentes OUVIDOS (meio anel bem distribuido pontua 1.0 igual a um anel inteiro).
Com um alcance global unico, a faixa interessante (abaixo de R = 20 m, onde o
alvo deixa de ouvir o anel) e' inobservavel: o aparelho de medida morre junto
com o fenomeno. Ver comm_role_aware.py e config_param secao 2.

AS TRES ASSERCOES (abortam o sweep inteiro, nao apenas a celula: um invariante
violado significa que a linha e' inconfiavel, e seguir queima horas gerando
lixo). Duas leem a linha "[comm] ..." que o main.py emite depois do build(), a
terceira le a telemetria do alvo:
  A1  matriz efetiva != default        -- pega o gate ligado com os alcances
                                          esquecidos em 200 (rodada no-op com
                                          rotulo de role-aware)
  A2  alive_count >= N-1 apos o warmup -- pega a corrupcao silenciosa do uplink
  A3  papeis = 1 target + N agents,    -- pega classe de protocolo fora do mapa
      zero unknown                        de papeis (seus links cairiam no
                                          default sem aviso)

Metricas: as MESMAS de run_breach_window.py (gmax_peak, t_close_125/110,
breach_area, egap_final), para que esta fase seja comparavel com a campanha de
breach ja publicada, em vez de introduzir uma definicao concorrente.

Uso:
    python experiments/scaling_law/run_comm_range_sweep.py
    # env: CRS_RANGES="6.3,8.4,10.4,15.7,26.1"  CRS_UPLINK="200"
    #      CRS_N="24"  CRS_SEEDS="0,1,2,3,4,5,6,7"  CRS_BUDGET="90"
    #      CRS_METHODS="baseline,dual_pulse"  CRS_TAU="1.0"  CRS_TAG=""
    #      CRS_DRY_RUN="1"   (imprime a grade e sai, sem simular)
"""
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_util import run_provenance  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("CRS_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "comm_range_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "comm_range_results" + _SUF + ".csv")
# O CSV incremental vive DENTRO de RUNS_DIR, que casa com "*_runs/" no
# .gitignore, e so' e' copiado para RESULTS_CSV no fim.
#
# Nao e' cosmetica: se o parcial ficar no EXP_DIR ele nasce nao-rastreado, e o
# git status deixa de estar limpo a partir da celula 2 -- cada filho seguinte
# grava git_dirty=True no proprio manifesto. E' o que aconteceu com a campanha
# de breach: breach_window_results_v*.csv tem exatamente 1 linha dirty=False (a
# primeira) e 29 dirty=True. Escrevendo o parcial num caminho ignorado, as 80
# linhas nascem dirty=False.
WORK_CSV = os.path.join(RUNS_DIR, "_partial_results.csv")

RANGES = [float(x) for x in os.environ.get("CRS_RANGES", "6.3,8.4,10.4,15.7,26.1").split(",") if x.strip()]
UPLINK = float(os.environ.get("CRS_UPLINK", "200"))
N = int(os.environ.get("CRS_N", "24"))
R_ENC = float(os.environ.get("CRS_RADIUS", "20"))
SEEDS = [int(x) for x in os.environ.get("CRS_SEEDS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
METHODS = [m.strip() for m in os.environ.get("CRS_METHODS", "baseline,dual_pulse").split(",") if m.strip()]
TAU = float(os.environ.get("CRS_TAU", "1.0"))
BUDGET = float(os.environ.get("CRS_BUDGET", "90"))
DT = float(os.environ.get("CONTROL_PERIOD", "0.05"))
T_FAIL = float(os.environ.get("CRS_T_FAIL", "5.0"))
DRY_RUN = os.environ.get("CRS_DRY_RUN", "").strip().lower() in ("1", "true", "yes", "y")

# Mesmos limiares do run_breach_window: 1.25 = primario, 1.10 = estrito.
THR_PRIMARY = 1.25
THR_STRICT = 1.10
# Janela em que os gatilhos de pulso ficam silenciados (FAST_CHANNEL_WARMUP_SEC).
# A A2 so vale DEPOIS dela: antes, alive_count ainda esta subindo de 0 conforme
# os primeiros AgentState chegam, e exigir N ali seria um falso positivo.
WARMUP = float(os.environ.get("FAST_CHANNEL_WARMUP_SEC", "1.0"))

_MARKER_RE = re.compile(
    r"\[comm\]\s+role_aware=1\s+roles=\{(?P<roles>[^}]*)\}\s+"
    r"matrix=\{(?P<matrix>[^}]*)\}\s+default=(?P<default>[-\d.eE+]+)\s+"
    r"differs=(?P<differs>[01])"
)


def chord(k_hops, n=None, radius=None):
    """Distancia entre duas posicoes do anel separadas por k saltos."""
    n = N if n is None else n
    radius = R_ENC if radius is None else radius
    return 2.0 * radius * math.sin(k_hops * math.pi / n)


def c_units(transmission_range):
    """Alcance em unidades do acorde de 1 salto -- a escala em que o penhasco vive."""
    return float(transmission_range) / chord(1)


def victim_node_id(n, seed=0):
    """Mesma regra dos demais runners: sementes pareadas => mesma vitima nos 2 metodos."""
    return 2 + ((n // 2 + seed) % n)


class AssertionFailed(RuntimeError):
    """Invariante violado: aborta o sweep inteiro, a linha nao e' confiavel."""


def parse_comm_marker(stdout):
    """Le a linha que o main.py emite depois do build(). None se ausente."""
    for line in (stdout or "").splitlines():
        m = _MARKER_RE.search(line)
        if m:
            roles = {}
            for tok in m.group("roles").split(","):
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    roles[k.strip()] = int(v)
            matrix = {}
            for tok in m.group("matrix").split(","):
                if ":" in tok:
                    k, v = tok.rsplit(":", 1)
                    matrix[k.strip()] = float(v)
            return {"roles": roles, "matrix": matrix,
                    "default": float(m.group("default")),
                    "differs": m.group("differs") == "1",
                    "line": line.strip()}
    return None


def assert_cell(label, run_dir, stdout):
    """As tres asserces. Levanta AssertionFailed na primeira violacao."""
    marker = parse_comm_marker(stdout)
    if marker is None:
        raise AssertionFailed(
            f"{label}: main.py nao emitiu a linha '[comm] role_aware=1'. "
            "O gate COMM_ROLE_AWARE_RANGES nao chegou ao filho, ou a rodada morreu antes do build()."
        )

    # A1 -- a matriz efetiva difere do default.
    if not marker["differs"]:
        raise AssertionFailed(
            f"{label}: A1 violada -- matriz efetiva == default ({marker['default']:g} m). "
            f"Rodada no-op com rotulo role-aware. matrix={marker['matrix']}"
        )

    # A3 -- censo de papeis (antes da A2: se o papel nao resolveu, alive_count
    # nao significa o que se pensa que significa).
    roles = marker["roles"]
    unknown = int(roles.get("unknown", 0))
    n_target = int(roles.get("target", 0))
    n_agent = int(roles.get("agent", 0))
    if unknown:
        raise AssertionFailed(
            f"{label}: A3 violada -- {unknown} no(s) com papel 'unknown'. "
            f"Seus links caem no default sem aviso. roles={roles}"
        )
    if n_target != 1 or n_agent != N:
        raise AssertionFailed(
            f"{label}: A3 violada -- esperado 1 target + {N} agents, veio "
            f"target={n_target} agent={n_agent}. roles={roles}"
        )

    # A2 -- o alvo continua ouvindo o anel inteiro depois do warmup.
    # N ate a falha, N-1 depois dela; o limite unico N-1 cobre as duas fases.
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        raise AssertionFailed(f"{label}: A2 nao verificavel -- {tgt} nao existe.")
    df = pd.read_csv(tgt, usecols=["timestamp", "alive_count"])
    post = df[df["timestamp"] >= WARMUP]
    if post.empty:
        raise AssertionFailed(f"{label}: A2 nao verificavel -- sem amostras apos o warmup ({WARMUP:g}s).")
    alive = post["alive_count"].to_numpy(float)
    floor = float(np.nanmin(alive))
    if floor < N - 1:
        t_bad = float(post["timestamp"].to_numpy(float)[int(np.nanargmin(alive))])
        raise AssertionFailed(
            f"{label}: A2 violada -- alive_count caiu a {floor:.0f} (< N-1 = {N - 1}) em t={t_bad:.2f}s. "
            "O alvo perdeu agentes vivos: uplink corrompido, as metricas desta linha sao ficcao."
        )
    return {"assert_alive_min": floor, "assert_roles_agent": n_agent,
            "assert_matrix_differs": True, "comm_marker": marker["line"]}


def close_time(t, g, thr):
    """Instante em que G_max para DE VEZ de exceder thr (nao a primeira descida)."""
    above = g > thr
    if not above.any():
        return 0.0
    last = int(np.max(np.where(above)[0]))
    if last >= g.size - 1:
        return float("inf")
    return float(t[last + 1] - t[0])


def failure_time(run_dir, fallback):
    p = os.path.join(run_dir, "events.csv")
    if not os.path.exists(p):
        return fallback
    try:
        ev = pd.read_csv(p)
    except Exception:
        return fallback
    if "event_type" not in ev.columns:
        return fallback
    f = ev[ev["event_type"] == "failure_start"]
    return float(f["timestamp"].min()) if len(f) else fallback


def metrics_from_run(run_dir):
    tgt = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(tgt):
        return {}
    cols = ["timestamp", "G_max", "E_gap", "alive_count", "gap_max_rad"]
    try:
        df = pd.read_csv(tgt, usecols=cols)
    except ValueError:
        return {"error": "telemetria sem alive_count/gap_max_rad"}
    t_fail = failure_time(run_dir, T_FAIL)
    post = df[df["timestamp"] >= t_fail].reset_index(drop=True)
    if post.empty:
        return {}
    t = post["timestamp"].to_numpy(float)
    g = post["G_max"].to_numpy(float)
    rad = post["gap_max_rad"].to_numpy(float)
    alive = post["alive_count"].to_numpy(float)

    excess = np.maximum(0.0, g - THR_PRIMARY)
    area = float(np.trapezoid(excess, t)) if t.size > 1 else 0.0
    i_pk = int(np.nanargmax(g))
    return {
        "t_fail": t_fail,
        "gmax_peak": float(g[i_pk]),
        "gap_peak_deg": float(np.degrees(rad[i_pk])),
        "alive_at_peak": float(alive[i_pk]),
        "t_close_125": close_time(t, g, THR_PRIMARY),
        "t_close_110": close_time(t, g, THR_STRICT),
        "breach_area_125": area,
        "gmax_final": float(g[-1]),
        "egap_final": float(post["E_gap"].to_numpy(float)[-1]),
        "alive_final": float(alive[-1]) if alive.size else float("nan"),
    }


def run_cell(method, rng_aa, seed):
    is_b2 = (method == "dual_pulse")
    tag = "B2" if is_b2 else "baseline"
    victim = victim_node_id(N, seed)
    label = f"{tag} range_aa={rng_aa:g} s={seed}"
    run_dir = os.path.join(RUNS_DIR, f"{method}_aa{rng_aa:g}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        # --- selecao da rodada
        "PROPAGATION_METHOD": method,
        "PROPAGATION_K_PROP": "0.0",
        # --- escala / laco (regra 3: fixar tudo que importa)
        "NUM_AGENTS": str(N),
        "ENCIRCLEMENT_RADIUS": f"{R_ENC:g}",
        "SIM_DURATION": f"{T_FAIL + BUDGET:g}",
        "CONTROL_PERIOD": f"{DT:g}",
        "AGENT_STATE_TIMEOUT": f"{5.0 * DT:g}",
        "K_E_TAU": f"{250.0 / N:.6f}",
        "EXPERIMENT_SEED": str(seed),
        "EXPERIMENT_REPRODUCIBLE": "True",
        "METRICS_T0": "0.0",
        "VM_TAU_XY": f"{TAU:g}",
        # --- O EIXO VARRIDO
        "COMM_ROLE_AWARE_RANGES": "True",
        "COMM_RANGE_AGENT_AGENT": f"{rng_aa:g}",
        "COMM_RANGE_AGENT_TARGET": f"{UPLINK:g}",
        "COMMUNICATION_TRANSMISSION_RANGE": f"{UPLINK:g}",   # demais links (adversario)
        # --- evento unico e limpo
        "DETERMINISTIC_FAILURE_ENABLE": "True",
        "DETERMINISTIC_FAILURE_AGENT_ID": str(victim),
        "DETERMINISTIC_FAILURE_TIME_T0": f"{T_FAIL:g}",
        "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
        "FAILURE_ENABLE": "True",
        "FAILURE_MEAN_FAILURES_PER_MIN": "0.0",
        # --- canal ideal fora do alcance, cenario estatico
        "COMMUNICATION_DELAY": "0.0",
        "COMMUNICATION_FAILURE_RATE": "0.0",
        "INIT_ANGLES_EQUIDISTANT": "True",
        "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        # --- saida
        "VIS_OPEN_BROWSER": "False",
        "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if is_b2:
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": f"{TAU:g}", "DUAL_PULSE_TTL_HOPS": str(3 * N)})
    else:
        env.pop("DUAL_PULSE_INTEGRATION", None)

    print(f"  -> {tag:8s} aa={rng_aa:<5g} (c={c_units(rng_aa):.2f}) s={seed} (vitima={victim}) ...",
          end="", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")

    checks = assert_cell(label, run_dir, proc.stdout)   # aborta o sweep se violar

    m = metrics_from_run(run_dir)
    if not m or "error" in m:
        print(f" FALHOU (rc={proc.returncode}) {m.get('error', '')}\n{(proc.stderr or '')[-500:]}")
        return None
    m.update(checks)
    m.update({"method": tag, "N": N, "radius": R_ENC, "range_aa": rng_aa, "range_at": UPLINK,
              "c_hops": c_units(rng_aa), "tau_xy": TAU, "seed": seed, "victim": victim,
              "agent_state_timeout": 5.0 * DT, "dt": DT})
    m.update(run_provenance(run_dir))
    tc = m["t_close_125"]
    print(f" pico={m['gmax_peak']:.3f}  t_close={'inf' if not np.isfinite(tc) else f'{tc:.2f}s'}"
          f"  alive_min={checks['assert_alive_min']:.0f}")
    return m


def _key(r):
    return (str(r["method"]), round(float(r["range_aa"]), 6), int(r["seed"]))


def iqr_row(series):
    s = pd.Series(series).astype(float)
    finite = s[np.isfinite(s)]
    if finite.empty:
        return float("nan"), float("nan"), float("nan"), 0, int((~np.isfinite(s)).sum())
    q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
    return float(finite.median()), q1, q3, len(finite), int((~np.isfinite(s)).sum())


def print_grid():
    print(f"Fase 8a (i): alcance do anel, N={N}, R={R_ENC:g} m, uplink={UPLINK:g} m")
    print(f"  acorde 1 salto = {chord(1):.3f} m | 2 saltos = {chord(2):.3f} m | 3 saltos = {chord(3):.3f} m")
    print(f"  metodos={METHODS}  sementes={SEEDS}  dt={DT:g}  tau_a={TAU:g}  budget={BUDGET:g}s")
    print(f"  celulas = {len(RANGES)} x {len(SEEDS)} x {len(METHODS)} = "
          f"{len(RANGES) * len(SEEDS) * len(METHODS)}")
    print(f"\n  {'range_aa':>9} {'c=r/acorde1':>12} {'saltos alcancados':>18}")
    for r in RANGES:
        hops = max([k for k in range(1, N // 2 + 1) if chord(k) <= r] + [0])
        print(f"  {r:>9g} {c_units(r):>12.3f} {hops:>18d}")
    print()


def report(df):
    for metric, fmt in (("gmax_peak", "{:.3f}"), ("t_close_125", "{:.2f}"), ("t_close_110", "{:.2f}")):
        print(f"\n=== {metric}: mediana [IQR] por (range, metodo) ===")
        print(f"{'range':>7} {'c':>6} | " + " | ".join(f"{m:^26}" for m in ("baseline", "B2")))
        for rng in sorted(df["range_aa"].unique()):
            cells = []
            for meth in ("baseline", "B2"):
                sub = df[(df.method == meth) & (df.range_aa == rng)][metric]
                med, q1, q3, n_ok, n_inf = iqr_row(sub)
                if n_ok == 0:
                    cells.append(f"{'todas inf/n.a.':^26}")
                else:
                    txt = f"{fmt.format(med)} [{fmt.format(q1)},{fmt.format(q3)}]"
                    if n_inf:
                        txt += f" +{n_inf}inf"
                    cells.append(f"{txt:^26}")
            print(f"{rng:>7g} {c_units(rng):>6.2f} | " + " | ".join(cells))

    print("\n=== penhasco (em unidades c = range / acorde de 1 salto) ===")
    print(f"  acorde de 1 salto a N={N}, R={R_ENC:g}: {chord(1):.3f} m")
    for meth in ("baseline", "B2"):
        ok_c, bad_c = [], []
        for rng in sorted(df["range_aa"].unique()):
            sub = df[(df.method == meth) & (df.range_aa == rng)]["t_close_125"].astype(float)
            if sub.empty:
                continue
            # "fechou" = mais da metade das sementes fecharam dentro do budget
            (ok_c if float(np.mean(np.isfinite(sub))) > 0.5 else bad_c).append(c_units(rng))
        if ok_c and bad_c:
            print(f"  {meth:>8}: fecha a partir de c={min(ok_c):.2f}; falha ate c={max(bad_c):.2f} "
                  f"-> penhasco em c ∈ ({max(bad_c):.2f}, {min(ok_c):.2f}]")
        elif ok_c:
            print(f"  {meth:>8}: fechou em todos os c varridos (>= {min(ok_c):.2f}) -- penhasco abaixo da grade")
        elif bad_c:
            print(f"  {meth:>8}: falhou em todos os c varridos (<= {max(bad_c):.2f}) -- penhasco acima da grade")


def main():
    print_grid()
    if DRY_RUN:
        print("CRS_DRY_RUN ligado: grade impressa, nada simulado.")
        return

    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    for src in (RESULTS_CSV, WORK_CSV):   # final primeiro, parcial sobrepoe
        if os.path.exists(src):
            try:
                for r in pd.read_csv(src).to_dict("records"):
                    store[_key(r)] = r
            except Exception:
                pass
    print(f"{len(store)} celulas ja no CSV (merge incremental)\n")

    for rng_aa in RANGES:
        for seed in SEEDS:
            for method in METHODS:
                if (("B2" if method == "dual_pulse" else "baseline"), round(rng_aa, 6), seed) in store:
                    continue
                r = run_cell(method, rng_aa, seed)
                if r:
                    store[_key(r)] = r
                    pd.DataFrame(list(store.values())).to_csv(WORK_CSV, index=False)

    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados.")
        return
    report(df)

    # So' agora o resultado sai para um caminho rastreado: durante o sweep a
    # arvore precisa continuar limpa para os filhos gravarem git_dirty=False.
    df.to_csv(RESULTS_CSV, index=False)
    if "git_dirty" in df.columns:
        n_dirty = int(df["git_dirty"].astype(str).str.lower().eq("true").sum())
        if n_dirty:
            print(f"\nAVISO: {n_dirty}/{len(df)} linhas com git_dirty=True -- a arvore sujou "
                  "durante o sweep (regra 1 da campanha). Verifique com 'git status'.")
        else:
            print(f"\ngit_dirty=False em todas as {len(df)} linhas.")
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except AssertionFailed as exc:
        print(f"\n*** SWEEP ABORTADO -- {exc}", file=sys.stderr)
        sys.exit(2)
