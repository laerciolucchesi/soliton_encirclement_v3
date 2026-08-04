#!/usr/bin/env python
"""Item 9 -- campanha m=2: baseline / m2 / overlay sob localidade, N in {24, 50}.

=======================================================================
PRE-REGISTRO -- escrito antes da grade; volta para revisao do usuario
ANTES de disparar (sequencia da aprovacao condicionada). Nao editar depois
de aprovado; emendas pre-analise com carimbo, como na fase 8a (ii).
=======================================================================

GRADE -- 192 celulas, sem poda (A.4 item 1):
  3 metodos {baseline, m2, dual_pulse} x 2 N {24, 50} x 2 regimes
  {obito unico, churn 12/min TOTAL} x 2 c {1.6089, 3.0071} x 8 sementes.
  Alcances por N (c * 2R sin(pi/N)): N=24 -> 8.4 / 15.7 m; N=50 -> 4.041 /
  7.553 m. Uplink 200 m. fd = 0.25 s unico (o eixo do timeout ja foi medido na
  8a-(ii)). Churn TOTAL constante entre N (mesmo fluxo de eventos por rodada).
  PROTECTION_ANGLE_DEG = 360 pinado (lambdas uniformes).
    ERRATA (pre-grade, aritmetica de config): o adendo A.4 item 4 escreveu
    "PROTECTION_ANGLE_DEG=0". O valor UNIFORME do config e' 360 (0 degenera
    para uniforme por fallback do edge_lambda, mas o pino correto -- e o que
    reproduz byte a byte as celulas da 8a-(ii), que rodaram no default 360 --
    e' 360). Corrigido aqui antes de qualquer celula rodar.
  Custo MEDIDO na calibracao de fumaca: ~45 s/celula em N=24, ~124 s em N=50
  -> 96*45 + 96*124 ~ 4.5 h.

PREDICOES (todas antes do dado):

P5 -- regime limpo (obito unico), c=3.0071: com o ganho justo do adendo A.2
  (mesma margem nominal g*dt*lambda_max), o m2 melhora o tempo de
  reconfiguracao sobre o baseline por
      (lambda2/lambda_max)_m2 / (lambda2/lambda_max)_m1
  = 3.1565 em N=24 e 3.1970 em N=50 (autovalores discretos, pinados em
  tests/test_m2_law.py). Metrica primaria: t_settle (event_metrics, a primaria
  da campanha); tau_fit secundaria com o R2 reportado. A predicao e' DERIVADA,
  nao ajustada -- mesmo estilo do 2cos(pi/N) da 8a.

P6 -- churn, c=3.0071: ordenacao prevista overlay >= m2 > baseline em
  egap_mean_steady20. A 8a-(ii) mediu overlay 1.19x sobre o baseline neste
  ponto; se o m2 cair entre 1.19x e 1.00x, a vantagem do overlay sobre a
  densificacao passiva e' o numero que falta ao texto.
  DESFECHO ADVERSO, registrado com significado (A.4 item 3): se m2 SUPERAR o
  overlay, a comparacao de alcance igual ENFRAQUECE o argumento do 4.1 -- o
  overlay perderia para densificacao passiva no mesmo raio de radio. Resultado
  publicavel, nao falha. Predicao cruzada no regime limpo, N=50:
  overlay/m2 ~ 16.0/3.197 ~ 5.0x.

P7 -- aperto (c=1.6089), REVISADO PELA FUMACA antes da grade: a identidade
  bit a bit m2==baseline vale no regime UNIFORME (verificada: telemetria
  byte-identica, guarda 100%, zero toggles, sem churn). Sob CHURN a identidade
  e' impossivel por mecanismo: quando uma morte contrai o anel, o 2o vizinho
  entra no alcance e o termo k=2 liga com dado REAL (fumaca: guarda 99.0%
  derrubada, 20 toggles em 35 s). P7 sob churn e' portanto ESTATISTICA:
  egap_mean_steady20 do m2 dentro do IQR do baseline, pareado por semente.

P8 -- chattering (pino b-ii): em c=1.6089 sob churn o chaveamento nao
  desestabiliza: m2_k2_toggles_per_s_steady20 cresce com a taxa de churn, e
  ainda assim P7-churn se sustenta. As duas colunas de guarda saem por celula
  com escopo no nome e run_duration_s na linha.

LINHA DE AMEACA (A.4 item 4): a margem nominal igual assume o fator de ganho
  efetivo da normalizacao por arcos (1/(2*gbar), lambdas uniformes) COMUM as
  duas leis. Lambdas nao uniformes ou desvios grandes do uniforme quebram a
  igualdade nominal -- por isso PROTECTION_ANGLE_DEG=360 pinado e a leitura de
  P5 restrita ao regime limpo quase-uniforme.

MENSAGENS (SCOPING_M2 secao 4): todos os metodos transmitem 1 broadcast de
  AgentState por agente vivo por tick -- a tabela IMPRIME tx por celula para
  provar a igualdade, nao para esconde-la. Colunas: tx_rows_steady20 (proxy de
  broadcasts: linhas de agent_telemetry com t>=20), pulse_payloads_fullrun
  (dual_pulse; 0 nos demais), range_required_c (o custo REAL do m2: alcance
  exigido), run_duration_s.

SENTINELAS (abortam o sweep, nao a celula):
  A1/A2/A3 herdadas da 8a-(ii) (matriz efetiva, vivos por superposicao, censo
  de papeis) -- importadas de run_comm_churn_sweep, nao copiadas.
  A4-INERCIA: as celulas baseline e dual_pulse dos blocos C/D (N=24, churn)
  devem sair com target_telemetry.csv BYTE-IDENTICO ao de comm_churn_runs/
  (mesma semente, mesmo commit-lineage de caminho de codigo). Divergiu => a
  implementacao m2 nao e' inerte => aborta. Verificacao em celulas REAIS de
  campanha, nao so na fumaca.
  A5-SIDECAR: celula m2 tem de produzir m2_guard.csv; baseline/overlay NAO.

REGRAS DA CAMPANHA ativas: contar observacoes independentes (n_run por celula
  = 8 sementes; eventos nao sao independentes); media so com janela nomeada;
  sentinela aborta em vez de devolver vazio; mediana+IQR+n em toda tabela.

Uso:
    python experiments/scaling_law/run_m2_campaign.py            # requer M2C_GO=1
    # env: M2C_DRY_RUN=1 (imprime grade e custo, nada roda)
    #      M2C_GO=1      (trava de seguranca: sem ela o runner NAO dispara)
    #      M2C_BLOCKS="C,D" M2C_SEEDS="0,1,..." M2C_TAG=""
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_comm_churn_sweep as ccs  # noqa: E402  -- assercoes + metricas churn
import run_comm_range_sweep as rcrs  # noqa: E402  -- geometria + metricas de evento
from metrics_util import effort_metrics, event_metrics, run_provenance  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

_TAG = os.environ.get("M2C_TAG", "")
_SUF = ("_" + _TAG) if _TAG else ""
RUNS_DIR = os.path.join(EXP_DIR, "m2_campaign_runs" + _SUF)
RESULTS_CSV = os.path.join(EXP_DIR, "m2_campaign_results" + _SUF + ".csv")
WORK_CSV = os.path.join(RUNS_DIR, "_partial_results.csv")
CCS_RUNS = os.path.join(EXP_DIR, "comm_churn_runs")   # sentinela A4

METHODS = ("baseline", "m2", "dual_pulse")
SEEDS = [int(x) for x in os.environ.get("M2C_SEEDS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
C_VALUES = (1.6089, 3.0071)
DT = 0.05
FD = 0.25
RATE_TOTAL = 12.0
OFF = 8.0
T0 = 5.0
CHURN_BUDGET = 150.0
CLEAN_BUDGET = 90.0
UPLINK = 200.0
R_ENC = 20.0
DRY_RUN = os.environ.get("M2C_DRY_RUN", "").strip().lower() in ("1", "true", "yes", "y")
GO = os.environ.get("M2C_GO", "").strip().lower() in ("1", "true", "yes", "y")
PROGRESS_EVERY = 8
BLOCK_FILTER = {b.strip().upper() for b in os.environ.get("M2C_BLOCKS", "").split(",") if b.strip()}

# blocos: (letra, N, regime, c)
BLOCKS = [
    ("C", 24, "churn", 1.6089), ("D", 24, "churn", 3.0071),
    ("A", 24, "clean", 1.6089), ("B", 24, "clean", 3.0071),
    ("G", 50, "churn", 1.6089), ("H", 50, "churn", 3.0071),
    ("E", 50, "clean", 1.6089), ("F", 50, "clean", 3.0071),
]


def range_m(n, c):
    import math
    return c * 2.0 * R_ENC * math.sin(math.pi / n)


def victim(n, seed):
    return 2 + ((n // 2 + seed) % n)


def cell_env(block, n, regime, c, method, seed, run_dir):
    rng_m = range_m(n, c)
    env = dict(os.environ)
    dur = T0 + (CHURN_BUDGET if regime == "churn" else CLEAN_BUDGET)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PROPAGATION_METHOD": method, "PROPAGATION_K_PROP": "0.0",
        "NUM_AGENTS": str(n), "ENCIRCLEMENT_RADIUS": f"{R_ENC:g}",
        "SIM_DURATION": f"{dur:g}", "CONTROL_PERIOD": f"{DT:g}",
        "AGENT_STATE_TIMEOUT": f"{FD:g}",
        "K_E_TAU": f"{250.0 / n:.6f}", "VM_MAX_SPEED_XY": "10", "VM_TAU_XY": "1",
        "EXPERIMENT_SEED": str(seed), "EXPERIMENT_REPRODUCIBLE": "True",
        "METRICS_T0": "0.0", "PROTECTION_ANGLE_DEG": "360",
        "COMM_ROLE_AWARE_RANGES": "True",
        "COMM_RANGE_AGENT_AGENT": f"{rng_m:.6g}",
        "COMM_RANGE_AGENT_TARGET": f"{UPLINK:g}",
        "COMMUNICATION_TRANSMISSION_RANGE": f"{UPLINK:g}",
        "COMMUNICATION_DELAY": "0.0", "COMMUNICATION_FAILURE_RATE": "0.0",
        "INIT_ANGLES_EQUIDISTANT": "True", "INIT_RADIUS_RANGE": "0.0",
        "TARGET_MOTION_SPEED_XY": "0.0",
        "VIS_OPEN_BROWSER": "False", "SKIP_TELEMETRY_PLOTS": "True",
        "RUNS_SUMMARY_CSV_PATH": os.path.join(run_dir, "runs_summary.csv"),
    })
    if regime == "churn":
        env.update({
            "DETERMINISTIC_FAILURE_ENABLE": "False", "FAILURE_ENABLE": "True",
            "FAILURE_MEAN_FAILURES_PER_MIN": f"{RATE_TOTAL / n:.6f}",
            "FAILURE_OFF_TIME": f"{OFF:g}",
        })
    else:
        env.update({
            "DETERMINISTIC_FAILURE_ENABLE": "True",
            "DETERMINISTIC_FAILURE_AGENT_ID": str(victim(n, seed)),
            "DETERMINISTIC_FAILURE_TIME_T0": f"{T0:g}",
            "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
            "FAILURE_ENABLE": "True", "FAILURE_MEAN_FAILURES_PER_MIN": "0.0",
        })
    if method == "dual_pulse":
        env.update({"DUAL_PULSE_INTEGRATION": "B2", "DUAL_PULSE_DELTA_SCALE": "1.0",
                    "DUAL_PULSE_T_FF": "1", "DUAL_PULSE_TTL_HOPS": str(3 * n)})
    if method == "m2":
        env.update({"M2_W2": "2.0", "M2_GAIN_SCALE": "auto"})
    return env


def sentinel_inertness(block, n, c, method, seed, run_dir):
    """A4: blocos C/D baseline+overlay byte-identicos as celulas da 8a-(ii)."""
    if block not in ("C", "D") or method == "m2":
        return
    m_dir = "baseline" if method == "baseline" else "dual_pulse"
    aa = {1.6089: "8.4", 3.0071: "15.7"}[c]
    ref = os.path.join(CCS_RUNS, f"{m_dir}_aa{aa}_fd0.25_s{seed}", "target_telemetry.csv")
    got = os.path.join(run_dir, "target_telemetry.csv")
    if not os.path.exists(ref):
        print(f"     [A4] referencia ausente ({ref}); sentinela pulada NESTA celula")
        return
    with open(ref, "rb") as fa, open(got, "rb") as fb:
        if fa.read() != fb.read():
            raise rcrs.AssertionFailed(
                f"A4-INERCIA violada: {method} {block} s={seed} difere de {ref}. "
                "A implementacao m2 nao e' inerte para os caminhos existentes.")


def sentinel_sidecar(method, run_dir):
    has = os.path.exists(os.path.join(run_dir, "m2_guard.csv"))
    if method == "m2" and not has:
        raise rcrs.AssertionFailed(f"A5: celula m2 sem m2_guard.csv em {run_dir}")
    if method != "m2" and has:
        raise rcrs.AssertionFailed(f"A5: m2_guard.csv em celula {method} ({run_dir})")


def guard_metrics(run_dir, dur):
    p = os.path.join(run_dir, "m2_guard.csv")
    out = {"m2_k2_dropped_frac_steady20": np.nan, "m2_k2_toggles_per_s_steady20": np.nan}
    if not os.path.exists(p):
        return out
    d = pd.read_csv(p)
    ts = float(d["ticks_steady20"].sum())
    if ts > 0:
        out["m2_k2_dropped_frac_steady20"] = float(d["k2_dropped_steady20"].sum() / ts)
    steady_span = max(dur - 20.0, 1e-9)
    out["m2_k2_toggles_per_s_steady20"] = float(d["k2_toggles_steady20"].sum() / steady_span)
    return out


def run_cell(block, n, regime, c, method, seed):
    tag = {"baseline": "baseline", "m2": "m2", "dual_pulse": "B2"}[method]
    run_dir = os.path.join(RUNS_DIR, f"{block}_{method}_N{n}_c{c:g}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for fn in ("target_telemetry.csv", "events.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    dur = T0 + (CHURN_BUDGET if regime == "churn" else CLEAN_BUDGET)

    print(f"  -> {block} {tag:8s} N={n:<3d} c={c:<7g} s={seed} ...", end="", flush=True)
    proc = subprocess.run([sys.executable, MAIN_PY], cwd=run_dir,
                          env=cell_env(block, n, regime, c, method, seed, run_dir),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")

    # sentinelas de modulo compartilhado operam com o N desta celula
    ccs.N = n
    rcrs.N = n
    checks = ccs.assert_cell(f"{tag} {block} s={seed}", run_dir, proc.stdout, FD)
    sentinel_sidecar(method, run_dir)
    sentinel_inertness(block, n, c, method, seed, run_dir)

    agent_csv = os.path.join(run_dir, "agent_telemetry.csv")
    at_df = None
    tx_rows_steady20 = np.nan
    if os.path.exists(agent_csv):
        try:
            at_df = pd.read_csv(agent_csv, usecols=["node_id", "timestamp", "theta_rel"])
            tx_rows_steady20 = int((at_df["timestamp"] >= 20.0).sum())
        except Exception:
            pass

    m = ccs.metrics_from_run(run_dir, FD, at_df=at_df)
    if not m:
        print(f" FALHOU (rc={proc.returncode})")
        return None
    if regime == "clean":
        try:
            tel = pd.read_csv(os.path.join(run_dir, "target_telemetry.csv"))
            m.update(event_metrics(tel, T0))
        except Exception as exc:
            print(f"\n     [aviso] event_metrics falhou: {exc}")

    m.update(effort_metrics(agent_csv, t0=20.0, vmax=10.0))
    try:
        os.remove(agent_csv)
    except OSError:
        pass

    m.update(guard_metrics(run_dir, dur))
    dpm = os.path.join(run_dir, "dual_pulse_messages.csv")
    m["pulse_payloads_fullrun"] = (
        float(pd.read_csv(dpm)["pulse_payloads_broadcast"].sum()) if os.path.exists(dpm) else 0.0
    )
    m.update(checks)
    m.update({
        "block": block, "method": tag, "N": n, "regime": regime, "c_hops": c,
        "range_aa_m": round(range_m(n, c), 4), "range_at": UPLINK,
        "fd_timeout": FD, "rate_total": RATE_TOTAL if regime == "churn" else 0.0,
        "seed": seed, "dt": DT, "run_duration_s": dur,
        "tx_rows_steady20": tx_rows_steady20,
        "range_required_c": {"baseline": 1.0, "m2": 1.9829 if n == 24 else 1.99605,
                             "B2": 1.9829 if n == 24 else 1.99605}[tag],
    })
    m.update(rcrs.provenance_with_retry(run_dir))
    print(f" ok egap20={m.get('egap_mean_steady20', float('nan')):.4f}"
          f" tset={m.get('t_settle', float('nan')):.6g}"
          f" drop={m.get('m2_k2_dropped_frac_steady20', float('nan')):.6g}")
    return m


def cells():
    out = []
    for (b, n, regime, c) in BLOCKS:
        if BLOCK_FILTER and b not in BLOCK_FILTER:
            continue
        for seed in SEEDS:
            for method in METHODS:
                out.append((b, n, regime, c, method, seed))
    return out


def _key(r):
    return (str(r["block"]), str(r["method"]), int(r["N"]), round(float(r["c_hops"]), 6), int(r["seed"]))


def print_grid():
    grid = cells()
    n24 = sum(1 for g in grid if g[1] == 24)
    n50 = len(grid) - n24
    est = n24 * 45 + n50 * 124
    print("Item 9 -- campanha m=2 (pre-registro no docstring)")
    print(f"  {'bloco':>6} {'N':>4} {'regime':>7} {'c':>8} {'alcance':>9} {'celulas':>8}")
    for (b, n, regime, c) in BLOCKS:
        if BLOCK_FILTER and b not in BLOCK_FILTER:
            continue
        print(f"  {b:>6} {n:>4} {regime:>7} {c:>8g} {range_m(n, c):>8.3f}m {len(SEEDS) * len(METHODS):>8}")
    print(f"\n  TOTAL = {len(grid)} celulas | custo MEDIDO ~ {est / 60:.0f} min "
          f"(45 s/celula N=24, 124 s N=50, calibrado na fumaca)")
    print(f"  predicoes: P5 3.1565 (N=24) / 3.1970 (N=50); P6 overlay>=m2>baseline, cruzada 5.0x;")
    print(f"  P7 uniforme=bit-exato (fumaca) / churn=estatistica; P8 chattering nao desestabiliza")


def main():
    print_grid()
    if DRY_RUN:
        print("\nM2C_DRY_RUN: nada rodou.")
        return
    if not GO:
        print("\nTrava: defina M2C_GO=1 para disparar (o pre-registro precisa de revisao antes).")
        return

    os.makedirs(RUNS_DIR, exist_ok=True)
    store = {}
    for src in (RESULTS_CSV, WORK_CSV):
        if os.path.exists(src):
            try:
                for r in pd.read_csv(src).to_dict("records"):
                    store[_key(r)] = r
            except Exception:
                pass
    print(f"{len(store)} celulas ja no CSV (merge incremental)\n")

    grid = cells()
    done = 0
    for (b, n, regime, c, method, seed) in grid:
        tag = {"baseline": "baseline", "m2": "m2", "dual_pulse": "B2"}[method]
        if (b, tag, n, round(c, 6), seed) in store:
            continue
        r = run_cell(b, n, regime, c, method, seed)
        done += 1
        if r:
            store[_key(r)] = r
            pd.DataFrame(list(store.values())).to_csv(WORK_CSV, index=False)
        if done % PROGRESS_EVERY == 0:
            print(f"[progress] {done}/{len(grid)} celulas", flush=True)

    df = pd.DataFrame(list(store.values()))
    if df.empty:
        print("\nSem resultados.")
        return
    df.to_csv(RESULTS_CSV, index=False)
    if "git_dirty" in df.columns:
        vals = sorted({"SEM MANIFESTO" if pd.isna(v) else str(v) for v in df.git_dirty})
        print(f"\ngit_dirty nas linhas: {vals}")
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except rcrs.AssertionFailed as exc:
        print(f"\n*** SWEEP ABORTADO -- {exc}", file=sys.stderr)
        sys.exit(2)
