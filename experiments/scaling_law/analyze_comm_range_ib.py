#!/usr/bin/env python
"""Fase 8a (i-b) -- detector de falhas OU alcance? Comparacao lado a lado.

A fase (i) fixou AGENT_STATE_TIMEOUT em 5*dt, copiado do run_breach_window --
que usa esse valor PORQUE o canal dele e' ideal. Aqui o canal e' degradado por
construcao, e para o detector um vizinho fora de alcance e' indistinguivel de um
vizinho morto. A (i-b) repete os dois pontos curtos com 20*dt (o FD-fix da
campanha de comunicacao) e este script poe os dois lado a lado.

PRE-REGISTRO (escrito antes do dado):
  P1  A inversao do B2 em 8.4 m PERSISTE com o timeout longo, porque a abstencao
      do sucessor e' geometria e nao deteccao.
        persiste -> mecanismo confirmado
        some     -> o achado e' acoplamento detector x alcance
  P2  O penhasco de fechamento PODE andar para a esquerda (6.3 m passa a fechar).
        anda -> era o detector
        fica -> e' alcance puro

As linhas da fase (i) sao anteriores as colunas de injecao, entao elas sao
RECALCULADAS aqui a partir do events.csv preservado de cada celula, usando a
MESMA funcao do runner (dual_pulse_coverage) -- nao uma segunda definicao que
poderia divergir.

Uso:
    python experiments/scaling_law/analyze_comm_range_ib.py
    # env: CRSA_RANGES="6.3,8.4"   CRSA_IB_TAG="ib"
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_comm_range_sweep as sweep  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
IB_TAG = os.environ.get("CRSA_IB_TAG", "ib")
RANGES = [float(x) for x in os.environ.get("CRSA_RANGES", "6.3,8.4").split(",") if x.strip()]

PHASE_I_CSV = os.path.join(EXP_DIR, "comm_range_results.csv")
PHASE_IB_CSV = os.path.join(EXP_DIR, f"comm_range_results_{IB_TAG}.csv")
PHASE_I_RUNS = os.path.join(EXP_DIR, "comm_range_runs")
PHASE_IB_RUNS = os.path.join(EXP_DIR, f"comm_range_runs_{IB_TAG}")


def med_iqr(series):
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    finite = s[np.isfinite(s)]
    n_inf = int((~np.isfinite(s)).sum())
    if finite.empty:
        return None, None, None, 0, n_inf
    return (float(finite.median()), float(finite.quantile(.25)),
            float(finite.quantile(.75)), len(finite), n_inf)


def fmt(series, prec=2):
    med, q1, q3, n_ok, n_inf = med_iqr(series)
    if n_ok == 0:
        return "inf (todas)"
    txt = f"{med:.{prec}f} [{q1:.{prec}f},{q3:.{prec}f}]"
    return txt + (f" +{n_inf}inf" if n_inf else "")


def retrofit_injections(df, runs_dir):
    """Recalcula as colunas de injecao a partir do events.csv de cada celula."""
    cols = ["topo_injections", "dp_landed_events", "dp_landed_saida",
            "dp_landed_entrada", "dp_seq_max", "dp_originators"]
    if all(c in df.columns and df[c].notna().any() for c in cols):
        return df, False          # ja tem as colunas (fase i-b)
    rows = []
    for r in df.to_dict("records"):
        method = "dual_pulse" if r["method"] == "B2" else "baseline"
        run_dir = os.path.join(runs_dir, f"{method}_aa{r['range_aa']:g}_s{int(r['seed'])}")
        survivors = int(round(float(r.get("alive_final", sweep.N - 1) or sweep.N - 1)))
        r.update(sweep.dual_pulse_coverage(run_dir, survivors))
        rows.append(r)
    return pd.DataFrame(rows), True


def successor_check(runs_dir, rng, label):
    """Reverifica no a no: quem NAO completou, e o hop_sum dos completantes."""
    print(f"\n--- {label}: verificacao por no em {rng:g} m (B2) ---")
    print(f"{'semente':>8} {'vitima':>7} {'orig':>5} {'sucessor completou?':>20} "
          f"{'cobertos':>9} {'hop_sum':>9} {'tipo':>9} {'seq':>4}")
    any_row = False
    for seed in range(8):
        run_dir = os.path.join(runs_dir, f"dual_pulse_aa{rng:g}_s{seed}")
        p = os.path.join(run_dir, "events.csv")
        if not os.path.exists(p):
            continue
        try:
            ev = pd.read_csv(p)
        except Exception:
            continue
        if "event_type" not in ev.columns or not len(ev):
            continue
        any_row = True
        types = ev["event_type"].astype(str)
        fail = ev[types == "failure_start"]
        victim = int(fail["node_id"].iloc[0]) if len(fail) else -1
        done = ev[types.str.startswith("dual_pulse_event_completed")]
        ss = ev[types.str.startswith("dual_pulse_self_shift")]
        covered = set(done["node_id"]) | set(ss["node_id"])
        orig = sorted(set(ss["node_id"]))
        succ = victim + 1 if victim >= 0 else -1
        hop = "-"
        if len(done) and {"h_CCW", "h_CW"} <= set(done.columns):
            h = (pd.to_numeric(done["h_CCW"], errors="coerce")
                 + pd.to_numeric(done["h_CW"], errors="coerce")).dropna()
            if len(h):
                hop = f"{h.median():.0f}" + ("" if (h == sweep.N - 1).all() else "*")
        kinds = {t.rsplit("_", 1)[-1] for t in set(done["event_type"]) | set(ss["event_type"])}
        seqs = [int(e.split("_")[1]) for e in ev["event_id"].dropna().astype(str).unique()
                if len(e.split("_")) == 2 and e.split("_")[1].isdigit()]
        print(f"{seed:>8} {victim:>7} {str(orig if orig else '-'):>5} "
              f"{('SIM' if succ in covered else 'NAO'):>20} {len(covered):>9} {hop:>9} "
              f"{','.join(sorted(kinds)) or '-':>9} {max(seqs) if seqs else 0:>4}")
    if not any_row:
        print("  (sem celulas)")
    print(f"  hop_sum esperado numa travessia completa = {sweep.N - 1}; '*' = nem todos bateram")


def main():
    if not os.path.exists(PHASE_I_CSV):
        print(f"falta {PHASE_I_CSV}"); return 1
    if not os.path.exists(PHASE_IB_CSV):
        print(f"falta {PHASE_IB_CSV} -- rode a fase (i-b) primeiro:\n"
              f"  CRS_TAG={IB_TAG} CRS_RANGES=6.3,8.4 CRS_FD_TIMEOUT=1.0 "
              f"python experiments/scaling_law/run_comm_range_sweep.py")
        return 1

    a = pd.read_csv(PHASE_I_CSV)
    b = pd.read_csv(PHASE_IB_CSV)
    a = a[a.range_aa.isin(RANGES)].copy()
    b = b[b.range_aa.isin(RANGES)].copy()
    a, retro_a = retrofit_injections(a, PHASE_I_RUNS)
    b, _ = retrofit_injections(b, PHASE_IB_RUNS)

    fd_a = sorted(set(pd.to_numeric(a.agent_state_timeout, errors="coerce").dropna()))
    fd_b = sorted(set(pd.to_numeric(b.agent_state_timeout, errors="coerce").dropna()))
    print("=" * 96)
    print("FASE 8a (i-b) -- detector de falhas ou alcance?")
    print("=" * 96)
    print(f"  (i)   AGENT_STATE_TIMEOUT efetivo nas linhas: {fd_a}  n={len(a)}")
    print(f"  (i-b) AGENT_STATE_TIMEOUT efetivo nas linhas: {fd_b}  n={len(b)}")
    print(f"  git_dirty  (i): {sorted(set(a.git_dirty.astype(str)))}   "
          f"(i-b): {sorted(set(b.git_dirty.astype(str)))}")
    if retro_a:
        print("  colunas de injecao da fase (i) RECALCULADAS do events.csv preservado")

    for metric, prec, title in (("t_close_125", 2, "t_close_125 (limiar primario)"),
                                ("t_close_110", 2, "t_close_110 (limiar estrito)"),
                                ("gmax_peak", 3, "gmax_peak")):
        print(f"\n=== {title}: mediana [IQR], (i) 5*dt  vs  (i-b) 20*dt ===")
        print(f"{'range':>6} {'metodo':>9} | {'(i) 0.25s':>26} | {'(i-b) 1.0s':>26}")
        for rng in RANGES:
            for meth in ("baseline", "B2"):
                sa = a[(a.range_aa == rng) & (a.method == meth)][metric]
                sb = b[(b.range_aa == rng) & (b.method == meth)][metric]
                print(f"{rng:>6g} {meth:>9} | {fmt(sa, prec):>26} | {fmt(sb, prec):>26}")

    print("\n=== injecoes e eventos que aterrissaram (mediana / maximo) ===")
    print("  topo_inj = linhas 'pulse_injected' (fast_layer; PROXY de agitacao topologica,")
    print("  nao a injecao do dual_pulse, que o protocolo nao registra).")
    print("  seq_max > 1 prova injecoes anteriores do mesmo originador que morreram sem completar ninguem.")
    print(f"{'range':>6} {'fase':>6} | {'topo_inj med/max':>18} | {'landed SAIDA':>13} | "
          f"{'landed ENTRADA':>15} | {'seq_max med/max':>16} | {'cobertura':>10}")
    for rng in RANGES:
        for name, df in (("(i)", a), ("(i-b)", b)):
            g = df[(df.range_aa == rng) & (df.method == "B2")]
            if not len(g):
                continue
            ti = pd.to_numeric(g.topo_injections, errors="coerce")
            sq = pd.to_numeric(g.dp_seq_max, errors="coerce")
            sd = pd.to_numeric(g.dp_landed_saida, errors="coerce")
            en = pd.to_numeric(g.dp_landed_entrada, errors="coerce")
            cv = pd.to_numeric(g.dp_coverage, errors="coerce")
            print(f"{rng:>6g} {name:>6} | {ti.median():>8.0f} /{ti.max():>8.0f} | "
                  f"{sd.median():>6.1f} /{sd.max():>4.0f} | {en.median():>8.1f} /{en.max():>4.0f} | "
                  f"{sq.median():>7.1f} /{sq.max():>7.0f} | {cv.median():>10.2f}")

    for rng in RANGES:
        successor_check(PHASE_IB_RUNS, rng, f"(i-b) 20*dt")

    print("\n" + "=" * 96)
    print("VEREDITO CONTRA O PRE-REGISTRO")
    print("=" * 96)
    for rng in RANGES:
        for meth in ("baseline", "B2"):
            ca = med_iqr(a[(a.range_aa == rng) & (a.method == meth)]["t_close_125"])
            cb = med_iqr(b[(b.range_aa == rng) & (b.method == meth)]["t_close_125"])
            print(f"  {rng:>5g} m {meth:>9}: (i) {'nunca fecha' if ca[3] == 0 else f'{ca[0]:.2f}s'}"
                  f"  ->  (i-b) {'nunca fecha' if cb[3] == 0 else f'{cb[0]:.2f}s'}")

    def adv(df, rng):
        base = med_iqr(df[(df.range_aa == rng) & (df.method == "baseline")]["t_close_125"])
        b2 = med_iqr(df[(df.range_aa == rng) & (df.method == "B2")]["t_close_125"])
        if base[3] == 0 or b2[3] == 0:
            return None
        return base[0] / b2[0]

    if 8.4 in RANGES:
        r_a, r_b = adv(a, 8.4), adv(b, 8.4)
        print(f"\n  P1 -- vantagem do B2 em 8.4 m (>1 = overlay melhor):")
        print(f"       (i) {'n.a.' if r_a is None else f'{r_a:.2f}x'}  ->  "
              f"(i-b) {'n.a.' if r_b is None else f'{r_b:.2f}x'}")
        if r_a is not None and r_b is not None:
            if r_b < 1.0:
                print("       INVERSAO PERSISTE -> P1 confirmada: a abstencao e' geometria, nao deteccao.")
            else:
                print("       INVERSAO SUMIU -> P1 refutada: o achado e' acoplamento detector x alcance.")
    if 6.3 in RANGES:
        closed_a = med_iqr(a[(a.range_aa == 6.3)]["t_close_125"])[3]
        closed_b = med_iqr(b[(b.range_aa == 6.3)]["t_close_125"])[3]
        print(f"\n  P2 -- celulas que fecham em 6.3 m: (i) {closed_a}/{len(a[a.range_aa == 6.3])}"
              f"  ->  (i-b) {closed_b}/{len(b[b.range_aa == 6.3])}")
        print("       " + ("PENHASCO ANDOU -> era o detector." if closed_b > closed_a
                           else "PENHASCO FICOU -> alcance puro."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
