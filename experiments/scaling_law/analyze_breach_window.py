#!/usr/bin/env python
"""Analise do experimento decisivo E4: a janela de brecha e' cinematica ou de coordenacao?

Le os breach_window_results*.csv produzidos por run_breach_window.py, pareia
baseline x B2 por (N, Vmax, tau_a, seed) e aplica a REGRA DE DECISAO de
docs/experiments/CHURN_PAIRED.md 5.3:

  pico G_max ~ 2(N-1)/N nos dois, plano em Vmax/tau_a  -> piso geometrico confirmado
  t_close ~ igual nos dois e escalando com tau_a       -> CONJECTURA CONFIRMADA
  t_close plano p/ B2 e crescendo p/ baseline com N    -> CONJECTURA REFUTADA
  t_close pior no B2                                   -> defeito real do feedforward

Nao roda simulacao.

Uso:
    python experiments/scaling_law/analyze_breach_window.py
    # env: BREACH_GLOB="breach_window_results_*.csv"  BREACH_NOFIG="1"
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    sys.exit("scipy nao instalado. Rode: pip install -r requirements.txt")

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.environ.get("BREACH_GLOB", "breach_window_results*.csv")
OUT_CSV = os.path.join(EXP_DIR, os.environ.get("BREACH_ANALYSIS_OUT",
                                               "breach_window_summary.csv"))
OUT_FIG = os.path.join(EXP_DIR, "figures", "fig_breach_window.png")

KEYS = ["N", "vmax", "tau_xy", "seed"]
C_BASE = "firebrick"
C_B2 = "royalblue"


def load_all():
    frames = []
    for p in sorted(glob.glob(os.path.join(EXP_DIR, PATTERN))):
        try:
            d = pd.read_csv(p)
        except Exception as exc:
            print(f"  ! {os.path.basename(p)}: {exc}")
            continue
        if d.empty:
            continue
        d["source_file"] = os.path.basename(p)
        frames.append(d)
    if not frames:
        sys.exit(f"Nenhum CSV casando com {PATTERN} em {EXP_DIR}")
    df = pd.concat(frames, ignore_index=True)
    # Uma celula = (metodo, N, Vmax, tau_a, seed). Duplicatas entre arquivos ficam
    # com a ultima ocorrencia (merge incremental dos runners).
    return df.drop_duplicates(subset=["method"] + KEYS, keep="last")


def paired(df, metric, **filt):
    g = df
    for k, v in filt.items():
        g = g[g[k] == v]
    b = g[g.method == "baseline"][KEYS + [metric]]
    o = g[g.method == "B2"][KEYS + [metric]]
    m = b.merge(o, on=KEYS, suffixes=("_b", "_o")).sort_values(KEYS)
    return m[f"{metric}_b"].to_numpy(float), m[f"{metric}_o"].to_numpy(float)


def stat_row(df, metric, label, **filt):
    base, b2 = paired(df, metric, **filt)
    if base.size == 0:
        return None
    fin = np.isfinite(base) & np.isfinite(b2)
    n_inf = int((~fin).sum())
    p = float("nan")
    if fin.sum() >= 1 and np.any(base[fin] != b2[fin]):
        try:
            p = float(wilcoxon(base[fin], b2[fin]).pvalue)
        except ValueError:
            p = float("nan")
    ratio = np.full(base.shape, np.nan)
    ok = fin & (b2 != 0)
    ratio[ok] = base[ok] / b2[ok]
    r = ratio[np.isfinite(ratio)]
    return {
        "metric": metric, "label": label, **{k: v for k, v in filt.items()},
        "n": int(base.size), "n_nonfinite": n_inf,
        "base_median": float(np.nanmedian(base)), "b2_median": float(np.nanmedian(b2)),
        "base_min": float(np.nanmin(base)) if fin.any() else np.nan,
        "base_max": float(np.nanmax(base)) if fin.any() else np.nan,
        "b2_min": float(np.nanmin(b2)) if fin.any() else np.nan,
        "b2_max": float(np.nanmax(b2)) if fin.any() else np.nan,
        "ratio_median": float(np.median(r)) if r.size else np.nan,
        "ratio_min": float(np.min(r)) if r.size else np.nan,
        "ratio_max": float(np.max(r)) if r.size else np.nan,
        "n_lose": int(np.sum(r < 1.0)),
        "p_value": p,
    }


def _table(df, metric, label, by, values, fixed=None):
    """Imprime e devolve as linhas de `metric` agrupadas por `by`."""
    fixed = fixed or {}
    rows = []
    fx = "".join(f" {k}={v:g}" for k, v in fixed.items())
    print(f"\n--- {label}  (por {by}{fx}) ---")
    print(f"{by:>8}{'n':>4}{'base med':>10}{'base[min,max]':>20}"
          f"{'B2 med':>9}{'B2[min,max]':>20}{'razao':>8}{'n_lose':>7}{'p':>9}")
    for v in values:
        r = stat_row(df, metric, label, **{by: v}, **fixed)
        if r is None:
            continue
        rows.append(r)
        p = "      n/a" if not np.isfinite(r["p_value"]) else f"{r['p_value']:9.4f}"
        b_rng = "[{:.2f}, {:.2f}]".format(r["base_min"], r["base_max"])
        o_rng = "[{:.2f}, {:.2f}]".format(r["b2_min"], r["b2_max"])
        inf_note = f"   ({r['n_nonfinite']} nao fecharam)" if r["n_nonfinite"] else ""
        print(f"{v:>8g}{r['n']:>4}{r['base_median']:>10.3f}{b_rng:>20}"
              f"{r['b2_median']:>9.3f}{o_rng:>20}"
              f"{r['ratio_median']:>8.2f}{r['n_lose']:>7}{p}{inf_note}")
    return rows


def a1_reconfiguration_vs_breach():
    """Is the reconfiguration time REALLY 'how long the gap stays open'?

    draft v1 (1-introduction.tex:26) asserts an identity between two quantities the
    simulator computes differently:
      t_settle / tau : fitted on E_gap  = RMS ACROSS THE RING of relative gap error
      t_close        : when G_max (the MAX gap) drops back under the threshold
    Both come from the same target_telemetry.csv, so this re-reads the breach runs
    without simulating anything. Returns a DataFrame (empty if the run dirs are gone
    -- they are gitignored, so this only works on the machine that ran the sweep).
    """
    sys.path.insert(0, EXP_DIR)
    from metrics_util import event_metrics

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(EXP_DIR, "breach_runs_*", "*"))):
        tgt = os.path.join(run_dir, "target_telemetry.csv")
        if not os.path.exists(tgt):
            continue
        name = os.path.basename(run_dir)
        try:
            d = pd.read_csv(tgt, usecols=["timestamp", "E_gap", "G_max"])
        except Exception:
            continue
        t_fail = 5.0
        ev = os.path.join(run_dir, "events.csv")
        if os.path.exists(ev):
            try:
                e = pd.read_csv(ev)
                f = e[e["event_type"] == "failure_start"]
                if len(f):
                    t_fail = float(f["timestamp"].min())
            except Exception:
                pass
        post = d[d["timestamp"] >= t_fail].reset_index(drop=True)
        if len(post) < 10:
            continue
        t = post["timestamp"].to_numpy(float)
        g = post["G_max"].to_numpy(float)
        m = event_metrics(d, t_fail)
        above = g > 1.25
        if not above.any():
            tc = 0.0
        else:
            last = int(np.max(np.where(above)[0]))
            tc = float("inf") if last >= g.size - 1 else float(t[last + 1] - t[0])
        rows.append({
            "run": name,
            "method": "B2" if name.startswith("dual_pulse") else "baseline",
            "t_close_125": tc,
            "t_settle_egap": m.get("t_settle", np.nan),
            "tau_fit_egap": m.get("tau_fit", np.nan),
        })
    return pd.DataFrame(rows)


def loglog_slope(x, y):
    """Exponent p of y ~ x^p. The claim under test is p ~ 2 (Theta(N^2))."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def make_figure(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "figure.dpi": 150})
    main = df[(df.N == 24)]
    taus = sorted(main.tau_xy.unique())
    vmaxes = sorted(main.vmax.unique())
    Ns = sorted(df.N.unique())

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    # (a) pico G_max vs Vmax -- deve ser PLANO e identico nos dois se o piso for geometrico
    ax = axes[0, 0]
    # The two curves coincide to 4 decimals, so the second would hide the first:
    # draw the baseline as a wide hollow ring and B2 as a small filled dot inside.
    for meth, c, ms, mfc, lw in (("baseline", C_BASE, 13, "none", 2.6),
                                 ("B2", C_B2, 6, C_B2, 1.4)):
        med = [main[(main.method == meth) & (main.vmax == v)]["gmax_peak"].median()
               for v in vmaxes]
        ax.plot(vmaxes, med, "o-", color=c, label=meth, lw=lw,
                markersize=ms, markerfacecolor=mfc, markeredgewidth=2.2)
    n0 = 24
    ax.axhline(2.0 * (n0 - 1) / n0, color="black", ls="--", lw=1.4,
               label=f"geometria 2(N-1)/N = {2.0*(n0-1)/n0:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("Vmax [m/s]")
    ax.set_ylabel("pico de G_max")
    ax.set_title("(a) O PICO tem piso geometrico?", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()

    # (b) t_close vs tau_a -- se for cinematico, escala com tau_a nos DOIS
    ax = axes[0, 1]
    for meth, c in (("baseline", C_BASE), ("B2", C_B2)):
        med, lo, hi = [], [], []
        for t in taus:
            s = main[(main.method == meth) & (main.tau_xy == t)]["t_close_125"]
            s = s[np.isfinite(s)]
            med.append(s.median()); lo.append(s.min()); hi.append(s.max())
        ax.plot(taus, med, "o-", color=c, label=meth, lw=2)
        ax.fill_between(taus, lo, hi, color=c, alpha=0.15)
    ax.set_xlabel("tau_a [s]"); ax.set_ylabel("t_close (G_max < 1.25) [s]")
    ax.set_title("(b) O FECHAMENTO e' limitado pela atuacao?", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()

    # (c) t_close vs N -- o teste da alegacao "Theta(N^2) deixa a brecha aberta por minutos"
    ax = axes[1, 0]
    sub = df[(df.vmax == 10.0) & (df.tau_xy == 1.0)]
    for meth, c in (("baseline", C_BASE), ("B2", C_B2)):
        med, lo, hi = [], [], []
        ns = [n for n in Ns if len(sub[(sub.method == meth) & (sub.N == n)])]
        for n in ns:
            s = sub[(sub.method == meth) & (sub.N == n)]["t_close_125"]
            s = s[np.isfinite(s)]
            med.append(s.median()); lo.append(s.min()); hi.append(s.max())
        if ns:
            ax.plot(ns, med, "o-", color=c, label=meth, lw=2)
            ax.fill_between(ns, lo, hi, color=c, alpha=0.15)
    ax.set_xlabel("N"); ax.set_ylabel("t_close (G_max < 1.25) [s]")
    ax.set_title("(c) A brecha do baseline cresce com N?", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()

    # (d) area da brecha pareada
    ax = axes[1, 1]
    base, b2 = paired(main, "breach_area_125")
    ax.scatter(base, b2, s=28, color="0.35", zorder=3)
    lim = [0, max(float(np.nanmax(base)), float(np.nanmax(b2))) * 1.08]
    ax.plot(lim, lim, "--", color="black", lw=1.4, label="empate")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("area da brecha - baseline"); ax.set_ylabel("area da brecha - B2")
    ax.set_title("(d) Area da brecha, par a par (N=24)", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle("E4 - A janela de brecha e' cinematica ou de coordenacao?\n"
                 "falha unica PERMANENTE, anel uniforme, comunicacao ideal",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSalvo: {out_path}")


def main():
    df = load_all()
    print(f"{len(df)} celulas de "
          f"{df.source_file.nunique()} arquivo(s); N={sorted(df.N.unique())}, "
          f"Vmax={sorted(df.vmax.unique())}, tau_a={sorted(df.tau_xy.unique())}, "
          f"seeds={sorted(df.seed.unique())}")
    if "git_commit" in df.columns:
        print(f"proveniencia: commit(s)={sorted(df.git_commit.dropna().unique())} "
              f"dirty={sorted(df.git_dirty.dropna().unique())}")

    main24 = df[df.N == 24]
    rows = []
    n0 = 24
    pred = 2.0 * (n0 - 1) / n0
    print(f"\n=== 1. O PICO: previsao geometrica 2(N-1)/N = {pred:.4f} (N={n0}) ===")
    for meth in ("baseline", "B2"):
        s = main24[main24.method == meth]["gmax_peak"]
        print(f"  {meth:<9} mediana={s.median():.4f}  [{s.min():.4f}, {s.max():.4f}]  "
              f"n={len(s)}  razao/previsto={s.median()/pred:.4f}")
    rows += _table(main24, "gmax_peak", "pico de G_max", "vmax",
                   sorted(main24.vmax.unique()))

    print(f"\n=== 2. O FECHAMENTO (t_close, G_max < 1.25) ===")
    rows += _table(main24, "t_close_125", "t_close 1.25", "tau_xy",
                   sorted(main24.tau_xy.unique()))
    rows += _table(main24, "t_close_125", "t_close 1.25", "vmax",
                   sorted(main24.vmax.unique()))
    rows += _table(main24, "t_close_110", "t_close 1.10", "tau_xy",
                   sorted(main24.tau_xy.unique()))

    sub = df[(df.vmax == 10.0) & (df.tau_xy == 1.0)]
    if sub.N.nunique() > 1:
        print(f"\n=== 3. ESCALA EM N (Vmax=10, tau_a=1.0) - o teste do 'aberta por minutos' ===")
        rows += _table(sub, "t_close_125", "t_close 1.25", "N", sorted(sub.N.unique()))
        rows += _table(sub, "t_close_110", "t_close 1.10", "N", sorted(sub.N.unique()))
        rows += _table(sub, "breach_area_125", "area da brecha", "N", sorted(sub.N.unique()))

        print("\n  Expoente p em t_close ~ N^p (a alegacao sob teste e' p ~ 2):")
        Ns = sorted(sub.N.unique())
        for metric in ("t_close_125", "t_close_110"):
            for meth in ("baseline", "B2"):
                med = [sub[(sub.method == meth) & (sub.N == n)][metric].median() for n in Ns]
                p = loglog_slope(Ns, med)
                at100 = med[-1] * (100.0 / Ns[-1]) ** p if np.isfinite(p) else float("nan")
                print(f"    {metric:<13} {meth:<9} p={p:5.2f}   "
                      f"medianas={[f'{v:.2f}' for v in med]}   "
                      f"extrapolado p/ N=100: {at100:.1f}s")

    print(f"\n=== 4. AREA DA BRECHA (N=24) ===")
    rows += _table(main24, "breach_area_125", "area da brecha", "tau_xy",
                   sorted(main24.tau_xy.unique()))

    print("\n=== 5. A1: 'o tempo de reconfiguracao E' quanto tempo o vao fica aberto' ===")
    a1 = a1_reconfiguration_vs_breach()
    if a1.empty:
        print("  (diretorios breach_runs_* ausentes -- so roda na maquina do sweep)")
    else:
        a1["ratio"] = a1["t_settle_egap"] / a1["t_close_125"]
        for meth in ("baseline", "B2"):
            s = a1[a1.method == meth]
            r = s["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            print(f"  {meth:<9} n={len(s):3d}  t_close={s.t_close_125.median():6.2f}s   "
                  f"t_settle(E_gap)={s.t_settle_egap.median():7.2f}s   "
                  f"tau_fit={s.tau_fit_egap.median():6.2f}s   "
                  f"razao={r.median():6.2f}x  [{r.min():.1f}, {r.max():.1f}]")
        a1_path = os.path.join(EXP_DIR, "breach_a1_reconfig_vs_breach.csv")
        a1.to_csv(a1_path, index=False)
        print(f"  Escrito: {os.path.basename(a1_path)}")
        print("  razao >> 1 => as duas grandezas NAO sao a mesma; a brecha fecha muito")
        print("  antes de o anel terminar de redistribuir.")

    out = pd.DataFrame([r for r in rows if r])
    out.to_csv(OUT_CSV, index=False)
    print(f"\nEscrito: {OUT_CSV}  ({len(out)} linhas)")

    if os.environ.get("BREACH_NOFIG", "").strip() not in ("1", "true", "True"):
        make_figure(df, OUT_FIG)


if __name__ == "__main__":
    main()
