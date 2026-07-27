#!/usr/bin/env python
"""P3 -- analise de tau_B2 vs N e dt, arbitragem da celula N=50, e o crossover.

Responde, com numero, as sete perguntas de docs/experiments/DT_CROSSOVER.md:
  0 arbitragem da celula N=50/dt=0.01 (4.06 da escada vs 2.115 do largeN)
  1 expoente de tau_B2 vs N por braco de dt, com R2, e o que isso faz com a Lei 1
  2 invariancia em dt, e se a parte nao invariante e' latencia de deteccao (GRADE C)
  3 quanto vale c (segundos por salto / dt) e se t_dissem bate com c*(N/2)*dt
  4 onde tau_B2 deixa de ser plano, e se coincide com t_dissem ~ tau
  5 se "tau plano ate N=100" sobrevive a dt=0.05

Nao roda simulacao.

Uso:
    python experiments/scaling_law/analyze_dt_crossover.py
    # env: DTS_GLOB="dt_scaling_results*.csv"  DTS_NOFIG="1"
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.environ.get("DTS_GLOB", "dt_scaling_results*.csv")
OUT_CSV = os.path.join(EXP_DIR, "dt_crossover_summary.csv")
OUT_FIG = os.path.join(EXP_DIR, "figures", "fig_dt_crossover.png")

# As duas medidas incompativeis que a celula N=50/dt=0.01 arbitra.
LADDER_N50_DT01 = 4.06        # ladder_results_d001_*, 3 seeds, arvore commitada
LARGEN_N50_DT01 = 2.115       # largeN_results.csv, 2 seeds, arvore de maio nao commitada
BASELINE_EXP = 1.94           # expoente do baseline na Lei 1 (campanha)
BASELINE_COEF = 0.0417        # tau_base ~ 0.0417 * N^1.94


def load():
    frames = []
    for p in sorted(glob.glob(os.path.join(EXP_DIR, PATTERN))):
        if "smoke" in os.path.basename(p):
            continue
        try:
            d = pd.read_csv(p)
        except Exception as exc:
            print(f"  ! {os.path.basename(p)}: {exc}")
            continue
        if not d.empty:
            d["source_file"] = os.path.basename(p)
            frames.append(d)
    if not frames:
        sys.exit(f"Nenhum CSV casando com {PATTERN} em {EXP_DIR}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(
        subset=["grid", "method", "N", "dt", "seed", "agent_state_timeout"], keep="last")


def agg(df, **filt):
    g = df
    for k, v in filt.items():
        g = g[np.isclose(g[k], v)] if isinstance(v, float) else g[g[k] == v]
    return g


def cell(df, metric="tau_fit", **filt):
    g = agg(df, **filt)
    v = g[metric].to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    r2 = g["tau_fit_r2"].to_numpy(float); r2 = r2[np.isfinite(r2)]
    return {"n": int(v.size), "median": float(np.median(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "r2": float(np.median(r2)) if r2.size else np.nan}


def loglog(ns, ys):
    ns = np.asarray(ns, float); ys = np.asarray(ys, float)
    ok = np.isfinite(ns) & np.isfinite(ys) & (ns > 0) & (ys > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(ns[ok]), np.log(ys[ok])
    p, b = np.polyfit(lx, ly, 1)
    pred = p * lx + b
    ss_res = float(np.sum((ly - pred) ** 2)); ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return float(p), float(np.exp(b)), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def q0_arbitration(df):
    print("\n" + "=" * 100)
    print("=== 0. ARBITRAGEM: a celula N=50 / dt=0.01 ===")
    print("=" * 100)
    c = cell(df, grid="A", method="B2", N=50, dt=0.01)
    if c is None:
        print("  celula ausente"); return None
    print(f"  MEDIDO (grade A, {c['n']} seeds): mediana {c['median']:.3f} s  "
          f"[{c['min']:.3f}, {c['max']:.3f}]  R2 med {c['r2']:.3f}")
    for lab, v in (("escada P2 (commitada, 3 seeds, tudo fixado)", LADDER_N50_DT01),
                   ("largeN_results.csv (maio, arvore nao commitada, 2 seeds)", LARGEN_N50_DT01)):
        print(f"    vs {lab:<52} {v:6.3f}  -> desvio {100*(c['median']-v)/v:+7.1f}%")
    return c


def q1_exponent(df):
    print("\n" + "=" * 100)
    print("=== 1. EXPOENTE de tau_B2 vs N, por braco de dt ===")
    print("=" * 100)
    out = {}
    b2 = df[(df.method == "B2") & (df.grid.isin(["A", "A2"]))]
    for dt in sorted(b2.dt.unique()):
        ns, meds, rows = [], [], []
        for n in sorted(b2[np.isclose(b2.dt, dt)].N.unique()):
            c = cell(b2, N=n, dt=dt)
            if c is None:
                continue
            ns.append(n); meds.append(c["median"]); rows.append((n, c))
        if len(ns) < 2:
            continue
        p, a, r2 = loglog(ns, meds)
        # Ajuste so no nucleo (grade A, N<=100) para nao deixar a assintota dominar
        core = [(n, m) for n, m in zip(ns, meds) if n <= 100]
        p_core, a_core, r2_core = loglog([x[0] for x in core], [x[1] for x in core])
        out[dt] = {"p_all": p, "r2_all": r2, "p_core": p_core, "a_core": a_core,
                   "r2_core": r2_core, "ns": ns, "meds": meds}
        print(f"\n  --- dt = {dt:g} ---")
        print(f"  {'N':>5}{'n':>4}{'tau med':>10}{'[min-max]':>18}{'R2 fit':>8}")
        for n, c in rows:
            rng = "[%.2f-%.2f]" % (c["min"], c["max"])
            print(f"  {n:>5}{c['n']:>4}{c['median']:>10.3f}{rng:>18}{c['r2']:>8.3f}")
        print(f"  ajuste log-log  N<=100 : tau ~ {a_core:.3f} * N^{p_core:.3f}  (R2={r2_core:.3f})")
        if max(ns) > 100:
            print(f"  ajuste log-log  todos  : tau ~ {a:.3f} * N^{p:.3f}  (R2={r2:.3f})")
        # O que isso faz com a Lei 1
        tb100 = BASELINE_COEF * (100 ** BASELINE_EXP)
        t2_100 = a_core * (100 ** p_core)
        print(f"  Lei 1 -> A ~ N^({BASELINE_EXP:.2f} - {p_core:.2f}) = N^{BASELINE_EXP - p_core:.2f}")
        print(f"           em N=100: tau_base={tb100:.1f}s / tau_B2={t2_100:.2f}s -> A = {tb100/t2_100:.0f}x")
    return out


def q2_dt_invariance(df):
    print("\n" + "=" * 100)
    print("=== 2. INVARIANCIA EM dt (segundos) + GRADE C (controle do detector) ===")
    print("=" * 100)
    b2 = df[(df.method == "B2") & (df.grid.isin(["A", "A2"]))]
    print(f"  {'N':>5}{'dt=0.01':>11}{'dt=0.05':>11}{'razao':>9}{'CV%':>8}")
    for n in sorted(b2.N.unique()):
        c1, c5 = cell(b2, N=n, dt=0.01), cell(b2, N=n, dt=0.05)
        if not (c1 and c5):
            continue
        m = np.array([c1["median"], c5["median"]])
        cv = float(np.std(m) / np.mean(m) * 100)
        print(f"  {n:>5}{c1['median']:>11.3f}{c5['median']:>11.3f}"
              f"{c5['median']/c1['median']:>9.3f}{cv:>8.1f}{' *' if cv > 10 else ''}")

    gc = df[df.grid == "C"]
    if len(gc):
        print("\n  --- GRADE C: N=50, AGENT_STATE_TIMEOUT FIXO em 0.25 s ---")
        print(f"  {'dt':>7}{'tmo=5*dt (A)':>15}{'tmo=0.25 (C)':>15}{'delta':>9}")
        for dt in sorted(gc.dt.unique()):
            ca = cell(df, grid="A", method="B2", N=50, dt=dt)
            cc = cell(gc, method="B2", N=50, dt=dt)
            if not (ca and cc):
                continue
            print(f"  {dt:>7g}{ca['median']:>15.3f}{cc['median']:>15.3f}"
                  f"{cc['median']-ca['median']:>+9.3f}")
        print("  Se delta ~ 0 -> tau_fit e' imune a latencia de deteccao;")
        print("  se nao, a deriva de dt do B2 em N grande e' o detector, nao a fisica.")


def q3_c(df):
    print("\n" + "=" * 100)
    print("=== 3. c = segundos por salto / dt, e t_dissem vs c*(N/2)*dt ===")
    print("=" * 100)
    b2 = df[(df.method == "B2") & (df.grid.isin(["A", "A2"]))]
    print(f"  {'N':>5}{'dt':>7}{'ms/salto':>10}{'c':>7}{'R2':>7}"
          f"{'t_dissem':>10}{'c*(N/2)*dt':>12}{'(N/2)*dt':>10}{'N*dt':>8}{'cobert.':>9}{'maxhop':>8}")
    rows = []
    for n in sorted(b2.N.unique()):
        for dt in sorted(b2[b2.N == n].dt.unique()):
            g = agg(b2, N=n, dt=dt)
            if g.empty:
                continue
            sph = float(np.nanmedian(g["sec_per_hop"])); c = float(np.nanmedian(g["c_ticks_per_hop"]))
            r2 = float(np.nanmedian(g["hop_fit_r2"])); td = float(np.nanmedian(g["t_dissem"]))
            cov = float(np.nanmedian(g["coverage"])); mh = float(np.nanmedian(g["max_hop"]))
            pred = c * (n / 2.0) * dt
            print(f"  {n:>5}{dt:>7g}{sph*1000:>10.2f}{c:>7.2f}{r2:>7.3f}"
                  f"{td:>10.3f}{pred:>12.3f}{(n/2)*dt:>10.3f}{n*dt:>8.3f}{cov:>9.3f}{mh:>8.0f}")
            rows.append({"N": n, "dt": dt, "sec_per_hop": sph, "c": c, "hop_r2": r2,
                         "t_dissem": td, "pred_c_halfN_dt": pred, "coverage": cov,
                         "max_hop": mh})
    if rows:
        cs = np.array([r["c"] for r in rows], float); cs = cs[np.isfinite(cs)]
        print(f"\n  c global: mediana {np.median(cs):.3f}  [{cs.min():.3f}, {cs.max():.3f}]  n={cs.size}")
    return rows


def q4_crossover(df, expo):
    print("\n" + "=" * 100)
    print("=== 4. CROSSOVER: onde tau deixa de ser plano, e t_dissem vs tau ===")
    print("=" * 100)
    b2 = df[(df.method == "B2") & (df.grid.isin(["A", "A2"]))]
    print(f"  {'N':>5}{'dt':>7}{'tau':>9}{'t_dissem':>10}{'t_dis/tau':>11}")
    for n in sorted(b2.N.unique()):
        for dt in sorted(b2[b2.N == n].dt.unique()):
            c = cell(b2, N=n, dt=dt)
            g = agg(b2, N=n, dt=dt)
            if c is None or g.empty:
                continue
            td = float(np.nanmedian(g["t_dissem"]))
            print(f"  {n:>5}{dt:>7g}{c['median']:>9.3f}{td:>10.3f}{td/c['median']:>11.3f}")
    print("\n  t_dis/tau -> 1 marca o regime dominado por disseminacao.")


def make_figure(df, expo, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "figure.dpi": 150})
    b2 = df[(df.method == "B2") & (df.grid.isin(["A", "A2"]))]
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    colors = {0.01: "royalblue", 0.05: "firebrick"}
    for dt in sorted(b2.dt.unique()):
        ns, med, lo, hi = [], [], [], []
        for n in sorted(b2[np.isclose(b2.dt, dt)].N.unique()):
            c = cell(b2, N=n, dt=dt)
            if c is None:
                continue
            ns.append(n); med.append(c["median"]); lo.append(c["min"]); hi.append(c["max"])
        if not ns:
            continue
        col = colors.get(dt, "0.4")
        ax.errorbar(ns, med,
                    yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                    fmt="o-", color=col, lw=2.2, capsize=3, label=f"tau_B2, dt={dt:g}")
        e = expo.get(dt)
        if e and np.isfinite(e["p_core"]):
            xs = np.array(sorted(ns), float)
            ax.plot(xs, e["a_core"] * xs ** e["p_core"], ":", color=col, lw=1.4,
                    label=f"  ajuste N<=100: N^{e['p_core']:.2f} (R2={e['r2_core']:.2f})")
        # limites teoricos de disseminacao e a reta ajustada c*(N/2)*dt
        g = b2[np.isclose(b2.dt, dt)]
        c_med = float(np.nanmedian(g["c_ticks_per_hop"]))
        xs = np.array(sorted(b2.N.unique()), float)
        ax.plot(xs, (xs / 2.0) * dt, "--", color=col, lw=1.0, alpha=0.55,
                label=f"  (N/2)*dt, dt={dt:g}")
        ax.plot(xs, xs * dt, "-.", color=col, lw=1.0, alpha=0.4,
                label=f"  N*dt, dt={dt:g}")
        ax.plot(xs, c_med * (xs / 2.0) * dt, "-", color=col, lw=2.6, alpha=0.35,
                label=f"  AJUSTADA c*(N/2)*dt, c={c_med:.2f}")
        # cruzamento tau x disseminacao ajustada
        tau_i = np.interp(xs, sorted(ns), [m for _, m in sorted(zip(ns, med))])
        dis_i = c_med * (xs / 2.0) * dt
        s = np.sign(dis_i - tau_i)
        idx = np.where(np.diff(s) != 0)[0]
        if idx.size:
            i = idx[0]
            xc = xs[i] + (xs[i+1] - xs[i]) * abs(dis_i[i]-tau_i[i]) / (
                abs(dis_i[i]-tau_i[i]) + abs(dis_i[i+1]-tau_i[i+1]) + 1e-12)
            ax.axvline(xc, color=col, ls=":", lw=2.0, alpha=0.8)
            ax.annotate(f"crossover dt={dt:g}\nN~{xc:.0f}", (xc, ax.get_ylim()[0]*1.4),
                        color=col, fontsize=8.5, ha="center", fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("tempo [s]")
    ax.set_title("P3 — tau_B2 vs N: atuacao ou disseminacao?\n"
                 "barras = min/max de 3 seeds; retas finas = limites teoricos de disseminacao",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSalvo: {out_path}")


def main():
    df = load()
    print(f"{len(df)} rodadas de {df.source_file.nunique()} arquivo(s)")
    print(f"  grades={sorted(df.grid.unique())}  metodos={sorted(df.method.unique())}")
    print(f"  N={sorted(df.N.unique())}  dt={sorted(df.dt.unique())}  seeds={sorted(df.seed.unique())}")
    if "git_commit" in df.columns:
        print(f"  proveniencia: commit={sorted(df.git_commit.dropna().unique())} "
              f"dirty={sorted(map(str, df.git_dirty.dropna().unique()))}")

    q0_arbitration(df)
    expo = q1_exponent(df)
    q2_dt_invariance(df)
    rows = q3_c(df)
    q4_crossover(df, expo)

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\nEscrito: {OUT_CSV}")
    if os.environ.get("DTS_NOFIG", "").strip() not in ("1", "true", "True"):
        make_figure(df, expo, OUT_FIG)


if __name__ == "__main__":
    main()
