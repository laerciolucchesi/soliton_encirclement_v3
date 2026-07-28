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


def q1b_robust_metric(df):
    """t_settle -- a metrica PRIMARIA da campanha -- contra tau_fit, a secundaria.

    README.md:62-63 define t_settle como "enter-and-STAY settling time" e tau_fit
    como "secundario; so confie com R2 >= 0.9". Em N grande o R2 do tau_fit desaba,
    e a razao e' mecanica: o pico de E_gap CAI com N (uma morte perturba menos um
    anel maior) enquanto o residuo SOBE, de modo que o residuo se aproxima do piso
    de 5%-do-pico da janela de ajuste e a exponencial passa a ajustar a cauda plana.
    """
    print("\n" + "=" * 100)
    print("=== 1b. METRICA ROBUSTA (t_settle) vs METRICA PUBLICADA (tau_fit) ===")
    print("=" * 100)
    b2 = df[(df.method == "B2") & (df.grid == "A")]
    print(f"  {'N':>5}{'dt':>7}{'tau_fit':>9}{'R2':>7}{'t_settle':>10}"
          f"{'pico':>9}{'5%pico':>9}{'egap_fin':>10}{'fin/piso':>10}")
    for n in sorted(b2.N.unique()):
        for dt in sorted(b2[b2.N == n].dt.unique()):
            g = agg(b2, N=n, dt=dt)
            pk = float(g["egap_peak"].median()); fi = float(g["egap_final"].median())
            print(f"  {n:>5}{dt:>7g}{g['tau_fit'].median():>9.2f}"
                  f"{g['tau_fit_r2'].median():>7.3f}{g['t_settle'].median():>10.2f}"
                  f"{pk:>9.4f}{0.05*pk:>9.5f}{fi:>10.5f}{fi/(0.05*pk):>10.2f}")
    print("\n  fin/piso -> 1 significa que a janela do ajuste exponencial e' dominada")
    print("  pela cauda plana; o R2 cai em passo e o tau_fit deixa de ser um tempo.")
    out = {}
    for dt in sorted(b2.dt.unique()):
        g = b2[np.isclose(b2.dt, dt)]
        ns = sorted(g.N.unique())
        line = f"\n  --- dt={dt:g} ---"
        print(line)
        for col in ("tau_fit", "t_settle"):
            med = [float(g[g.N == n][col].median()) for n in ns]
            p, a, r2 = loglog(ns, med)
            out[(dt, col)] = {"p": p, "a": a, "r2": r2, "ns": ns, "med": med}
            print(f"    {col:9s} ~ {a:.4f} * N^{p:+.3f}  (R2={r2:.3f})   "
                  f"{[round(v,2) for v in med]}")
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


def make_figure(df, robust, out_path):
    """Dois paineis: a metrica PUBLICADA (tau_fit) e a PRIMARIA (t_settle).

    Sao a mesma grade lida de duas formas, e discordam -- que e' o resultado. As
    retas de disseminacao entram nos dois para localizar (ou descartar) o crossover.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
                         "legend.fontsize": 8, "figure.dpi": 150})
    b2 = df[(df.method == "B2") & (df.grid == "A")]
    colors = {0.01: "royalblue", 0.05: "firebrick"}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharex=True)

    panels = [
        ("tau_fit", axes[0], "(a) tau_fit — a metrica PUBLICADA\n"
                             "cresce, mas o R2 desaba: ajuste dominado pela cauda"),
        ("t_settle", axes[1], "(b) t_settle — a metrica PRIMARIA da campanha\n"
                              "robusta a forma do decaimento (README.md:62)"),
    ]
    for col_name, ax, title in panels:
        for dt in sorted(b2.dt.unique()):
            g = b2[np.isclose(b2.dt, dt)]
            ns = sorted(g.N.unique())
            med = [float(g[g.N == n][col_name].median()) for n in ns]
            lo = [float(g[g.N == n][col_name].min()) for n in ns]
            hi = [float(g[g.N == n][col_name].max()) for n in ns]
            c = colors.get(dt, "0.4")
            ax.errorbar(ns, med,
                        yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                        fmt="o-", color=c, lw=2.4, capsize=3, ms=7,
                        label=f"B2, dt={dt:g}")
            r = robust.get((dt, col_name))
            if r and np.isfinite(r["p"]):
                xs = np.array(ns, float)
                ax.plot(xs, r["a"] * xs ** r["p"], ":", color=c, lw=1.6,
                        label=f"   N^{r['p']:+.2f} (R2={r['r2']:.2f})")
            if col_name == "tau_fit":
                for n, m in zip(ns, med):
                    r2 = float(g[g.N == n]["tau_fit_r2"].median())
                    if r2 < 0.9:
                        ax.annotate(f"R2={r2:.2f}", (n, m), textcoords="offset points",
                                    xytext=(6, -12), fontsize=7.5, color=c,
                                    fontweight="bold")
            # disseminacao: a reta correta e' c*(N-1)*dt, nao c*(N/2)*dt --
            # o ultimo no a completar espera o SEGUNDO pulso, ou seja N-1 saltos.
            c_med = float(np.nanmedian(g["c_ticks_per_hop"]))
            xs = np.array(ns, float)
            ax.plot(xs, c_med * (xs - 1) * dt, "-", color=c, lw=3.0, alpha=0.28,
                    label=f"   disseminacao c*(N-1)*dt, c={c_med:.2f}")
            ax.plot(xs, (xs / 2.0) * dt, "--", color=c, lw=1.0, alpha=0.45,
                    label=f"   (N/2)*dt (limite teorico)")
            td = [float(g[g.N == n]["t_dissem"].median()) for n in ns]
            ax.plot(xs, td, "^--", color=c, lw=1.2, alpha=0.7, ms=5,
                    label="   t_dissem medido")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N"); ax.set_ylabel("tempo [s]")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(21, 118)          # sem isso a escala log deixa meia figura vazia
        ax.set_xticks([24, 32, 40, 50, 75, 100])
        ax.set_xticklabels(["24", "32", "40", "50", "75", "100"])
        ax.legend(loc="upper left", fontsize=7.5)

    fig.suptitle("P3 — o tempo de reconfiguracao do B2 e' plano em N? "
                 "Depende de qual metrica se usa.\n"
                 "falha unica permanente, comunicacao ideal, barras = min/max de 3 seeds",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSalvo: {out_path}")


def main():
    df = load()
    print(f"{len(df)} rodadas de {df.source_file.nunique()} arquivo(s)")
    print(f"  grades={sorted(df.grid.unique())}  metodos={sorted(df.method.unique())}")
    print(f"  N={sorted(df.N.unique())}  dt={sorted(df.dt.unique())}  seeds={sorted(df.seed.unique())}")
    if "git_commit" in df.columns:
        # Coage para str: a coluna pode misturar tipos quando um CSV antigo entra no merge.
        print(f"  proveniencia: commit={sorted(map(str, df.git_commit.dropna().unique()))} "
              f"dirty={sorted(map(str, df.git_dirty.dropna().unique()))}")

    q0_arbitration(df)
    expo = q1_exponent(df)
    robust = q1b_robust_metric(df)
    q2_dt_invariance(df)

    # Vantagem no MESMO codigo, pelas duas metricas (grade B = baseline N=50).
    bl = df[df.method == "baseline"]
    if len(bl):
        print("\n" + "=" * 100)
        print("=== VANTAGEM em N=50, mesmo codigo, pelas duas metricas ===")
        print("=" * 100)
        for dt in sorted(bl.dt.unique()):
            b = agg(bl, N=50, dt=dt); o = agg(df[(df.method == "B2") & (df.grid == "A")], N=50, dt=dt)
            if b.empty or o.empty:
                continue
            for col in ("tau_fit", "t_settle"):
                print(f"  dt={dt:g}  {col:9s}: baseline {b[col].median():7.2f} / "
                      f"B2 {o[col].median():5.2f} = {b[col].median()/o[col].median():5.1f}x")
    rows = q3_c(df)
    q4_crossover(df, expo)

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\nEscrito: {OUT_CSV}")
    if os.environ.get("DTS_NOFIG", "").strip() not in ("1", "true", "True"):
        make_figure(df, robust, OUT_FIG)


if __name__ == "__main__":
    main()
