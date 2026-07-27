#!/usr/bin/env python
"""Analise da escada de integracao regerada (E3'): tabela 6x3, historico, dt-invariancia.

Le ladder_results_*.csv (produzidos por run_ladder.py) e responde as tres perguntas
de fechamento do prompt E3':
  (i)   qual escala o "Option B" da proposta realmente usa;
  (ii)  B-min@0.5 x B-min@1.0 x B2@1.0 sustenta a narrativa do duplo-drive?
  (iii) tau e' invariante em dt nesta grade?

Tambem verifica se o CRITERIO DE ESTABILIDADE separa limpo (histograma de egap_final
e lista de casos de fronteira) -- se nao separar, o criterio precisa mudar e isso tem
de ser dito antes de aplica-lo.

Nao roda simulacao.

Uso:
    python experiments/scaling_law/analyze_ladder.py
    # env: LADDER_GLOB="ladder_results_*.csv"  LADDER_NOFIG="1"
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.environ.get("LADDER_GLOB", "ladder_results_*.csv")
OUT_CSV = os.path.join(EXP_DIR, "ladder_summary.csv")
OUT_FIG = os.path.join(EXP_DIR, "figures", "fig_ladder.png")

# Historico registrado em docs/thesis/tese_estrutura.md:55-60 (1 seed, 2026-05,
# arvore NAO commitada). tau_fit em segundos, N = 24/40/50.
HIST = {
    "V2_baseline_fixed": (7.08, 12.26, 140.1),
    "V1_baseline_norm":  (19.48, 54.79, 85.35),
    "V3_A_s05":          (11.63, 42.02, 74.71),
    "V4_Bmin_s05":       (3.27, 7.78, 12.20),
    "V5_Bmin_s10":       (16.51, 43.00, 62.59),
    "V6_B2_s10":         (2.17, 2.13, 2.12),
}
# Ordem de apresentacao = a ordem da escada, nao a alfabetica.
ORDER = ["V2_baseline_fixed", "V1_baseline_norm", "V3_A_s05",
         "V4_Bmin_s05", "V5_Bmin_s10", "V6_B2_s10"]
LABEL = {
    "V2_baseline_fixed": "baseline (ganho fixo 25)",
    "V1_baseline_norm":  "baseline (ganho estavel 250/N)",
    "V3_A_s05":          "Option A       scale 0.5",
    "V4_Bmin_s05":       "Option B-min   scale 0.5",
    "V5_Bmin_s10":       "Option B-min   scale 1.0",
    "V6_B2_s10":         "Option B2      scale 1.0",
}
NS = [24, 40, 50]
DEVIATION_FLAG = 0.10       # destaca celula que desvia mais de 10% do historico

# Criterio de estabilidade (o mesmo de run_ladder.py; repetido aqui para o relatorio).
SETTLED_EGAP_FINAL = 1e-2
SETTLED_LATE_STD = 1e-3
SETTLED_R2 = 0.80


def load():
    frames = []
    for p in sorted(glob.glob(os.path.join(EXP_DIR, PATTERN))):
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
    return df.drop_duplicates(subset=["variant", "N", "seed", "dt"], keep="last")


def cell(df, variant, n, dt, metric="tau_fit"):
    s = df[(df.variant == variant) & (df.N == n) & (np.isclose(df.dt, dt))]
    v = s[metric].to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    r2 = s["tau_fit_r2"].to_numpy(float)
    r2 = r2[np.isfinite(r2)]
    return {
        "n": int(v.size), "median": float(np.median(v)),
        "min": float(np.min(v)), "max": float(np.max(v)),
        "r2_median": float(np.median(r2)) if r2.size else float("nan"),
        "settled_frac": float(s["settled"].mean()) if "settled" in s else float("nan"),
        "n_settled": int(s["settled"].sum()) if "settled" in s else -1,
    }


def ladder_table(df, dt):
    print(f"\n{'='*100}\n=== ESCADA DE INTEGRACAO, dt={dt:g}  "
          f"(mediana [min-max] (R2 med), n seeds, settled/n) ===\n{'='*100}")
    print(f"{'variante':<32}" + "".join(f"{'N=' + str(n):>22}" for n in NS))
    rows = []
    for variant in ORDER:
        line = f"{LABEL[variant]:<32}"
        for i, n in enumerate(NS):
            c = cell(df, variant, n, dt)
            if c is None:
                line += f"{'--':>22}"
                continue
            hist = HIST[variant][i]
            dev = (c["median"] - hist) / hist if hist else float("nan")
            flag = "*" if abs(dev) > DEVIATION_FLAG else " "
            line += f"{c['median']:>10.2f}{flag}[{c['min']:.1f}-{c['max']:.1f}]".rjust(22)
            rows.append({"dt": dt, "variant": variant, "N": n, **c,
                         "hist": hist, "dev_frac": dev,
                         "flag_gt10pct": abs(dev) > DEVIATION_FLAG})
        print(line)
    print(f"\n--- historico -> medido (desvio) ---")
    print(f"{'variante':<32}" + "".join(f"{'N=' + str(n):>24}" for n in NS))
    for variant in ORDER:
        line = f"{LABEL[variant]:<32}"
        for i, n in enumerate(NS):
            c = cell(df, variant, n, dt)
            hist = HIST[variant][i]
            if c is None:
                line += f"{hist:>8.2f} ->      --".rjust(24)
            else:
                dev = (c["median"] - hist) / hist * 100.0
                flag = "*" if abs(dev) > DEVIATION_FLAG * 100 else " "
                line += f"{hist:>7.2f} ->{c['median']:>7.2f} ({dev:>+5.0f}%){flag}".rjust(24)
        print(line)
    print("\n  * = desvio > 10% do historico (tese_estrutura.md:55-60, 1 seed, arvore nao commitada)")
    print(f"  R2 medianos e contagem settled: ver {os.path.basename(OUT_CSV)}")
    return rows


def settled_diagnostics(df):
    print(f"\n{'='*100}\n=== CRITERIO DE ESTABILIDADE: separa limpo? ===\n{'='*100}")
    print(f"  settled = egap_final < {SETTLED_EGAP_FINAL:g} AND "
          f"egap_late_std < {SETTLED_LATE_STD:g} AND tau_fit_r2 > {SETTLED_R2:g}")

    for col, thr in (("egap_final", SETTLED_EGAP_FINAL),
                     ("egap_late_std", SETTLED_LATE_STD),
                     ("tau_fit_r2", SETTLED_R2)):
        v = df[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        print(f"\n  --- {col} (limiar {thr:g}) --- n={v.size}")
        lo, hi = float(np.min(v)), float(np.max(v))
        # Histograma log quando a faixa cobre ordens de magnitude.
        use_log = lo > 0 and hi / max(lo, 1e-12) > 100
        edges = (np.geomspace(lo, hi, 13) if use_log else np.linspace(lo, hi, 13))
        counts, edges = np.histogram(v, bins=edges)
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
            mark = " <== LIMIAR" if (e0 <= thr < e1) else ""
            bar = "#" * int(round(40 * c / max(counts.max(), 1)))
            print(f"    [{e0:11.3e}, {e1:11.3e}) {c:>4} {bar}{mark}")
        # Fronteira. Para grandezas positivas sem escala natural (egap_*), "perto"
        # e' multiplicativo: um fator 3 dos dois lados. Para R2, que vive em [0,1],
        # fator 3 abrangeria tudo -- ali a vizinhanca tem de ser ADITIVA.
        if col == "tau_fit_r2":
            near = df[(df[col] > thr - 0.10) & (df[col] < thr + 0.10)]
        else:
            near = df[(df[col] > thr / 3.0) & (df[col] < thr * 3.0)]
        if len(near):
            print(f"    FRONTEIRA (limiar/3 .. limiar*3): {len(near)} rodada(s)")
            for _, r in near.sort_values(col).iterrows():
                print(f"      {r['variant']:<18} N={int(r['N']):>2} s={int(r['seed'])} "
                      f"dt={r['dt']:g}  {col}={r[col]:.5g}  "
                      f"(settled={bool(r['settled'])})")
        else:
            print(f"    FRONTEIRA: nenhuma rodada dentro de um fator 3 do limiar")

    print("\n  --- resultado do criterio por variante ---")
    print(f"{'variante':<32}{'settled':>10}{'total':>7}   não-settled em")
    for variant in ORDER:
        s = df[df.variant == variant]
        if not len(s):
            continue
        ns = s[~s.settled.astype(bool)]
        # r["dt"] e nao r.dt: em pandas, .dt e' o acessor datetime, nao a coluna.
        where = ", ".join(sorted({f"N{int(r['N'])}/dt{r['dt']:g}" for _, r in ns.iterrows()}))
        print(f"{LABEL[variant]:<32}{int(s.settled.sum()):>10}{len(s):>7}   {where}")


def dt_invariance(df):
    dts = sorted(df.dt.unique())
    print(f"\n{'='*100}\n=== (iii) tau e' invariante em dt? dts presentes: {dts} ===\n{'='*100}")
    if len(dts) < 2:
        print("  Apenas um dt na grade -- a replica ainda nao rodou/terminou.")
        return []
    rows = []
    print(f"{'variante':<32}{'N':>4}" + "".join(f"{'dt=' + f'{d:g}':>12}" for d in dts)
          + f"{'razao':>9}{'CV%':>7}")
    for variant in ORDER:
        for n in NS:
            meds = []
            for d in dts:
                c = cell(df, variant, n, d)
                meds.append(c["median"] if c else float("nan"))
            meds = np.asarray(meds, float)
            if np.sum(np.isfinite(meds)) < 2:
                continue
            fin = meds[np.isfinite(meds)]
            ratio = fin[-1] / fin[0] if fin[0] else float("nan")
            cv = float(np.std(fin) / np.mean(fin) * 100.0)
            flag = " *" if cv > 10.0 else ""
            print(f"{LABEL[variant]:<32}{n:>4}"
                  + "".join(f"{m:>12.2f}" for m in meds)
                  + f"{ratio:>9.2f}{cv:>7.1f}{flag}")
            rows.append({"variant": variant, "N": n, "dts": dts,
                         "medians": list(meds), "ratio": ratio, "cv_pct": cv})
    print("\n  * = CV > 10% entre dts (a lei preve invariancia em SEGUNDOS; CV<5% foi o")
    print("      valor medido na campanha Ciclo 1)")
    return rows


def closing_questions(df):
    dt0 = min(df.dt.unique())
    print(f"\n{'='*100}\n=== PERGUNTAS DE FECHAMENTO (dt={dt0:g}) ===\n{'='*100}")

    def med(v, n):
        c = cell(df, v, n, dt0)
        return c["median"] if c else float("nan")

    print("\n(i) Qual escala o 'Option B' da proposta realmente usa?")
    print("    A tabela publicada (draft v1, 5-preliminary-results.tex:30) rotula a linha")
    print("    'Feedforward (B), scale 0,5' com 3,27 / 7,78 / 12,20.")
    for n, h in zip(NS, HIST["V4_Bmin_s05"]):
        print(f"      N={n:>2}: historico {h:6.2f} | B-min@0.5 medido {med('V4_Bmin_s05', n):6.2f} "
              f"| B-min@1.0 medido {med('V5_Bmin_s10', n):6.2f}")

    print("\n(ii) A narrativa do duplo-drive se sustenta?")
    print("     Esperado: B-min@1.0 MUITO pior que B-min@0.5 (over-drive), e B2@1.0")
    print("     recuperando tudo (vies cancelador completo remove o duplo-drive).")
    for n in NS:
        b05, b10, b2 = med('V4_Bmin_s05', n), med('V5_Bmin_s10', n), med('V6_B2_s10', n)
        print(f"      N={n:>2}: B-min@0.5={b05:6.2f}  B-min@1.0={b10:6.2f}  B2@1.0={b2:5.2f}   "
              f"| B10/B05={b10 / b05 if b05 else float('nan'):5.2f}x  "
              f"B10/B2={b10 / b2 if b2 else float('nan'):6.2f}x")

    print("\n(iii) ver a secao de dt-invariancia acima.")


def make_figure(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "figure.dpi": 150})
    dts = sorted(df.dt.unique())
    colors = {"V2_baseline_fixed": "darkorange", "V1_baseline_norm": "firebrick",
              "V3_A_s05": "purple", "V4_Bmin_s05": "seagreen",
              "V5_Bmin_s10": "olive", "V6_B2_s10": "royalblue"}

    ncol = 1 + (1 if len(dts) > 1 else 0)
    fig, axes = plt.subplots(1, 1 + ncol, figsize=(6.0 * (1 + ncol), 4.6))
    axes = np.atleast_1d(axes)

    # (a) a escada, medido vs historico
    ax = axes[0]
    dt0 = dts[0]
    for variant in ORDER:
        med = [cell(df, variant, n, dt0)["median"] if cell(df, variant, n, dt0) else np.nan
               for n in NS]
        lo = [cell(df, variant, n, dt0)["min"] if cell(df, variant, n, dt0) else np.nan for n in NS]
        hi = [cell(df, variant, n, dt0)["max"] if cell(df, variant, n, dt0) else np.nan for n in NS]
        ax.plot(NS, med, "o-", color=colors[variant], lw=2, label=LABEL[variant])
        ax.fill_between(NS, lo, hi, color=colors[variant], alpha=0.15)
        ax.plot(NS, HIST[variant], "x--", color=colors[variant], lw=1.0, alpha=0.7,
                markersize=7)
    ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("tau_fit [s]")
    ax.set_title(f"(a) A escada, dt={dt0:g}\nlinha cheia = medido, x tracejado = historico",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=7.5)

    # (b) desvio relativo ao historico
    ax = axes[1]
    width = 0.13
    for k, variant in enumerate(ORDER):
        devs = []
        for i, n in enumerate(NS):
            c = cell(df, variant, n, dt0)
            devs.append((c["median"] - HIST[variant][i]) / HIST[variant][i] * 100.0
                        if c else np.nan)
        ax.bar(np.arange(len(NS)) + (k - 2.5) * width, devs, width,
               color=colors[variant], label=LABEL[variant])
    ax.axhline(0, color="black", lw=1.0)
    for y in (-10, 10):
        ax.axhline(y, color="0.4", ls="--", lw=1.0)
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_ylabel("desvio do historico [%]")
    ax.set_title("(b) Reproduz o historico?\ntracejado = +-10%", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # (c) dt-invariancia
    if len(dts) > 1:
        ax = axes[2]
        for variant in ORDER:
            for i, n in enumerate(NS):
                meds = [cell(df, variant, n, d)["median"] if cell(df, variant, n, d) else np.nan
                        for d in dts]
                if np.sum(np.isfinite(meds)) < 2:
                    continue
                ax.plot(dts, meds, "o-", color=colors[variant], lw=1.4,
                        alpha=0.55 + 0.15 * i,
                        label=LABEL[variant] if i == 0 else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("CONTROL_PERIOD [s]"); ax.set_ylabel("tau_fit [s]")
        ax.set_title("(c) tau invariante em dt?\nlinha horizontal = invariante",
                     fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=7)

    fig.suptitle("E3' - escada de integracao regerada em codigo commitado, tudo fixado",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSalvo: {out_path}")


def main():
    df = load()
    print(f"{len(df)} rodadas de {df.source_file.nunique()} arquivo(s)")
    print(f"  variantes={sorted(df.variant.unique())}")
    print(f"  N={sorted(df.N.unique())}  seeds={sorted(df.seed.unique())}  "
          f"dt={sorted(df.dt.unique())}")
    if "git_commit" in df.columns:
        print(f"  proveniencia: commit={sorted(df.git_commit.dropna().unique())} "
              f"dirty={sorted(map(str, df.git_dirty.dropna().unique()))}")

    rows = []
    for dt in sorted(df.dt.unique()):
        rows += ladder_table(df, dt)
    settled_diagnostics(df)
    dt_invariance(df)
    closing_questions(df)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nEscrito: {OUT_CSV}  ({len(out)} linhas)")

    if os.environ.get("LADDER_NOFIG", "").strip() not in ("1", "true", "True"):
        make_figure(df, OUT_FIG)


if __name__ == "__main__":
    main()
