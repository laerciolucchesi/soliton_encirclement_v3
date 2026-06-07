#!/usr/bin/env python
"""Fase 2 -- analisa o colapso adimensional (suporta multi-seed).

Le um collapse_results[_TAG].csv (via env COLLAPSE_RESULTS). Se houver coluna 'seed',
agrega por MEDIANA entre seeds por celula (method,N,tau_a,dt) e reporta o spread.
Verifica as leis dt-invariantes:  tau_base = a*N^2 ,  tau_B2 = b*tau_a , logo A ~ (a/b)*N^2/tau_a.
Ajusta o EXPOENTE de tau_base vs N (log-log) e o colapso de A vs N^2/tau_a.

Uso:
    python experiments/scaling_law/analyze_collapse.py
    # ex.: $env:COLLAPSE_RESULTS="collapse_results_ms.csv"
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(EXP_DIR, os.environ.get("COLLAPSE_RESULTS", "collapse_results.csv"))


def aggregate(df):
    """Mediana de tau_fit entre seeds por (method,N,tau_xy,dt). Retorna df agregado + spread."""
    if "seed" not in df.columns:
        df = df.assign(seed=0)
    g = df.groupby(["method", "N", "tau_xy", "dt"])
    agg = g["tau_fit"].agg(["median", "std", "count"]).reset_index()
    agg = agg.rename(columns={"median": "tau", "std": "tau_std", "count": "nseed"})
    # T_FF e r2 medianos p/ referencia
    agg["T_FF"] = g["T_FF"].median().values
    agg["r2"] = g["tau_fit_r2"].median().values
    return agg


def cells_from(agg):
    rows = []
    for (n, tau, dt), gg in agg.groupby(["N", "tau_xy", "dt"]):
        b = gg[gg.method == "baseline"]; o = gg[gg.method == "B2"]
        if not len(b) or not len(o):
            continue
        tb = float(b["tau"].iloc[0]); to = float(o["tau"].iloc[0])
        if not (np.isfinite(tb) and np.isfinite(to) and to > 0):
            continue
        tff = float(o["T_FF"].iloc[0])
        rows.append({
            "N": int(n), "tau_xy": float(tau), "dt": float(dt), "nseed": int(o["nseed"].iloc[0]),
            "Q": n * n / tau, "Pe": n * dt / tau, "tau_base": tb, "tau_B2": to, "A": tb / to,
            "tb_std": float(b["tau_std"].iloc[0]), "to_std": float(o["tau_std"].iloc[0]),
            "a": tb / (n * n), "b": to / tff if tff > 0 else np.nan, "r2_B2": float(o["r2"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("Q").reset_index(drop=True)


def main():
    if not os.path.exists(RESULTS):
        print(f"NAO encontrei {RESULTS}."); return
    raw = pd.read_csv(RESULTS)
    agg = aggregate(raw)
    cells = cells_from(agg)
    if cells.empty:
        print("Sem pares baseline+B2."); return

    pd.set_option("display.width", 200)
    print(f"\n=== {os.path.basename(RESULTS)}: celulas (mediana entre seeds) ===")
    print(cells[["N","tau_xy","dt","nseed","Q","Pe","tau_base","tb_std","tau_B2","to_std","A","a","b","r2_B2"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    good = cells[cells.r2_B2 >= 0.88]
    a, b = good["a"], good["b"]
    print("\n=== LEIS-COMPONENTE (R2_B2>=0.88) ===")
    print(f"  a = tau_base/N^2 = {a.mean():.4f} +/- {a.std():.4f} s   (CV {100*a.std()/a.mean():.1f}%)")
    print(f"  b = tau_B2/tau_a = {b.mean():.3f} +/- {b.std():.3f}     (CV {100*b.std()/b.mean():.1f}%)")
    print(f"  => A ~ {a.mean()/b.mean():.4f} * N^2/tau_a")
    for grp, lbl in (("Q", "N^2/tau_a"), ("Pe", "N*dt/tau_a")):
        k = good["A"] / good[grp]
        print(f"  colapso A/({lbl:11s}): CV {100*k.std()/k.mean():.1f}%")

    # expoente de tau_base vs N (log-log), por tau_a (usa medianas)
    print("\n=== EXPOENTE de tau_base vs N (log-log) ===")
    for tau, gg in cells.groupby("tau_xy"):
        gg = gg.drop_duplicates("N").sort_values("N")
        if len(gg) >= 2:
            p = np.polyfit(np.log(gg["N"]), np.log(gg["tau_base"]), 1)
            print(f"  tau_a={tau:g}: tau_base ~ N^{p[0]:.2f}  (N={list(gg['N'])})")

    # spread multi-seed
    ns = cells["nseed"].max()
    if ns > 1:
        cvb = (cells["to_std"] / cells["tau_B2"]).replace([np.inf], np.nan).dropna()
        print(f"\n=== MULTI-SEED ({ns} seeds): CV(tau_B2) entre seeds = {100*cvb.mean():.1f}% (medio) ===")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for tau, gg in cells.groupby("tau_xy"):
        gg = gg.sort_values("Q")
        ax.plot(gg["Q"], gg["A"], "o", ms=7, label=f"τ_a={tau:g}")
    qq = np.array(sorted(cells["Q"].unique()))
    ax.plot(qq, (a.mean()/b.mean()) * qq, "k--", alpha=0.6, label=f"A={a.mean()/b.mean():.3f}·N²/τ_a")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Q = N² / τ_a"); ax.set_ylabel("vantagem A = τ_base/τ_B2")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.6)
    ax.set_title("Colapso da vantagem vs N²/τ_a (multi-seed)")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
    fig.tight_layout()
    out = os.path.join(EXP_DIR, "collapse_advantage" + ("_ms" if "seed" in raw.columns and ns > 1 else "") + ".png")
    fig.savefig(out, dpi=130)
    print(f"\nSalvo: {out}")


if __name__ == "__main__":
    main()
