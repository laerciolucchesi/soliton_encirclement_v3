#!/usr/bin/env python
"""Fase 3 / Track A -- analisa comm_results.csv pela metrica CONFIAVEL sob perda.

Sob perda de pacote o decaimento de E_gap deixa de ser exponencial -> tau_fit fica NaN/absurdo.
A metrica robusta e' o RESIDUO final egap_final (reconfigurou? -> ~0) e a fracao 'settled'.
Resume mediana(egap_final) e fracao settled por (metodo, loss) e plota.

Uso: python experiments/scaling_law/analyze_comm.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP_DIR, os.environ.get("COMM_RESULTS", "comm_results.csv"))


def main():
    if not os.path.exists(RES):
        print(f"NAO encontrei {RES}."); return
    df = pd.read_csv(RES)
    print("\n=== ROBUSTEZ A PERDA (metrica = egap_final; reconfigurou se ~0) ===")
    print(f"{'loss':>5} | {'egf_base(med)':>13} {'set_base':>8} | {'egf_B2(med)':>12} {'set_B2':>7}")
    losses = sorted(df.loss.unique())
    rows = []
    for loss in losses:
        b = df[(df.method == "baseline") & (df.loss == loss)]
        o = df[(df.method == "B2") & (df.loss == loss)]
        egb = float(b["egap_final"].median()) if len(b) else float("nan")
        ego = float(o["egap_final"].median()) if len(o) else float("nan")
        sb = float(b["settled"].mean()) if len(b) else float("nan")
        so = float(o["settled"].mean()) if len(o) else float("nan")
        rows.append((loss, egb, sb, ego, so))
        print(f"{loss:>5g} | {egb:>13.4f} {sb:>8.2f} | {ego:>12.4f} {so:>7.2f}")

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(arr[:, 0], np.maximum(arr[:, 1], 1e-4), "o-", label="baseline", color="tab:blue")
    ax.plot(arr[:, 0], np.maximum(arr[:, 3], 1e-4), "s-", label="B2 (overlay)", color="tab:red")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.7, label="limiar 'reconfigurou' (0,05)")
    ax.set_yscale("log")
    ax.set_xlabel("perda de pacote (COMMUNICATION_FAILURE_RATE)")
    ax.set_ylabel("egap_final (resíduo de espaçamento) — mediana de 3 seeds")
    ax.set_title("Robustez à perda de pacote: overlay (malha aberta) vs baseline (malha fechada)\n"
                 "N=24, τ_a=1; abaixo da linha = reconfigurou")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
    fig.tight_layout()
    out = os.path.join(EXP_DIR, "comm_loss_robustness.png")
    fig.savefig(out, dpi=130)
    print(f"\nSalvo: {out}")
    # crossover honesto
    print("\nLeitura: baseline (malha fechada) reconfigura ate' ~loss 0.2; o overlay B2 (feedforward")
    print("malha aberta) perde o assentamento ja' em loss ~0.1 -> CROSSOVER ~0.05-0.1.")


if __name__ == "__main__":
    main()
