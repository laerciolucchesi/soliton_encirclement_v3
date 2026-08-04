#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A CENSURA do t_close como RESULTADO: existe tempo de ganhar entre eventos?

A censura de t_close (a brecha nao fecha antes do proximo evento) nao e' um
obstaculo de medicao -- e' a medida direta do grupo adimensional Pi_2: "ha tempo
de ganhar entre eventos?". E ela e' medivel EXATAMENTE onde t_close nao e',
inclusive na taxa 48, que foi excluida da analise por evento do pico.

Produz:
  * fracao de censura x Pi_2', nas QUATRO taxas, para dois limiares;
  * decomposicao de SINAL da diferenca pareada da fracao-da-janela-em-brecha
    (positivos / empates / negativos), com teste de sinal nao parametrico.

Uso:
    python experiments/scaling_law/analysis_churn/fig_censura.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LANG = os.environ.get("FIG_LANG", "pt").strip().lower()
OUT_DIR = os.environ.get("FIG_OUT_DIR", HERE)


def T(pt, en):
    """Seletor de texto da FIGURA: default pt (saida identica a committada);
    FIG_LANG=en da a variante em ingles para o draft v3. NENHUM valor muda aqui
    -- so idioma dos rotulos (e virgula decimal -> ponto nas literais traduzidas)."""
    return en if LANG == "en" else pt

RUNS = os.path.join(HERE, "rerun_runs")
T0_WINDOW, W_MIN, W_MAX, T_OFF, N_AG = 20.0, 0.6, 1.5, 8.0, 24.0
THRS = (1.25, 1.5)
BASE, OVER = "baseline", "B2"
_log = []


def say(m=""):
    print(m)
    _log.append(str(m))


def pi2(rate):
    return (rate / 60.0) * T_OFF


def extract():
    rows = []
    for d in sorted(os.listdir(RUNS)):
        tgt = os.path.join(RUNS, d, "target_telemetry.csv")
        evp = os.path.join(RUNS, d, "events.csv")
        if not (os.path.exists(tgt) and os.path.exists(evp)):
            continue
        meth = OVER if d.startswith("dual_pulse") else BASE
        rate = float(d.split("_rate")[1].split("_s")[0])
        seed = int(d.split("_s")[-1])
        tel, ev = pd.read_csv(tgt), pd.read_csv(evp)
        t = tel["timestamp"].to_numpy(float)
        g = tel["G_max"].to_numpy(float)
        fs = ev[ev.event_type == "failure_start"]
        allev = np.sort(np.concatenate([
            fs["timestamp"].to_numpy(float),
            ev[ev.event_type == "failure_end"]["timestamp"].to_numpy(float)]))
        for tf, nid in sorted(zip(fs["timestamp"].to_numpy(float),
                                  fs["node_id"].to_numpy(int))):
            nxt = allev[allev > tf + 1e-9]
            gap_next = float(nxt[0] - tf) if nxt.size else float("inf")
            # DUAS janelas, e a distincao e' essencial:
            #  w_cens = ate o PROXIMO EVENTO (sem teto). E' a janela certa para a
            #     pergunta "fechou antes do proximo evento?" = Pi_2. Cortar em
            #     W_MAX mediria "fechou em 1,5 s", que e' outra pergunta e daria
            #     censura ~100% em qualquer taxa por construcao.
            #  w_adapt = clip(gap_next, W_MIN, W_MAX). E' a janela do PICO, onde o
            #     teto existe para conter contaminacao entre eventos.
            w_cens = min(gap_next, float(t[-1]) - tf)
            w_adapt = float(np.clip(gap_next, W_MIN, W_MAX))
            if tf < T0_WINDOW or w_cens <= 0 or tf + w_adapt > t[-1]:
                continue
            idx_c = np.where((t > tf) & (t <= tf + w_cens))[0]
            idx_a = np.where((t > tf) & (t <= tf + w_adapt))[0]
            if idx_c.size < 3 or idx_a.size < 3:
                continue
            r = {"method": meth, "rate_total": rate, "seed": seed,
                 "t_fail": float(tf), "node": int(nid),
                 "w_cens": w_cens, "w_adapt": w_adapt, "gap_next": gap_next}
            for thr in THRS:
                ac_, aa_ = g[idx_c] > thr, g[idx_a] > thr
                r[f"cens_{thr:g}"] = bool(ac_[-1])          # janela ate o proximo evento
                r[f"censW_{thr:g}"] = bool(aa_[-1])         # janela com teto (contraste)
                r[f"frac_{thr:g}"] = float(np.mean(aa_))    # fracao usa a janela do pico
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"{RUNS} nao existe.")
    ev = extract()
    key = ["rate_total", "seed", "t_fail", "node"]
    a = ev[ev.method == BASE].set_index(key).sort_index()
    b = ev[ev.method == OVER].set_index(key).sort_index()
    common = a.index.intersection(b.index)
    A_, B_ = a.loc[common], b.loc[common]
    pair = pd.DataFrame({"rate_total": [k[0] for k in common],
                         "seed": [k[1] for k in common]})
    pair["gap_next"] = A_["gap_next"].to_numpy(float)
    for thr in THRS:
        for meth_tag, src in (("base", A_), ("b2", B_)):
            pair[f"cens_{thr:g}_{meth_tag}"] = src[f"cens_{thr:g}"].to_numpy(bool)
        pair[f"cens_{thr:g}"] = (A_[f"cens_{thr:g}"].to_numpy(bool)
                                 | B_[f"cens_{thr:g}"].to_numpy(bool))
        pair[f"censW_{thr:g}"] = (A_[f"censW_{thr:g}"].to_numpy(bool)
                                  | B_[f"censW_{thr:g}"].to_numpy(bool))
        pair[f"d_frac_{thr:g}"] = (A_[f"frac_{thr:g}"].to_numpy(float)
                                   - B_[f"frac_{thr:g}"].to_numpy(float))
    pair.to_csv(os.path.join(HERE, "censura_paired.csv"), index=False)

    rates = sorted(pair.rate_total.unique())
    say("=" * 78)
    say("A CENSURA COMO RESULTADO: ha tempo de ganhar entre eventos?")
    say("=" * 78)
    say("Censura = a brecha NAO fechou antes do proximo evento. E' a medida direta de")
    say("Pi_2 ('ha tempo de ganhar entre eventos?'), e e' medivel inclusive na taxa 48,")
    say("onde o pico do evento NAO e' identificavel.")
    say("")
    say("ARMADILHA EVITADA: medir a censura dentro da janela COM TETO (1,5 s) daria ~100%")
    say("em qualquer taxa por construcao -- estaria medindo 'fechou em 1,5 s', nao 'fechou")
    say("antes do proximo evento'. As duas colunas abaixo mostram a diferenca.")
    say("")
    say(f"{'taxa':>6}{'Pi_2':>8}{'pares':>8}{'inter_med':>11}"
        + "".join(f"{'cens>'+f'{t:g}':>12}" for t in THRS)
        + "".join(f"{'[teto]>'+f'{t:g}':>13}" for t in THRS))
    cens = {t: [] for t in THRS}
    censW = {t: [] for t in THRS}
    for r in rates:
        s = pair[pair.rate_total == r]
        line = (f"{r:>6g}{pi2(r):>8.2f}{len(s):>8}"
                f"{np.median(s['gap_next'].replace(np.inf, np.nan).dropna()):>11.2f}")
        for t in THRS:
            f = float(s[f"cens_{t:g}"].mean())
            cens[t].append(f)
            line += f"{f:>12.4f}"
        for t in THRS:
            f = float(s[f"censW_{t:g}"].mean())
            censW[t].append(f)
            line += f"{f:>13.4f}"
        say(line)
    say("")
    say("'cens' = janela ate o proximo evento (a medida certa de Pi_2).")
    say("'[teto]' = janela cortada em 1,5 s (contraste; mede outra coisa).")
    say("")
    say(f"{'taxa':>6}{'Pi_2':>8}" + "".join(f"{'base>'+f'{t:g}':>12}{'B2>'+f'{t:g}':>12}" for t in THRS))
    for r in rates:
        s = pair[pair.rate_total == r]
        line = f"{r:>6g}{pi2(r):>8.2f}"
        for t in THRS:
            line += (f"{float(s[f'cens_{t:g}_base'].mean()):>12.4f}"
                     f"{float(s[f'cens_{t:g}_b2'].mean()):>12.4f}")
        say(line)

    # ---- decomposicao de sinal -------------------------------------------
    say("")
    say("=" * 78)
    say("FRACAO DA JANELA EM BRECHA -- decomposicao de SINAL (sem media, sem modelo)")
    say("=" * 78)
    sign = {}
    for thr in THRS:
        x = pair[f"d_frac_{thr:g}"].to_numpy(float)
        pos, neg = int((x > 1e-12).sum()), int((x < -1e-12).sum())
        tie, nz = len(x) - pos - neg, pos + neg
        bt = stats.binomtest(pos, nz, 0.5) if nz else None
        sign[thr] = (pos, tie, neg)
        say("")
        say(f"-- limiar G_max > {thr:g}  (n={len(x)} pares) --")
        say(f"   B2 sai da brecha e baseline NAO : {pos:>4}  ({pos/len(x):.2%})")
        say(f"   EMPATE (nenhum dos dois sai)    : {tie:>4}  ({tie/len(x):.2%})")
        say(f"   baseline sai e B2 NAO           : {neg:>4}  ({neg/len(x):.2%})")
        if bt:
            ci = bt.proportion_ci()
            say(f"   teste de SINAL sobre os {nz} nao-empatados: {pos}/{nz} a favor de B2")
            say(f"   proporcao {pos/nz:.3f}, IC95 [{ci.low:.3f}, {ci.high:.3f}], p={bt.pvalue:.3g}")
        say(f"   {'taxa':>5}{'n':>6}{'B2 sai':>8}{'empate':>9}{'base sai':>10}{'p sinal':>10}"
            f"{'prop':>8}")
        per_rate = {}
        for r in rates:
            s = pair[pair.rate_total == r][f"d_frac_{thr:g}"].to_numpy(float)
            p_, n_ = int((s > 1e-12).sum()), int((s < -1e-12).sum())
            pv = stats.binomtest(p_, p_ + n_, 0.5).pvalue if (p_ + n_) else float("nan")
            pr = p_ / (p_ + n_) if (p_ + n_) else float("nan")
            per_rate[r] = (p_, n_, pv, pr)
            say(f"   {r:>5g}{len(s):>6}{p_:>8}{len(s)-p_-n_:>9}{n_:>10}{pv:>10.3g}{pr:>8.3f}")
        # veredito POR TAXA antes do agregado (regra fixada na analise do pico)
        n_sig = sum(1 for v in per_rate.values() if np.isfinite(v[2]) and v[2] < 0.05)
        say("")
        if n_sig == len(rates):
            say("   => CONCORDANTE: efeito na mesma direcao em TODAS as taxas. Resultado NAO")
            say("      PARAMETRICO, imune a inflacao do pipeline: sem media, sem modelo, sem")
            say("      necessidade de calibracao por nula sintetica.")
        elif n_sig == 0:
            say("   => positivos e negativos em proporcao semelhante em todas as taxas: RUIDO.")
        else:
            drivers = [f"{r:g}" for r, v in per_rate.items()
                       if np.isfinite(v[2]) and v[2] < 0.05]
            say(f"   => DISCORDANTE entre taxas: o efeito aparece so em {drivers} e some nas")
            say("      demais. Pela regra fixada, a DISCORDANCIA e' o resultado -- o agregado")
            say("      NAO e' interpretavel e nao deve ser citado sozinho.")

    # ---- figura -----------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    x = [pi2(r) for r in rates]
    for thr, color, mk in zip(THRS, ("firebrick", "royalblue"), ("o", "s")):
        axA.plot(x, cens[thr], marker=mk, color=color, lw=2.0,
                 label=T(f"limiar $G_{{max}}$ > {thr:g}", f"threshold $G_{{max}}$ > {thr:g}"))
    axA.axhline(1.0, ls=":", color="0.5", lw=1.2)
    axA.set_xscale("log"); axA.set_ylim(-0.03, 1.08)
    axA.set_xlabel(T(r"$\Pi_2'$ = $\lambda_{anel}\cdot T_{off}$  (agentes ausentes)",
                    r"$\Pi_2'$ = $\lambda_{ring}\cdot T_{off}$  (absent agents)"))
    axA.set_ylabel(T("fracao de eventos CENSURADOS", "fraction of CENSORED events"))
    axA.set_title(T("A) a brecha nao fecha antes do proximo evento\n"
                    "(censura = medida direta de $\\Pi_2$)",
                    "A) the breach does not close before the next event\n"
                    "(censoring = a direct measure of $\\Pi_2$)"), fontweight="bold")
    axA.legend(loc="lower right"); axA.grid(alpha=0.3)
    for xi, r in zip(x, rates):
        axA.annotate(f"{r:g}/min", (xi, 0.02), ha="center", fontsize=7.5, color="0.35")
    axA.annotate(T("taxa 48: EXCLUIDA da analise do pico,\nmedivel aqui",
                   "rate 48: EXCLUDED from the peak analysis,\nmeasurable here"),
                 (x[-1], cens[THRS[1]][-1]), xytext=(-8, -52),
                 textcoords="offset points", fontsize=7.5, color="0.3",
                 ha="right", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))

    w = 0.35
    ypos = np.arange(len(THRS))
    for i, thr in enumerate(THRS):
        pos, tie, neg = sign[thr]
        tot = pos + tie + neg
        axB.barh(i, pos / tot, w, color="seagreen",
                 label=T("B2 sai, baseline nao", "B2 exits, baseline does not") if i == 0 else None)
        axB.barh(i, tie / tot, w, left=pos / tot, color="0.8",
                 label=T("empate (nenhum sai)", "tie (neither exits)") if i == 0 else None)
        axB.barh(i, neg / tot, w, left=(pos + tie) / tot, color="firebrick",
                 label=T("baseline sai, B2 nao", "baseline exits, B2 does not") if i == 0 else None)
        axB.annotate(f"{pos}", (pos / tot / 2, i), ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
        axB.annotate(f"{neg}", (1 - neg / tot / 2, i), ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold")
        nz = pos + neg
        axB.annotate(T(f"{pos}/{nz} = {pos/nz:.0%} a favor de B2 entre os nao-empatados",
                       f"{pos}/{nz} = {pos/nz:.0%} in favor of B2 among the non-ties"),
                     (0.5, i + 0.26), ha="center", fontsize=8, color="0.25")
    axB.set_yticks(ypos)
    axB.set_yticklabels([f"$G_{{max}}$ > {t:g}" for t in THRS])
    axB.invert_yaxis(); axB.set_xlim(0, 1)
    axB.set_xlabel(T("fracao dos pares de eventos", "fraction of event pairs"))
    axB.set_title(T("B) quem sai da brecha primeiro, no mesmo evento\n"
                    "(proporcao; sem media, sem modelo)",
                    "B) who exits the breach first, in the same event\n"
                    "(proportions; no mean, no model)"), fontweight="bold")
    axB.legend(fontsize=7.5, loc="center right", framealpha=0.95)
    axB.grid(alpha=0.3, axis="x")

    fig.suptitle(T("A censura do $t_{close}$ e' o resultado, nao o obstaculo",
                   "The censoring of $t_{close}$ is the result, not the obstacle"),
                 fontsize=12.5, fontweight="bold")
    p125, _t125, n125 = sign[1.25]
    p15, _t15, n15 = sign[1.5]
    # per-taxa do limiar 1,5, para nao esconder a discordancia no rodape
    pr15 = []
    for r in rates:
        s = pair[pair.rate_total == r]["d_frac_1.5"].to_numpy(float)
        a_, b_ = int((s > 1e-12).sum()), int((s < -1e-12).sum())
        pr15.append(f"{r:g}/min: {a_}v{b_}")
    foot = T(
        f"LIMIAR 1,25: {p125}/{p125+n125} = {p125/(p125+n125):.0%} dos eventos DECIDIDOS a favor "
        f"do overlay (teste de sinal, p={stats.binomtest(p125,p125+n125,0.5).pvalue:.1e}),\n"
        f"   e a mesma direcao nas QUATRO taxas. Nao usa media, nao usa modelo, nao precisa de "
        f"calibracao por nula.\n"
        f"LIMIAR 1,50: {p15}/{p15+n15} = {p15/(p15+n15):.0%} no agregado, mas DISCORDANTE entre "
        f"taxas ({'; '.join(pr15)}):\n"
        f"   moeda em 6/12/24 (p=1,00 / 0,88 / 0,61) e efeito so na taxa 48. O agregado deste "
        f"limiar NAO e' interpretavel.\n"
        "O EFEITO DEPENDE DO LIMIAR, e os dois vao juntos. Leitura coerente com o resto: o pico e' "
        "geometrico (~1,9) e nenhum\nmetodo escapa de 1,5 rapidamente; a diferenca aparece na CAUDA "
        "da recuperacao, no nivel 1,25.\n"
        "ESCOPO: N=24, tau_xy=1 s, T_off=8 s, dt=0,05 s, 8 sementes pareadas, regime NAO saturado. "
        "Painel A: janela ate o proximo\nevento (sem teto). Painel B: janela do pico, "
        "W_e=clip(t_prox-t_f, 0,6, 1,5) s, identica nos dois bracos.",
        f"THRESHOLD 1.25: {p125}/{p125+n125} = {p125/(p125+n125):.0%} of DECIDED events in favor "
        f"of the overlay (sign test, p={stats.binomtest(p125,p125+n125,0.5).pvalue:.1e}),\n"
        f"   and the same direction at all FOUR rates. Uses no mean, no model, and needs no "
        f"synthetic-null calibration.\n"
        f"THRESHOLD 1.50: {p15}/{p15+n15} = {p15/(p15+n15):.0%} in aggregate, but DISCORDANT across "
        f"rates ({'; '.join(pr15)}):\n"
        f"   a coin flip at 6/12/24 (p=1.00 / 0.88 / 0.61) and an effect only at rate 48. This "
        f"threshold's aggregate is NOT interpretable.\n"
        "THE EFFECT IS THRESHOLD-DEPENDENT, and the two belong together. Consistent with the rest: "
        "the peak is geometric (~1.9) and neither\nmethod escapes 1.5 quickly; the difference lives "
        "in the recovery TAIL, at the 1.25 level.\n"
        "SCOPE: N=24, tau_xy=1 s, T_off=8 s, dt=0.05 s, 8 paired seeds, non-saturated regime. "
        "Panel A: window to the next\nevent (no cap). Panel B: peak window, "
        "W_e=clip(t_next-t_f, 0.6, 1.5) s, identical in both arms.")
    fig.text(0.5, -0.02, foot, ha="center", va="top", fontsize=7.8, color="0.25",
             family="monospace")
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"fig_censura_pi2.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        say(f"\ngravado: {os.path.basename(out)}")
    plt.close(fig)
    with open(os.path.join(HERE, "LOG_censura.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
