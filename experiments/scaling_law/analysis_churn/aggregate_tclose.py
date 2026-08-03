#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A SEGUNDA METADE do resultado: a coordenacao atua sobre a INCLINACAO, nao sobre o piso.

DECOMPOSICAO DA BRECHA (moldura, fixada antes do dado):
    pico            -> NULO em duas analises independentes  = "nao atua sobre o piso"
    t_close / area  -> 1,28-1,42x na campanha de falha unica = "atua sobre a inclinacao"
Este script mede a SEGUNDA metade SOB CHURN, no mesmo desenho pareado por evento
ja construido e validado. So muda o desfecho: em vez do PICO de G_max, a AREA da
brecha e o TEMPO ate fechar.

Nao e' plano B nem busca por metrica que de significancia: e' o outro lado de uma
afirmacao que so fica completa com os dois. Um nulo onde a teoria preve nulo e um
efeito onde a teoria preve efeito.

REUSO INTEGRAL do que ja passou pelas sentinelas:
  * janela adaptativa W_e = clip(t_prox_evento - t_f, 0.6, 1.5) s, identica nos
    dois bracos (S11) -> exogena ao metodo;
  * pareamento 1-para-1 por (taxa, semente, instante, no);
  * erro-padrao agrupado por rodada (taxa, semente);
  * taxas identificaveis apenas (48 fica de fora, ver aggregate_gmax.py).

DESFECHOS:
  area_thr   = integral de max(0, G_max - thr) dt na janela   [adimensional*s]
  area_rad   = integral de max(0, gap_max_rad - thr*2pi/M) dt  [rad*s]
  frac_acima = fracao da janela com G_max > thr
  t_close    = 1o instante em que G_max cai abaixo de thr e LA PERMANECE ate o fim
               da janela. CENSURADO se nunca ocorre -- e sob churn a censura e' o
               caso comum, porque o intervalo entre eventos e' menor que o tempo de
               fechar. A fracao censurada e' reportada e, se alta, t_close e'
               DECLARADO nao identificavel em vez de estimado.

Uso:
    python experiments/scaling_law/analysis_churn/aggregate_tclose.py
    # env: TCLOSE_SKIP_GATE="1" (so inspecao)
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "rerun_runs")
T0_WINDOW = 20.0
W_MIN, W_MAX = 0.6, 1.5
R_ENC = 20.0
THRS = (1.25, 1.5)
USABLE = (6.0, 12.0, 24.0)      # taxa 48 nao identificavel (aggregate_gmax.py)
BASE, OVER = "baseline", "B2"
_log = []


def say(m=""):
    print(m)
    _log.append(str(m))


def die(m):
    say(""); say("=" * 78); say("ABORTADO"); say(m); say("=" * 78)
    with open(os.path.join(HERE, "LOG_tclose.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")
    sys.exit(1)


def gate():
    if os.environ.get("TCLOSE_SKIP_GATE", "").strip() in ("1", "true", "True"):
        say("AVISO: gate PULADO -- NAO vale como evidencia.")
        return
    r = subprocess.run([sys.executable, os.path.join(HERE, "check_rerun.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        say("\n".join((r.stdout or "").strip().splitlines()[-4:]))
        die(f"check_rerun.py codigo {r.returncode}.")
    say("GATE OK -- 64/64 celulas bit-reprodutiveis.")


def ols_cluster(y, X, groups, names):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    gs = np.unique(groups)
    meat = np.zeros((k, k))
    for gg in gs:
        m = groups == gg
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = gs.size
    if G < 3:
        return None
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    tc = stats.t.ppf(0.975, G - 1)
    return pd.DataFrame({"termo": names, "coef": beta, "se": se,
                         "ic95_lo": beta - tc * se, "ic95_hi": beta + tc * se,
                         "p": 2 * (1 - stats.t.cdf(np.abs(beta / se), G - 1)), "n": n})


def paired_mean(d, col, cluster=("rate_total", "seed")):
    d = d.dropna(subset=[col])
    if len(d) < 8:
        return None
    grp = d[list(cluster)].astype(str).agg("_".join, axis=1).to_numpy()
    return ols_cluster(d[col].to_numpy(float), np.ones((len(d), 1)), grp, [col])


def event_rows(tel, ev, meth, rate, seed):
    t = tel["timestamp"].to_numpy(float)
    g = tel["G_max"].to_numpy(float)
    gr = tel["gap_max_rad"].to_numpy(float)
    ac = tel["alive_count"].to_numpy(float)
    fs = ev[ev.event_type == "failure_start"]
    allev = np.sort(np.concatenate([
        fs["timestamp"].to_numpy(float),
        ev[ev.event_type == "failure_end"]["timestamp"].to_numpy(float)]))
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.05
    rows = []
    for tf, nid in sorted(zip(fs["timestamp"].to_numpy(float), fs["node_id"].to_numpy(int))):
        nxt = allev[allev > tf + 1e-9]
        gap_next = float(nxt[0] - tf) if nxt.size else float("inf")
        w = float(np.clip(gap_next, W_MIN, W_MAX))
        if tf < T0_WINDOW or tf + w > t[-1]:
            continue
        idx = np.where((t > tf) & (t <= tf + w))[0]
        if idx.size < 3:
            continue
        gg, gg_rad, mm = g[idx], gr[idx], ac[idx]
        row = {"method": meth, "rate_total": rate, "seed": seed,
               "t_fail": float(tf), "node": int(nid), "w_e": w,
               "gap_next": gap_next, "n_amostras": int(idx.size)}
        for thr in THRS:
            exc = np.maximum(0.0, gg - thr)
            row[f"area_{thr:g}"] = float(np.sum(exc) * dt)
            row[f"frac_{thr:g}"] = float(np.mean(gg > thr))
            thr_rad = thr * (2.0 * np.pi / np.maximum(mm, 1.0))
            row[f"arearad_{thr:g}"] = float(np.sum(np.maximum(0.0, gg_rad - thr_rad)) * dt)
            row[f"arcom_{thr:g}"] = row[f"arearad_{thr:g}"] * R_ENC
            # t_close: ultimo instante ACIMA do limiar; fecha se sobra janela depois
            above = gg > thr
            if not above.any():
                row[f"tclose_{thr:g}"], row[f"cens_{thr:g}"] = 0.0, False
            elif above[-1]:
                row[f"tclose_{thr:g}"], row[f"cens_{thr:g}"] = np.nan, True
            else:
                last = int(np.max(np.where(above)[0]))
                row[f"tclose_{thr:g}"] = float(t[idx[last]] - tf + dt)
                row[f"cens_{thr:g}"] = False
        rows.append(row)
    return rows


def main():
    if not os.path.isdir(RUNS):
        die(f"{RUNS} nao existe.")
    say("=" * 78)
    say("AREA DE BRECHA e t_close SOB CHURN -- a segunda metade da decomposicao")
    say("=" * 78)
    say("pico            -> NULO (duas analises independentes) = nao atua sobre o piso")
    say("t_close / area  -> 1,28-1,42x na falha unica          = atua sobre a inclinacao")
    say("Aqui: a segunda metade SOB CHURN, mesmo desenho pareado por evento.")
    gate()

    rows = []
    for d in sorted(os.listdir(RUNS)):
        tgt, evp = (os.path.join(RUNS, d, "target_telemetry.csv"),
                    os.path.join(RUNS, d, "events.csv"))
        if not (os.path.exists(tgt) and os.path.exists(evp)):
            continue
        meth = "B2" if d.startswith("dual_pulse") else "baseline"
        rate = float(d.split("_rate")[1].split("_s")[0])
        seed = int(d.split("_s")[-1])
        rows += event_rows(pd.read_csv(tgt), pd.read_csv(evp), meth, rate, seed)
    ev = pd.DataFrame(rows)
    if ev.empty:
        die("nenhum evento extraido.")

    key = ["rate_total", "seed", "t_fail", "node"]
    a = ev[ev.method == BASE].set_index(key).sort_index()
    b = ev[ev.method == OVER].set_index(key).sort_index()
    common = a.index.intersection(b.index)
    A_, B_ = a.loc[common], b.loc[common]
    pair = pd.DataFrame({"rate_total": [k[0] for k in common],
                         "seed": [k[1] for k in common],
                         "t_fail": [k[2] for k in common],
                         "w_e": A_["w_e"].to_numpy(float),
                         "gap_next": A_["gap_next"].to_numpy(float)})
    for thr in THRS:
        for stem in ("area", "frac", "arcom", "tclose"):
            c = f"{stem}_{thr:g}"
            pair[f"{c}_base"] = A_[c].to_numpy(float)
            pair[f"{c}_b2"] = B_[c].to_numpy(float)
            pair[f"d_{c}"] = pair[f"{c}_base"] - pair[f"{c}_b2"]
        pair[f"cens_{thr:g}"] = (A_[f"cens_{thr:g}"].to_numpy(bool)
                                 | B_[f"cens_{thr:g}"].to_numpy(bool))
    pair = pair[pair.rate_total.isin(USABLE)].reset_index(drop=True)
    ev.to_csv(os.path.join(HERE, "tclose_events.csv"), index=False)
    pair.to_csv(os.path.join(HERE, "tclose_paired.csv"), index=False)
    say(f"\npares (taxas {[f'{r:g}' for r in USABLE]}): {len(pair)}")

    # ---- t_close e' identificavel sob churn? ------------------------------
    say("")
    say("=" * 78)
    say("t_close E' IDENTIFICAVEL SOB CHURN?")
    say("=" * 78)
    say(f"{'limiar':>8}{'pares':>8}{'censurados':>12}{'frac':>9}  veredito")
    ident = {}
    for thr in THRS:
        c = pair[f"cens_{thr:g}"]
        fr = float(c.mean())
        ok = fr < 0.30
        ident[thr] = ok
        say(f"{thr:>8g}{len(pair):>8}{int(c.sum()):>12}{fr:>9.3f}  "
            f"{'estimavel' if ok else 'NAO IDENTIFICAVEL (censura alta)'}")
    say("")
    say("Censura = a brecha NAO fechou dentro da janela adaptativa. Sob churn o intervalo")
    say("entre eventos e' MENOR que o tempo de fechar, entao t_close completo nao cabe na")
    say("janela -- isso e' propriedade do REGIME, nao defeito da medida. A campanha de")
    say("falha unica mede t_close porque la ha um evento so.")
    say("A AREA dentro da janela nao tem esse problema: e' comparavel entre bracos porque")
    say("W_e e' identico nos dois (S11), e nao exige que a brecha feche.")

    # ---- desfecho principal: area -----------------------------------------
    say("")
    say("=" * 78)
    say("AREA DE BRECHA na janela -- pareada por evento")
    say("=" * 78)
    say("As quatro estatisticas juntas, sempre (regra fixada apos o episodio dos 18 cm):")
    say("media, mediana, fracao de sinal e media-sem-decil-superior.")
    for thr in THRS:
        say("")
        say(f"-- limiar G_max > {thr:g} " + "-" * 50)
        for stem, unid in (("area", "adim*s"), ("arcom", "m*s"), ("frac", "adim")):
            col = f"d_{stem}_{thr:g}"
            d = pair.dropna(subset=[col])
            if len(d) < 8:
                continue
            x = d[col].to_numpy(float)
            t = paired_mean(d, col)
            if t is None:
                continue
            m = t.iloc[0]
            p90 = np.quantile(x, 0.9)
            razao = (float(np.median(d[f"{stem}_{thr:g}_base"]))
                     / max(float(np.median(d[f"{stem}_{thr:g}_b2"])), 1e-12))
            say(f"  {stem:<6} [{unid:>6}]  media {m['coef']:+.5f} "
                f"IC95 [{m['ic95_lo']:+.5f}, {m['ic95_hi']:+.5f}] p={m['p']:.4f}")
            say(f"  {'':<6}           mediana {np.median(x):+.5f}  "
                f"frac>0 {float((x > 0).mean()):.3f}  "
                f"media s/ decil sup {float(x[x < p90].mean()):+.5f}")
            say(f"  {'':<6}           medianas: base {np.median(d[f'{stem}_{thr:g}_base']):.5f} "
                f"vs B2 {np.median(d[f'{stem}_{thr:g}_b2']):.5f}  -> razao {razao:.3f}")
        # por taxa
        col = f"d_area_{thr:g}"
        say(f"  por taxa (area):")
        for r in USABLE:
            s = pair[pair.rate_total == r].dropna(subset=[col])
            t = paired_mean(s, col, cluster=("seed",))
            if t is None:
                say(f"     {r:>3g}/min  n={len(s):<5} nao rodado")
                continue
            m = t.iloc[0]
            x = s[col].to_numpy(float)
            say(f"     {r:>3g}/min  n={len(s):<5} media {m['coef']:+.5f} "
                f"IC95 [{m['ic95_lo']:+.5f}, {m['ic95_hi']:+.5f}] p={m['p']:.4f}  "
                f"mediana {np.median(x):+.5f}  frac>0 {float((x > 0).mean()):.3f}")

    # ---- t_close onde identificavel ---------------------------------------
    for thr in THRS:
        if not ident[thr]:
            continue
        col = f"d_tclose_{thr:g}"
        d = pair[~pair[f"cens_{thr:g}"]].dropna(subset=[col])
        t = paired_mean(d, col)
        if t is None:
            continue
        m = t.iloc[0]
        x = d[col].to_numpy(float)
        say("")
        say(f"-- t_close (limiar {thr:g}), so eventos NAO censurados, n={len(d)} --")
        say(f"   media {m['coef']:+.5f} s IC95 [{m['ic95_lo']:+.5f}, {m['ic95_hi']:+.5f}] "
            f"p={m['p']:.4f}   mediana {np.median(x):+.5f} s   frac>0 {float((x>0).mean()):.3f}")
        say(f"   medianas: base {np.median(d[f'tclose_{thr:g}_base']):.4f} s vs "
            f"B2 {np.median(d[f'tclose_{thr:g}_b2']):.4f} s")
        say("   AVISO: condicionar em 'nao censurado' e' condicionar no DESFECHO. Este")
        say("   numero e' descritivo; o desfecho principal e' a AREA.")

    with open(os.path.join(HERE, "LOG_tclose.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")
    say("\nEscrito: tclose_events.csv, tclose_paired.csv, LOG_tclose.txt")


if __name__ == "__main__":
    main()
