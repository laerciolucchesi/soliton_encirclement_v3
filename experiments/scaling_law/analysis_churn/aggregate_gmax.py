#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G_max POR EVENTO: efeitos fixos de evento + JANELA ADAPTATIVA.

=======================================================================
PRE-REGISTRO (antes do dado; nao editar depois)
=======================================================================
Pre-registro anterior ("G_max se comporta como egap_max") RETIRADO: egap_max e' o
maximo no tempo do RMS ESPACIAL, e um RMS sobre M vaos e' estruturalmente
insensivel a UM vao grande (entra dividido por M); G_max E' esse vao.

  H1 (piso puro):         G_max nao distingue os metodos.
  H2 (piso condicionado): o metodo nao muda o pico de UM evento, mas melhora o
                          ESTADO que o PROXIMO encontra. Refina o piso para "o
                          pico e' invariante do evento DADO o estado pre-evento".

=======================================================================
DESENHO
=======================================================================
S11 (verificada por dado): os dois metodos veem fluxos de falha IDENTICOS. Logo
cada evento existe duas vezes, diferindo so no metodo e no estado que aquele
metodo produziu. Para cada evento e:

    d_pico = gmax_peak_base(e) - gmax_peak_B2(e)     (idem para os pre)
    (TOTAL)  d_pico = alpha_tot + erro
    (DIRETO) d_pico = alpha + b1*d_gmax_pre + b2*d_egap_pre + erro

alpha e' o efeito DIRETO purgado de tudo que pertence ao evento (no, instante,
fase da rodada). Erro-padrao SEMPRE agrupado por rodada (taxa, semente): eventos
proximos compartilham historia -- o estado pre de um e' o pos do anterior.

JANELA ADAPTATIVA (elimina contaminacao por construcao, sem descartar evento):
    W_e = clip(t_proximo_evento - t_f, W_MIN=0.6 s, W_MAX=1.5 s)
Nao se apoia na hipotese de que a contaminacao e' modo comum entre bracos -- ela
nao e': os eventos contaminantes sao os mesmos, mas a MAGNITUDE do pico
contaminante tambem depende do metodo. Como o fluxo de eventos e' identico nos
dois bracos (S11), W_e tambem e' identico -> exogeno ao metodo.

VEREDITO:
  alpha ~ 0 e b1 > 0             -> H2 no nivel do evento individual
  alpha != 0                     -> enunciado do piso geometrico ERRADO
  alpha ~ 0, b1 ~ 0, alpha_tot~0 -> H1 pura
Rotulos H1/H2 pressupoem aditividade: se houver interacao, nao se aplicam sem
qualificacao.

Uso:
    python experiments/scaling_law/analysis_churn/aggregate_gmax.py
    # env: GMAX_SKIP_GATE="1" (so inspecao de codigo; invalida a evidencia)
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
W_MIN, W_MAX = 0.6, 1.5          # janela adaptativa
W_FIXED_DIAG = (1.5, 3.0, 5.0)   # so p/ o diagnostico D.1 (assinatura da contaminacao)
W_PRE = 0.5
R_ENC = 20.0
BASE, OVER = "baseline", "B2"

sys.path.insert(0, EXP_DIR)
import probe_gmax_floor as pg   # noqa: E402

_log = []


def say(m=""):
    print(m)
    _log.append(str(m))


def flush():
    with open(os.path.join(HERE, "LOG_gmax.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log) + "\n")


def die(m):
    say(""); say("=" * 78); say("ABORTADO"); say(m); say("=" * 78)
    flush(); sys.exit(1)


def gate():
    if os.environ.get("GMAX_SKIP_GATE", "").strip() in ("1", "true", "True"):
        say("AVISO: gate de reprodutibilidade PULADO -- NAO vale como evidencia.")
        return
    r = subprocess.run([sys.executable, os.path.join(HERE, "check_rerun.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
    if r.returncode != 0:
        say(tail)
        die(f"check_rerun.py codigo {r.returncode}: "
            + ("re-run incompleto." if r.returncode == 2
               else "REPRODUTIBILIDADE QUEBRADA -- bissectar o git antes de qualquer figura."))
    say("GATE OK -- reprodutibilidade confirmada nas 64 celulas:")
    say(tail)


# ---------------------------------------------------------------- leitura
def load_cells():
    cells = {}
    for d in sorted(os.listdir(RUNS)):
        tgt, evp = (os.path.join(RUNS, d, "target_telemetry.csv"),
                    os.path.join(RUNS, d, "events.csv"))
        if not (os.path.exists(tgt) and os.path.exists(evp)):
            continue
        meth, rate, seed = pg.parse_dir(d)
        tel = pd.read_csv(tgt)
        for c in ("G_max", "E_gap", "alive_count", "gap_max_rad"):
            if c not in tel.columns:
                die(f"S12: {d} sem '{c}' -- schema antigo; vao ABSOLUTO nao recuperavel.")
        cells[(meth, rate, seed)] = (tel, pd.read_csv(evp))
    if not cells:
        die(f"S12: nenhuma celula com telemetria+events em {RUNS}.")
    return cells


def s11(cells):
    keys = sorted({(r, s) for (_m, r, s) in cells})
    bad = 0
    for (r, s) in keys:
        a, b = cells.get((BASE, r, s)), cells.get((OVER, r, s))
        if a is None or b is None:
            continue
        def k(ev):
            f = ev[ev.event_type == "failure_start"]
            return tuple(sorted(zip(f.node_id.astype(int), np.round(f.timestamp, 6))))
        if k(a[1]) != k(b[1]):
            bad += 1
            say(f"  S11 DIVERGE rate={r:g} seed={s}")
    if bad:
        die(f"S11: {bad} par(es) com fluxos de falha DIFERENTES -- desenho invalido.")
    say(f"S11 OK  {len(keys)} pares (taxa, semente) com fluxo de falhas IDENTICO")
    return keys


def s15_alive_identical(cells, keys):
    """(item 3) Por S11 os fluxos de falha sao identicos; T_off=8,0 s e' deterministico
    (logo as recuperacoes tambem), e a latencia de deteccao e' deterministica. Entao
    alive_count(t) DEVE ser identico nos dois bracos, amostra a amostra.

    Se for 100%: o padrao de quem esta fora e' o mesmo nos dois bracos em cada
    instante, logo SAI na diferenca pareada -- m_unstable vira descritivo, nao ameaca.
    Se nao for: contradiz S11 e e' outro achado, a investigar antes de concluir."""
    say("")
    say("-- S15: alive_count(t) e' identico entre os bracos? (item 3) --")
    rows = []
    for (r, s) in keys:
        a, b = cells.get((BASE, r, s)), cells.get((OVER, r, s))
        if a is None or b is None:
            continue
        ta, tb = a[0], b[0]
        m = ta[["timestamp", "alive_count"]].merge(
            tb[["timestamp", "alive_count"]], on="timestamp", suffixes=("_a", "_b"))
        if not len(m):
            continue
        eq = float((m["alive_count_a"] == m["alive_count_b"]).mean())
        rows.append({"rate_total": r, "seed": s, "n": len(m), "frac_igual": eq,
                     "n_dif": int((m["alive_count_a"] != m["alive_count_b"]).sum())})
    d = pd.DataFrame(rows)
    if d.empty:
        say("   sem pares comparaveis.")
        return None
    say(f"{'taxa':>6}{'celulas':>9}{'frac_igual_min':>16}{'frac_igual_med':>16}{'amostras_dif':>14}")
    for r in sorted(d.rate_total.unique()):
        s_ = d[d.rate_total == r]
        say(f"{r:>6g}{len(s_):>9}{s_.frac_igual.min():>16.6f}"
            f"{s_.frac_igual.median():>16.6f}{int(s_.n_dif.sum()):>14}")
    allsame = bool((d.frac_igual == 1.0).all())
    say("")
    if allsame:
        say("   => 100% IDENTICO em todas as celulas. O padrao de ausencia e' o MESMO nos")
        say("      dois bracos a cada instante, logo sai na diferenca pareada. A flag")
        say("      m_unstable (D.2) e' DESCRITIVA, nao ameaca ao teste principal.")
    else:
        say("   => NAO e' 100%. Isto CONTRADIZ S11 (mesmo fluxo de falhas + T_off")
        say("      deterministico deveriam dar alive_count identico). Achado a investigar")
        say("      ANTES de qualquer conclusao sobre G_max.")
    return allsame


# ------------------------------------------------- extracao (janela adaptativa)
def event_rows(tel, ev, meth, rate, seed, w_fixed=None):
    t = tel["timestamp"].to_numpy(float)
    g = tel["G_max"].to_numpy(float)
    e = tel["E_gap"].to_numpy(float)
    ac = tel["alive_count"].to_numpy(float)
    gr = tel["gap_max_rad"].to_numpy(float)
    fs = ev[ev.event_type == "failure_start"]
    allev = np.sort(np.concatenate([
        fs["timestamp"].to_numpy(float),
        ev[ev.event_type == "failure_end"]["timestamp"].to_numpy(float)]))
    rows = []
    for tf, nid in sorted(zip(fs["timestamp"].to_numpy(float), fs["node_id"].to_numpy(int))):
        nxt = allev[allev > tf + 1e-9]
        gap_next = float(nxt[0] - tf) if nxt.size else float("inf")
        # gap_prev = tempo desde o evento anterior. EXOGENO: vem so do fluxo de
        # falhas, identico nos dois bracos (S11). Cresce com "o anel teve tempo de
        # se re-uniformizar" -- e' o seletor limpo para estratificar, no lugar de
        # gmax_pre (que e' POS-TRATAMENTO: e' o estado que o metodo produziu).
        prv = allev[allev < tf - 1e-9]
        gap_prev = float(tf - prv[-1]) if prv.size else float("inf")
        if w_fixed is None:
            w = float(np.clip(gap_next, W_MIN, W_MAX))
        else:
            w = float(w_fixed)
        if tf < T0_WINDOW or tf + w > t[-1]:
            continue
        pre = (t < tf) & (t >= tf - W_PRE)
        post = (t > tf) & (t <= tf + w)
        if not pre.any() or not post.any():
            continue
        i_last = int(np.where(t < tf)[0][-1])
        idx = np.where(post)[0]
        i_pk = int(idx[np.argmax(g[idx])])
        m_win = ac[idx]
        # PRIMEIRA amostra pos-deteccao (M ja caiu). O MAXIMO sobre a janela e' uma
        # estatistica de valor extremo sobre ~12-30 amostras: mesmo ruido puro em
        # torno do valor verdadeiro enviesa o max para CIMA. A primeira amostra
        # pos-deteccao nao tem esse vies, e testa o teorema de forma limpa.
        drop = idx[ac[idx] < ac[i_last]]
        i_fst = int(drop[0]) if drop.size else -1
        rows.append({
            "method": meth, "rate_total": rate, "seed": seed,
            "t_fail": float(tf), "node": int(nid),
            "w_e": w, "gap_next": gap_next, "gap_prev": gap_prev,
            "lb_max": float(g[i_last] * (ac[i_last] - 1.0) / ac[i_last])
                      if ac[i_last] >= 2 else np.nan,
            "w_clip_floor": bool(gap_next < W_MIN),
            "w_full": bool(gap_next >= W_MAX),
            "gmax_pre": float(g[i_last]), "egap_pre": float(e[i_last]),
            "alive_pre": float(ac[i_last]),
            "gmax_peak": float(g[i_pk]), "t_peak_rel": float(t[i_pk] - tf),
            "gmax_first": float(g[i_fst]) if i_fst >= 0 else np.nan,
            "alive_first": float(ac[i_fst]) if i_fst >= 0 else np.nan,
            "gap_rad_first": float(gr[i_fst]) if i_fst >= 0 else np.nan,
            "floor_first": (float(2.0 * (ac[i_fst] - 1.0) / ac[i_fst])
                            if i_fst >= 0 and ac[i_fst] >= 2 else np.nan),
            "gap_rad_peak": float(gr[i_pk]), "alive_peak": float(ac[i_pk]),
            "arc_peak_m": float(gr[i_pk] * R_ENC),
            "geom_floor": (float(2.0 * (ac[i_pk] - 1.0) / ac[i_pk])
                           if ac[i_pk] >= 2 else np.nan),
            "n_concurrent": int(np.sum((allev > tf - 1.0) & (allev < tf + w) & (allev != tf))),
            "peak_late": bool((t[i_pk] - tf) > 0.9 * w),
            "m_nvals": int(np.unique(m_win).size),
            "m_shift": float(ac[i_pk] - ac[i_last]),
        })
    for r in rows:
        r["m_unstable"] = bool(r["m_nvals"] > 2 or r["m_shift"] != -1.0)
        r["isolated"] = bool(r["n_concurrent"] == 0)
        # TERCEIRA forma funcional (nem aditiva nem multiplicativa): o pico e'
        # CENSURADO pelo piso geometrico 2(M-1)/M. Um piso gera b1<0 com b3>0 em
        # QUALQUER escala -- log nao remove kink, so remove escala multiplicativa.
        f = r["geom_floor"]
        rel = r["gmax_peak"] / f if (f and np.isfinite(f) and f > 0) else np.nan
        r["rel_floor"] = float(rel)
        r["at_floor"] = bool(np.isfinite(rel) and 0.98 < rel <= 1.02)
        r["above_floor"] = bool(np.isfinite(rel) and rel > 1.02)
        r["below_floor"] = bool(np.isfinite(rel) and rel <= 0.98)
    return rows


def build(cells, w_fixed=None):
    rows = []
    for (meth, rate, seed), (tel, ev) in cells.items():
        rows += event_rows(tel, ev, meth, rate, seed, w_fixed)
    ev = pd.DataFrame(rows)
    if ev.empty:
        die("S13: nenhum evento na janela de regime.")
    key = ["rate_total", "seed", "t_fail", "node"]
    a = ev[ev.method == BASE].set_index(key).sort_index()
    b = ev[ev.method == OVER].set_index(key).sort_index()
    common = a.index.intersection(b.index)
    if len(common) == 0:
        die("S14: nenhum evento pareado.")
    A_, B_ = a.loc[common], b.loc[common]
    eps = 1e-12
    pair = pd.DataFrame({
        "rate_total": [k[0] for k in common], "seed": [k[1] for k in common],
        "t_fail": [k[2] for k in common], "node": [k[3] for k in common],
        "w_e": A_["w_e"].to_numpy(float), "gap_next": A_["gap_next"].to_numpy(float),
        "gap_prev": A_["gap_prev"].to_numpy(float),
        "gap_prev_b": B_["gap_prev"].to_numpy(float),
        "gap_rad_base": A_["gap_rad_peak"].to_numpy(float),
        "gap_rad_b2": B_["gap_rad_peak"].to_numpy(float),
        "alive_pre_base": A_["alive_pre"].to_numpy(float),
        "w_clip_floor": A_["w_clip_floor"].to_numpy(bool),
        "w_full": A_["w_full"].to_numpy(bool),
        "d_pico": A_["gmax_peak"].to_numpy(float) - B_["gmax_peak"].to_numpy(float),
        "d_gmax_pre": A_["gmax_pre"].to_numpy(float) - B_["gmax_pre"].to_numpy(float),
        "d_egap_pre": A_["egap_pre"].to_numpy(float) - B_["egap_pre"].to_numpy(float),
        "d_arco_m": A_["arc_peak_m"].to_numpy(float) - B_["arc_peak_m"].to_numpy(float),
        "pico_base": A_["gmax_peak"].to_numpy(float),
        "pico_b2": B_["gmax_peak"].to_numpy(float),
        "pre_base": A_["gmax_pre"].to_numpy(float),
        "pre_b2": B_["gmax_pre"].to_numpy(float),
        "pre_bar": 0.5 * (A_["gmax_pre"].to_numpy(float) + B_["gmax_pre"].to_numpy(float)),
        "arco_base": A_["arc_peak_m"].to_numpy(float),
        "arco_b2": B_["arc_peak_m"].to_numpy(float),
        "isolated": A_["isolated"].to_numpy(bool) & B_["isolated"].to_numpy(bool),
        "m_unstable": A_["m_unstable"].to_numpy(bool) | B_["m_unstable"].to_numpy(bool),
        "peak_late": A_["peak_late"].to_numpy(bool) | B_["peak_late"].to_numpy(bool),
        "floor_base": A_["geom_floor"].to_numpy(float),
        "floor_b2": B_["geom_floor"].to_numpy(float),
        "rel_base": A_["rel_floor"].to_numpy(float),
        "rel_b2": B_["rel_floor"].to_numpy(float),
        # nao-censurado = pico acima do piso nos DOIS bracos
        "both_above": A_["above_floor"].to_numpy(bool) & B_["above_floor"].to_numpy(bool),
        "any_at_floor": A_["at_floor"].to_numpy(bool) | B_["at_floor"].to_numpy(bool),
    })
    # (item 2) versao MULTIPLICATIVA: diferencas em log. A relacao estrutural prevista
    # e' pico ~ (g_pred+g_succ)/ideal, multiplicativa; ajustar aditivo a ela gera
    # residuo com forma de interacao (b3 > 0 = "inclinacao cresce com o nivel"),
    # que e' exatamente a assinatura de forma funcional errada.
    ok = ((pair.pico_base > eps) & (pair.pico_b2 > eps)
          & (pair.pre_base > eps) & (pair.pre_b2 > eps)
          & (A_["egap_pre"].to_numpy(float) > eps) & (B_["egap_pre"].to_numpy(float) > eps))
    pair["log_ok"] = ok.to_numpy(bool) if hasattr(ok, "to_numpy") else np.asarray(ok)
    pair["dl_pico"] = np.log(pair.pico_base.where(pair.log_ok)) - np.log(pair.pico_b2.where(pair.log_ok))
    pair["dl_pre"] = np.log(pair.pre_base.where(pair.log_ok)) - np.log(pair.pre_b2.where(pair.log_ok))
    pair["dl_egap_pre"] = (np.log(pd.Series(A_["egap_pre"].to_numpy(float)).where(pair.log_ok))
                           - np.log(pd.Series(B_["egap_pre"].to_numpy(float)).where(pair.log_ok)))
    pair["l_pre_bar"] = np.log(pair.pre_bar.where(pair.log_ok))
    # desfecho NORMALIZADO PELO PISO: remove a dependencia em M do proprio piso.
    okf = pair.log_ok & (pair.floor_base > eps) & (pair.floor_b2 > eps)
    pair["dl_rel"] = (np.log((pair.pico_base / pair.floor_base).where(okf))
                      - np.log((pair.pico_b2 / pair.floor_b2).where(okf)))
    return ev, pair.sort_values(["rate_total", "seed", "t_fail"]).reset_index(drop=True)


# ------------------------------------------------------------- estimacao
def ols_cluster(y, X, groups, names):
    """OLS + erro-padrao CR1 agrupado. (item 4) O cluster e' SEMPRE (taxa, semente):
    eventos proximos na mesma rodada compartilham historia -- o estado pre de um e' o
    pos do anterior. Sem cluster o IC de alpha sai otimista, e alpha decide o veredito."""
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
        return None, np.nan, G, n
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    tc = stats.t.ppf(0.975, G - 1)
    ss_res, ss_tot = float(u @ u), float(((y - y.mean()) ** 2).sum())
    return (pd.DataFrame({"termo": names, "coef": beta, "se_cl": se, "t": beta / se,
                          "ic95_lo": beta - tc * se, "ic95_hi": beta + tc * se,
                          "p": 2 * (1 - stats.t.cdf(np.abs(beta / se), G - 1))}),
            (1 - ss_res / ss_tot if ss_tot > 0 else np.nan), G, n)


def show(tab, r2, G, n, title):
    say("")
    if tab is None:
        say(f"-- {title}: clusters={G} < 3 -- NAO rodado (ausencia declarada).")
        return
    say(f"-- {title}   n={n}  clusters={G}  R2={r2:.4f}")
    say(f"{'termo':<26}{'coef':>11}{'se_cl':>10}{'t':>8}{'IC95 lo':>11}{'IC95 hi':>11}{'p':>10}")
    for _, r in tab.iterrows():
        say(f"{r['termo']:<26}{r['coef']:>11.5f}{r['se_cl']:>10.5f}{r['t']:>8.2f}"
            f"{r['ic95_lo']:>11.5f}{r['ic95_hi']:>11.5f}{r['p']:>10.5f}")


def fit(d, ycol, xcols, names, title, cluster_cols=("rate_total", "seed"), quiet=False):
    d = d.dropna(subset=[ycol] + xcols)
    if len(d) < 8:
        if not quiet:
            say(f"-- {title}: so {len(d)} eventos -- NAO rodado (ausencia declarada).")
        return None
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in xcols])
    grp = d[list(cluster_cols)].astype(str).agg("_".join, axis=1).to_numpy()
    tab, r2, G, n = ols_cluster(d[ycol].to_numpy(float), X, grp, ["alpha"] + names)
    if not quiet:
        show(tab, r2, G, n, title)
    return tab


def zero_in(m):
    return bool(m["ic95_lo"] <= 0.0 <= m["ic95_hi"])


def main():
    if not os.path.isdir(RUNS):
        die(f"{RUNS} nao existe -- rode rerun_c3_gmax.py primeiro.")
    say("=" * 78)
    say("G_max POR EVENTO -- efeitos fixos de evento, janela ADAPTATIVA")
    say("=" * 78)
    gate()
    cells = load_cells()
    say("")
    keys = s11(cells)
    alive_same = s15_alive_identical(cells, keys)

    # ---------- D.1: assinatura da contaminacao (janelas FIXAS) -------------
    say("")
    say("=" * 78)
    say("D.1 -- por que janela fixa nao serve: truncamento vs CONTAMINACAO")
    say("=" * 78)
    say(f"{'W [s]':>7}{'pares':>8}{'tardio_todos':>14}{'tardio_ISO':>12}{'n_iso':>7}"
        f"{'lag_p99':>9}{'p99/W':>8}")
    for w in W_FIXED_DIAG:
        e2, p2 = build(cells, w_fixed=w)
        iso = p2[p2.isolated]
        q99 = float(e2["t_peak_rel"].quantile(0.99))
        say(f"{w:>7g}{len(p2):>8}{p2['peak_late'].mean():>14.4f}"
            f"{(iso['peak_late'].mean() if len(iso) else np.nan):>12.4f}{len(iso):>7}"
            f"{q99:>9.3f}{q99/w:>8.3f}")
    say("   Duas assinaturas, e as duas apontam CONTAMINACAO, nao truncamento:")
    say("   (a) a fracao tardia entre TODOS NAO cai quando W cresce -- janela maior")
    say("       engole mais eventos concorrentes e o maximo vira o pico de outro evento;")
    say("   (b) o lag p99 ACOMPANHA W (razao p99/W ~ 0,97-0,99 nas tres janelas) em vez")
    say("       de ESTACIONAR num valor fisico. Se fosse truncamento de um transiente")
    say("       unico, p99 estabilizaria assim que W passasse a duracao do transiente.")
    say("   Entre os ISOLADOS a fracao tardia e' ~0: a janela nunca foi curta.")

    # ---------- janela adaptativa ------------------------------------------
    ev, pair = build(cells)
    ev.to_csv(os.path.join(HERE, "gmax_events.csv"), index=False)
    pair.to_csv(os.path.join(HERE, "gmax_events_paired.csv"), index=False)
    say("")
    say("=" * 78)
    say(f"JANELA ADAPTATIVA  W_e = clip(t_prox_evento - t_f, {W_MIN:g}, {W_MAX:g}) s")
    say("=" * 78)
    say("W_e e' identico nos dois bracos (o fluxo de eventos e' o mesmo, S11) -> exogeno.")
    say(f"eventos por braco: {len(ev)}  |  pares: {len(pair)}")
    say("")
    say(f"{'taxa':>6}{'pares':>8}{'W_e med':>10}{'W_e p25':>10}{'W_e p75':>10}"
        f"{'frac_piso':>11}{'frac_cheia':>12}  identificavel?")
    nao_ident = []
    for r in sorted(pair.rate_total.unique()):
        s = pair[pair.rate_total == r]
        med = float(s.w_e.median())
        fl = float(s.w_clip_floor.mean())
        ok = not (np.isclose(med, W_MIN) or fl >= 0.5)
        if not ok:
            nao_ident.append(r)
        say(f"{r:>6g}{len(s):>8}{med:>10.3f}{s.w_e.quantile(.25):>10.3f}"
            f"{s.w_e.quantile(.75):>10.3f}{fl:>11.3f}{float(s.w_full.mean()):>12.3f}"
            f"  {'sim' if ok else 'NAO IDENTIFICAVEL'}")
    if nao_ident:
        say("")
        say(f"   SENTINELA: taxa(s) {[f'{r:g}' for r in nao_ident]} com mediana de W_e no")
        say(f"   piso ({W_MIN:g} s) ou >=50% dos eventos truncados pelo piso. Nessas taxas o")
        say("   pico do evento NAO e' separavel do proximo: DECLARADAS NAO IDENTIFICAVEIS")
        say("   para o teste por evento, em vez de produzir um numero.")
    usable = [r for r in sorted(pair.rate_total.unique()) if r not in nao_ident]
    pu = pair[pair.rate_total.isin(usable)]

    # ---------- item 5: POR TAXA primeiro -----------------------------------
    say("")
    say("=" * 78)
    say("TESTE PRINCIPAL -- alpha POR TAXA (agregado so depois, e so se concordarem)")
    say("=" * 78)
    say("d_pico = gmax_peak(baseline) - gmax_peak(B2), mesmo evento nos dois bracos.")
    say("Cluster = semente dentro de cada taxa; = (taxa, semente) no agregado.")
    say("")
    say(f"{'taxa':>6}{'pares':>8}{'alpha_tot':>12}{'alpha (DIRETO)':>16}{'IC95 lo':>11}"
        f"{'IC95 hi':>11}{'p':>10}{'b1':>10}{'IC95 b1':>22}")
    per_rate = {}
    for r in usable:
        s = pu[pu.rate_total == r]
        t_tot = fit(s, "d_pico", [], [], "", cluster_cols=("seed",), quiet=True)
        t_dir = fit(s, "d_pico", ["d_gmax_pre", "d_egap_pre"], ["b1", "b2"], "",
                    cluster_cols=("seed",), quiet=True)
        if t_dir is None or t_tot is None:
            say(f"{r:>6g}{len(s):>8}   nao rodado (clusters/n insuficientes)")
            continue
        a_, b1_ = t_dir.iloc[0], t_dir.iloc[1]
        per_rate[r] = {"alpha": a_, "b1": b1_, "tot": t_tot.iloc[0], "n": len(s)}
        say(f"{r:>6g}{len(s):>8}{t_tot.iloc[0]['coef']:>12.5f}{a_['coef']:>16.5f}"
            f"{a_['ic95_lo']:>11.5f}{a_['ic95_hi']:>11.5f}{a_['p']:>10.5f}"
            f"{b1_['coef']:>10.4f}  [{b1_['ic95_lo']:>8.4f},{b1_['ic95_hi']:>8.4f}]")

    if not per_rate:
        die("S16: nenhuma taxa identificavel produziu estimativa. Nada a concluir.")

    sig = {r: (not zero_in(v["alpha"])) for r, v in per_rate.items()}
    signs = {r: np.sign(v["alpha"]["coef"]) for r, v in per_rate.items()}
    concord = (len(set(sig.values())) == 1 and
               (not any(sig.values()) or len(set(signs[r] for r in sig if sig[r])) == 1))
    say("")
    if concord:
        say("   As taxas CONCORDAM (mesmo veredito de IC, mesmo sinal quando nao-nulo).")
        say("   O agregado entre taxas e' legitimo e vem abaixo.")
    else:
        say("   AS TAXAS DISCORDAM. Por regra fixada antes do dado, a DISCORDANCIA E' O")
        say("   RESULTADO -- o agregado entre taxas nao e' interpretavel e e' impresso")
        say("   abaixo apenas para registro.")

    # ---------- agregado ----------------------------------------------------
    t_tot = fit(pu, "d_pico", [], [], "AGREGADO TOTAL: d_pico ~ 1")
    t_dir = fit(pu, "d_pico", ["d_gmax_pre", "d_egap_pre"], ["b1", "b2"],
                "AGREGADO DIRETO: d_pico ~ 1 + d_gmax_pre + d_egap_pre")
    m_tot, m_dir, b1 = t_tot.iloc[0], t_dir.iloc[0], t_dir.iloc[1]

    # ---------- item 2: interacao vs forma funcional ------------------------
    say("")
    say("=" * 78)
    say("ITEM 2 -- b3 e' INTERACAO ou FORMA FUNCIONAL ERRADA?")
    say("=" * 78)
    say("A relacao estrutural prevista e' MULTIPLICATIVA: pico ~ (g_pred+g_succ)/ideal.")
    say("Ajustar um modelo ADITIVO a uma relacao multiplicativa produz residuo com forma")
    say("de interacao, e b3 > 0 ('a inclinacao cresce com o nivel') e' a assinatura disso.")
    say("Se b3 SUMIR em log, nao havia interacao -- havia forma funcional errada.")
    t_add = fit(pu, "d_pico", ["d_gmax_pre", "d_egap_pre", "int_add"], ["b1", "b2", "b3"],
                "ADITIVO + interacao (b3 = d_gmax_pre x pre_bar)"
                ) if "int_add" in pu.columns else None
    # MODERADOR CENTRADO. Num modelo com interacao, o "efeito principal" b1 e' a
    # inclinacao no ponto em que o moderador vale ZERO. Aqui pre_bar >= 1 sempre, e
    # log(pre_bar) ~ 0.3: a inclinacao em pre_bar=0 e' EXTRAPOLACAO fora do dado, e
    # nao e' interpretavel. Centrando o moderador na media, b1 passa a ser a
    # inclinacao no centro da amostra -- que e' o numero comparavel com -2/M.
    pb_c = pu["pre_bar"] - pu["pre_bar"].mean()
    lb_c = pu["l_pre_bar"] - pu["l_pre_bar"].mean()
    pu = pu.assign(int_add=pu["d_gmax_pre"] * pu["pre_bar"],
                   int_log=pu["dl_pre"] * pu["l_pre_bar"],
                   int_add_c=pu["d_gmax_pre"] * pb_c,
                   int_log_c=pu["dl_pre"] * lb_c)
    t_add = fit(pu, "d_pico", ["d_gmax_pre", "d_egap_pre", "int_add"], ["b1", "b2", "b3"],
                "ADITIVO  + interacao NAO centrada (b1 = inclinacao em pre_bar=0)")
    t_addc = fit(pu, "d_pico", ["d_gmax_pre", "d_egap_pre", "int_add_c"], ["b1", "b2", "b3"],
                 "ADITIVO  + interacao CENTRADA    (b1 = inclinacao na media)")
    t_log = fit(pu, "dl_pico", ["dl_pre", "dl_egap_pre", "int_log"], ["b1", "b2", "b3"],
                "EM LOG   + interacao NAO centrada")
    t_logc = fit(pu, "dl_pico", ["dl_pre", "dl_egap_pre", "int_log_c"], ["b1", "b2", "b3"],
                 "EM LOG   + interacao CENTRADA")
    say("")
    say("   ATENCAO: b1 muda de sinal entre a versao nao centrada e a centrada se o")
    say("   moderador nunca chega perto de zero. A versao CENTRADA e' a interpretavel;")
    say("   b3 e' identico nas duas (centrar nao altera a interacao).")
    # ---- S16: o modelo MAX, testado de graca ------------------------------
    # G_max e' o maximo sobre TODOS os vaos, nao o vao recem-formado:
    #   gmax_peak = max( gmax_pre*(M-1)/M , (g_pred+g_succ)*(M-1)/(2*pi) )
    # O 1o termo e' o vao pre-existente apenas RENORMALIZADO de M para M-1: ele
    # continua existindo, entao a desigualdade tem de valer sempre.
    say("")
    say("-- S16: gmax_peak >= gmax_pre*(M-1)/M (modelo MAX; testa a EXTRACAO) --")
    evm = ev.dropna(subset=["lb_max"])
    viol = evm["gmax_peak"].to_numpy(float) < evm["lb_max"].to_numpy(float)
    say(f"   satisfazem: {int((~viol).sum())}/{len(evm)} ({(~viol).mean():.4%})   "
        f"violam: {int(viol.sum())} ({viol.mean():.4%})")
    if viol.mean() > 0.01:
        die(f"S16: {viol.mean():.2%} dos eventos violam a desigualdade do max. Isso NAO e' "
            f"resultado -- e' EXTRACAO ERRADA (janela perdeu o pico ou a amostra 'pre' esta "
            f"mal cronometrada). Corrigir antes de qualquer conclusao.")
    say("   => desigualdade satisfeita: a extracao esta correta (a janela nao perdeu o")
    say("      pico e a amostra 'pre' esta bem cronometrada).")
    dom_pre = np.isclose(evm["gmax_peak"], evm["lb_max"], rtol=0.02)
    say(f"   ramo ativo do max: vao PRE-existente em {dom_pre.mean():.2%} dos eventos; "
        f"FUSAO em {(~dom_pre).mean():.2%}")
    if (~dom_pre).mean() > 0.95:
        say("   => o max e' DEGENERADO na pratica: um unico ramo ativo em >95% dos casos.")
        say("      Um max com o 2o ramo quase sempre inativo NAO gera b1<0 com b3>0, entao")
        say("      ma-especificacao de max nao explica o padrao observado.")

    # ---- TEOREMA: E[pico] = 2(M-1)/M EXATO, sem hipotese de configuracao ---
    # Anel com M vivos, vaos g_1..g_M, Soma(g_k)=2*pi, SEM hipotese sobre a
    # distribuicao. O agente i e' ladeado por g_{i-1} e g_i; morrendo, forma
    # g_{i-1}+g_i. Somando sobre todos os agentes, cada vao e' contado DUAS vezes:
    #     Soma_i (g_{i-1}+g_i) = 2*Soma_k g_k = 4*pi
    # Como a vitima e' uniforme entre os VIVOS (protocol_agent.py:918-920, um
    # sorteio independente por agente na mesma taxa):
    #     E[vao de fusao] = 4*pi/M   e, normalizando por 2*pi/(M-1):
    #     E[pico] = 2(M-1)/M   EXATO, PARA QUALQUER CONFIGURACAO.
    # Nao e' cota: e' a MEDIA. A dispersao e' variancia em torno de uma media exata.
    say("")
    say("-- TEOREMA: E[pico]/(2(M-1)/M) tem de valer 1,000 (media exata, nao cota) --")
    et = ev.dropna(subset=["geom_floor", "gap_rad_peak", "alive_pre"]).copy()
    et["ratio_norm"] = et["gmax_peak"] / et["geom_floor"]
    et["ratio_rad"] = et["gap_rad_peak"] / (4.0 * np.pi / et["alive_pre"])
    for lab, col in (("pico normalizado / 2(M-1)/M", "ratio_norm"),
                     ("vao de fusao [rad] / (4*pi/M)", "ratio_rad")):
        t = fit(et, col, [], [], "", quiet=True)
        if t is None:
            continue
        m = t.iloc[0]
        say(f"   {lab:<32} media = {m['coef']:.5f}  se_cl = {m['se_cl']:.5f}  "
            f"IC95 [{m['ic95_lo']:.5f}, {m['ic95_hi']:.5f}]  n={len(et)}")
        um = bool(m["ic95_lo"] <= 1.0 <= m["ic95_hi"])
        say(f"   {'':32} -> IC {'CONTEM' if um else 'NAO contem'} 1,000")
    # Desvio POSITIVO tem tres causas benignas conhecidas; a maior delas e' que o
    # MAXIMO sobre a janela e' estatistica de valor extremo. A PRIMEIRA amostra
    # pos-deteccao nao tem esse vies e testa o teorema de forma limpa.
    ef = ev.dropna(subset=["gmax_first", "floor_first", "alive_first", "gap_rad_first"]).copy()
    if len(ef) >= 20:
        ef["ratio_first"] = ef["gmax_first"] / ef["floor_first"]
        ef["ratio_first_rad"] = ef["gap_rad_first"] / (4.0 * np.pi / ef["alive_pre"])
        for lab, col in (("1a amostra / 2(M-1)/M", "ratio_first"),
                         ("1a amostra [rad] / (4pi/M)", "ratio_first_rad")):
            t = fit(ef, col, [], [], "", quiet=True)
            if t is None:
                continue
            m = t.iloc[0]
            um = bool(m["ic95_lo"] <= 1.0 <= m["ic95_hi"])
            say(f"   {lab:<32} media = {m['coef']:.5f}  se_cl = {m['se_cl']:.5f}  "
                f"IC95 [{m['ic95_lo']:.5f}, {m['ic95_hi']:.5f}]  n={len(ef)}")
            say(f"   {'':32} -> IC {'CONTEM' if um else 'NAO contem'} 1,000")
    say("   Se der sistematicamente != 1: ou as falhas nao sao uniformes entre os vivos, ou")
    say("   a extracao tem vies. Qualquer das duas e' achado.")
    say("   Tres causas BENIGNAS de desvio POSITIVO, todas de sinal conhecido:")
    say("     (i)   o pico e' o MAXIMO sobre a janela = estatistica de valor extremo sobre")
    say("           ~12-30 amostras; enviesa para cima mesmo com ruido puro;")
    say("     (ii)  o pico e' o max sobre TODOS os vaos, >= o vao de fusao;")
    say("     (iii) e' medido ~0,30 s apos a falha, ja com movimento.")
    say("   A linha '1a amostra' remove (i). Se a razao cair para ~1,000 la, o desvio do")
    say("   pico era valor extremo. Desvio NEGATIVO contradiria o teorema; positivo nao.")

    # ---- retratacao explicita ---------------------------------------------
    say("")
    say("-- RETIRADO: a evidencia de 'censura pelo piso' era artefato --")
    evf = ev.dropna(subset=["rel_floor"])
    say(f"   distribuicao vs 2(M-1)/M: abaixo {evf['below_floor'].mean():.1%} | "
        f"no piso {evf['at_floor'].mean():.1%} | acima {evf['above_floor'].mean():.1%}")
    say("   A versao anterior comparava corr(gmax_pre, gmax_peak) DENTRO de faixas estreitas")
    say("   de gmax_peak -- isto e', condicionando na VARIAVEL DEPENDENTE. Selecionar uma")
    say("   faixa de y remove quase toda a variancia de y e atenua ou INVERTE a correlacao")
    say("   com qualquer preditor, HAJA CENSURA OU NAO. As duas correlacoes condicionais")
    say("   foram RETIRADAS do argumento; so a incondicional e' interpretavel.")
    c_all = float(np.corrcoef(evf.gmax_pre, evf.gmax_peak)[0, 1])
    say(f"   corr(gmax_pre, gmax_peak) incondicional, n={len(evf)}: {c_all:+.3f}")
    say("   Alem disso o modelo 'pico ~ max(piso, f(pre))' e' internamente incoerente com o")
    say(f"   proprio dado: se fosse um max com esse piso, NADA estaria abaixo dele -- e")
    say(f"   {evf['below_floor'].mean():.1%} esta. O piso nao e' um ramo do max.")
    say("")
    say("   ACHADO SEPARADO (nao e' sobre b3): 2(M-1)/M e' violado por baixo em "
        f"{evf['below_floor'].mean():.1%}")
    say("   dos eventos de churn. Como COTA INFERIOR por evento, nao vale sob churn; e' o")
    say("   valor do caso UNIFORME. Ver a lista de formulacoes em docs/experiments/.")
    t_unc = fit(pu[pu.both_above], "dl_pico", ["dl_pre", "dl_egap_pre", "int_log"],
                ["b1", "b2", "b3"], "EM LOG, so NAO-CENSURADOS (pico > piso nos 2 bracos)")
    t_rel = fit(pu, "dl_rel", ["dl_pre", "dl_egap_pre", "int_log"], ["b1", "b2", "b3"],
                "DESFECHO NORMALIZADO log(pico/piso) + interacao")

    b3_add = t_add.iloc[3] if t_add is not None else None
    b3_log = t_log.iloc[3] if t_log is not None else None
    b3_unc = t_unc.iloc[3] if t_unc is not None else None
    b3_rel = t_rel.iloc[3] if t_rel is not None else None
    inter_flag = False
    if b3_add is not None and b3_log is not None:
        za, zl = zero_in(b3_add), zero_in(b3_log)
        zu = zero_in(b3_unc) if b3_unc is not None else None
        zr = zero_in(b3_rel) if b3_rel is not None else None
        say("")
        for lab, m in (("ADITIVO", b3_add), ("EM LOG", b3_log),
                       ("EM LOG, nao-censurados", b3_unc),
                       ("log(pico/piso)", b3_rel)):
            if m is None:
                say(f"  b3 {lab:<24} nao rodado (ausencia declarada)")
                continue
            say(f"  b3 {lab:<24} = {m['coef']:+.5f} [{m['ic95_lo']:+.5f}, "
                f"{m['ic95_hi']:+.5f}] p={m['p']:.5f} -> IC "
                f"{'contem' if zero_in(m) else 'NAO contem'} zero")
        survives = (not zl) and (zu is False) and (zr is False)
        say("")
        if za and zl:
            say("  => indistinguivel de zero nas duas escalas: decomposicao aditiva vale.")
        elif (not za) and zl:
            say("  => b3 SUMIU em log: era FORMA FUNCIONAL ERRADA (multiplicativa ajustada")
            say("     por reta), nao interacao. b3 segue DESCRITIVO.")
        else:
            say("  => b3 sobrevive aos testes de ESCALA. Isso NAO o promove a achado:")
            say("     todos os quatro modelos assumem forma aditiva ou log-aditiva, e nenhum")
            say("     acomoda um MAX. A explicacao concorrente nomeada e' ma-especificacao")
            say("     de max -- que S16 mostrou ser a estrutura correta, ainda que degenerada")
            say("     na amostra. b3 permanece DESCRITIVO ate o desfecho ser trocado pela")
            say("     FUSAO no subconjunto identificado por variavel exogena (abaixo).")
        # b3 NUNCA marca inter_flag sozinho: rotulos H1/H2 so sao suspensos por
        # heterogeneidade medida com moderador EXOGENO (por taxa / por gap_prev).
        inter_flag = False

    # ---- QUINTA explicacao para b1<0: RESTRICAO COMPOSICIONAL --------------
    # Os vaos somam 2*pi. Se o maior vale G (em unidades de 2*pi/M), os outros M-1
    # somam 2*pi(1 - G/M) e valem em media 2*pi(1-G/M)/(M-1). O sitio da morte e'
    # aleatorio entre esses outros, entao pico ~ 2(1 - G/M) e d(pico)/dG = -2/M.
    # Ou seja: b1 < 0 e' IDENTIDADE CONTABIL, nao efeito -- e vem com PREDICAO DE
    # PONTO sem parametro livre.
    say("")
    say("=" * 78)
    say("QUINTA explicacao para b1 < 0: RESTRICAO COMPOSICIONAL (predicao de ponto)")
    say("=" * 78)
    say("Os vaos somam 2*pi: um vao maior obriga os outros a serem menores, por aritmetica.")
    say("Com o sitio da morte aleatorio entre os outros: pico ~ 2(1 - G/M) => d(pico)/dG = -2/M.")
    m_med = float(ev["alive_pre"].median())
    pred_b1 = -2.0 / m_med
    b1_noint = b1                       # modelo SEM interacao: inclinacao media unica
    b1_cent = t_addc.iloc[1] if t_addc is not None else None   # COM interacao, centrado
    b1_unc = t_add.iloc[1] if t_add is not None else None       # COM interacao, NAO centrado
    say("")
    say(f"{'M mediano (vivos)':<38}{m_med:>10.2f}")
    say(f"{'PREVISTO pela composicao (-2/M)':<38}{pred_b1:>10.5f}")
    for mm in sorted(ev["alive_pre"].quantile([0.1, 0.5, 0.9]).unique()):
        say(f"   -2/M em M={mm:>5.1f}: {-2.0/mm:+.5f}")
    say("")
    say("Tres estimativas de b1 em NIVEL, e elas NAO sao a mesma coisa:")
    say(f"{'termo':<38}{'coef':>10}{'IC95 lo':>11}{'IC95 hi':>11}  contem -2/M?")
    for lab, m in (("(a) SEM interacao (inclinacao media)", b1_noint),
                   ("(b) COM interacao, CENTRADO na media", b1_cent),
                   ("(c) COM interacao, NAO centrado (em 0)", b1_unc)):
        if m is None:
            continue
        cont = bool(m["ic95_lo"] <= pred_b1 <= m["ic95_hi"])
        say(f"{lab:<38}{m['coef']:>10.5f}{m['ic95_lo']:>11.5f}{m['ic95_hi']:>11.5f}"
            f"  {'SIM' if cont else 'nao'}")
    say("")
    say("   (c) e' a inclinacao onde pre_bar = 0 -- ponto que NUNCA ocorre (pre_bar >= 1).")
    say("       E' extrapolacao fora do dado e NAO deve ser comparada com -2/M.")
    say("   (b) e' a inclinacao no centro da amostra: e' a comparavel com a identidade.")
    if b1_cent is not None:
        cont = bool(b1_cent["ic95_lo"] <= pred_b1 <= b1_cent["ic95_hi"])
        if cont:
            say("   => (b) CONTEM -2/M: b1 e' compativel com a identidade contabil. Nao ha")
            say("      magnitude em excesso para explicar, e b3 nao pode estar morando nela.")
        else:
            razao = b1_cent["coef"] / pred_b1 if abs(pred_b1) > 1e-9 else np.nan
            say(f"   => (b) NAO contem -2/M. Razao b1/(-2/M) = {razao:+.3f}: a composicao nao")
            say("      basta, e o excesso e' a grandeza a explicar.")
    if b1_noint is not None and b1_cent is not None:
        if (b1_noint["coef"] > 0) != (b1_cent["coef"] > 0):
            say("   AVISO: (a) e (b) tem SINAIS OPOSTOS. Com interacao presente, a inclinacao")
            say("   media unica e a inclinacao no centro nao coincidem; reportar as duas e")
            say("   nunca uma so.")

    # ---------- ITEM 4: desfecho = FUSAO, subconjunto por variavel exogena ---
    say("")
    say("=" * 78)
    say("ITEM 4 -- gmax_peak e' o desfecho ERRADO para D21; usar a FUSAO")
    say("=" * 78)
    say("gmax_peak so mede a fusao quando a fusao domina. No subconjunto onde o anel nao")
    say("tem nenhum vao grande, gap_max_rad NO PICO = g_pred + g_succ exatamente, e D21 e'")
    say("testavel SEM MODELO.")
    say("")
    say("Dois seletores, e a diferenca entre eles importa:")
    say("  * gmax_pre baixo  -> identifica o subconjunto certo, mas e' POS-TRATAMENTO (e' o")
    say("    estado que o metodo produziu). Serve para a checagem GEOMETRICA de D21, que e'")
    say("    identidade e nao comparacao entre bracos. NAO serve para comparar bracos.")
    say("  * gap_prev (tempo desde o evento anterior) -> EXOGENO: vem so do fluxo de falhas")
    say("    e e' identico nos dois bracos (S11). E' o seletor usado para comparar bracos.")
    if not np.allclose(pu["gap_prev"].to_numpy(float), pu["gap_prev_b"].to_numpy(float),
                       equal_nan=True):
        die("S17: gap_prev difere entre os bracos -- contradiz S11; investigar.")
    say("   S17 OK  gap_prev identico nos dois bracos (seletor exogeno confirmado)")

    say("")
    say("-- 4a. checagem GEOMETRICA de D21 (descritiva; sem comparar bracos) --")
    evg = ev.dropna(subset=["gap_rad_peak", "alive_pre"])
    say(f"{'recorte por gmax_pre':<26}{'n':>7}{'gap_rad_pico':>14}{'2*(2pi/M)':>12}"
        f"{'razao':>9}{'gmax_peak':>11}{'2(M-1)/M':>11}")
    for lab, q in (("todos", 1.0), ("<= P50", 0.50), ("<= P25", 0.25), ("<= P10", 0.10)):
        s = evg if q >= 1.0 else evg[evg.gmax_pre <= evg.gmax_pre.quantile(q)]
        if len(s) < 20:
            continue
        M = s["alive_pre"].to_numpy(float)
        pred = 2.0 * (2.0 * np.pi / M)
        obs = s["gap_rad_peak"].to_numpy(float)
        say(f"{lab:<26}{len(s):>7}{np.median(obs):>14.4f}{np.median(pred):>12.4f}"
            f"{np.median(obs / pred):>9.4f}{s['gmax_peak'].median():>11.4f}"
            f"{np.median(2.0 * (M - 1.0) / M):>11.4f}")
    say("   razao -> 1 conforme o recorte aperta = D21 confirmada: o pico E' a soma dos dois")
    say("   vaos que ladeiam quem sai, e 2(M-1)/M e' o caso particular uniforme.")

    say("")
    say("-- 4b. FUSAO comparada entre bracos, estratificada por gap_prev (EXOGENO) --")
    pu2 = pu.assign(d_fusao=pu["gap_rad_base"] - pu["gap_rad_b2"])
    qs = pu2["gap_prev"].replace(np.inf, np.nan).quantile([0.25, 0.5, 0.75]).to_numpy()
    say(f"   quartis de gap_prev: {qs[0]:.2f} / {qs[1]:.2f} / {qs[2]:.2f} s")
    say(f"{'estrato gap_prev':<22}{'n':>7}{'d_fusao [rad]':>15}{'IC95 lo':>11}"
        f"{'IC95 hi':>11}{'p':>10}")
    bins = [(-np.inf, qs[0], f"<= {qs[0]:.2f}s"), (qs[0], qs[1], f"{qs[0]:.2f}-{qs[1]:.2f}s"),
            (qs[1], qs[2], f"{qs[1]:.2f}-{qs[2]:.2f}s"), (qs[2], np.inf, f"> {qs[2]:.2f}s")]
    for lo, hi, lab in bins:
        s = pu2[(pu2.gap_prev > lo) & (pu2.gap_prev <= hi)]
        t = fit(s, "d_fusao", [], [], "", quiet=True)
        if t is None:
            say(f"{lab:<22}{len(s):>7}   nao rodado (n/clusters insuficientes)")
            continue
        m = t.iloc[0]
        say(f"{lab:<22}{len(s):>7}{m['coef']:>15.5f}{m['ic95_lo']:>11.5f}"
            f"{m['ic95_hi']:>11.5f}{m['p']:>10.5f}")
    t_fus = fit(pu2, "d_fusao", [], [], "FUSAO agregada: d_fusao ~ 1 [rad]")
    say("   d_fusao > 0 = a fusao do baseline abre um vao MAIOR, em radianos, no mesmo")
    say("   evento. E' a comparacao sem modelo e sem normalizacao por M.")

    # ---------- recortes descritivos ---------------------------------------
    say("")
    say("-- recortes (descritivos; com janela adaptativa o isolamento deixa de ser")
    say("   condicao do teste e vira caso particular: w_full = sem evento ate W_MAX) --")
    m_full = fit(pu[pu.w_full], "d_pico", ["d_gmax_pre", "d_egap_pre"], ["b1", "b2"],
                 "so eventos com janela CHEIA (w_full)")
    m_mst = fit(pu[~pu.m_unstable], "d_pico", ["d_gmax_pre", "d_egap_pre"], ["b1", "b2"],
                "so M ESTAVEL na janela (D.2)")
    m_mun = fit(pu[pu.m_unstable], "d_pico", ["d_gmax_pre", "d_egap_pre"], ["b1", "b2"],
                "so M INSTAVEL (D.2)")

    # ---------- vao absoluto ------------------------------------------------
    say("")
    t_arc = fit(pu, "d_arco_m", [], [], f"vao ABSOLUTO: d_arco_m ~ 1  (R={R_ENC:g} m)")
    say(f"   arco mediano no pico: baseline {np.median(pu['arco_base']):.3f} m, "
        f"B2 {np.median(pu['arco_b2']):.3f} m")

    # ---------- 4.5 conferencia por rodada ---------------------------------
    say("")
    say("=" * 78)
    say("CONFERENCIA (secundaria): agregado por rodada -- probe_gmax_floor.py")
    say("=" * 78)
    say("Mistura pico-do-evento com estado-pre-evento; NAO decide H1 vs H2.")
    pg.RUNS_DIR, pg.OUT_CSV, pg.T0 = RUNS, os.path.join(HERE, "gmax_por_rodada.csv"), T0_WINDOW
    pg.main()

    # ---------- veredito ----------------------------------------------------
    say("")
    say("=" * 78)
    say("VEREDITO")
    say("=" * 78)
    if alive_same is False:
        say("RESSALVA GRAVE: S15 falhou (alive_count difere entre bracos). Investigar")
        say("antes de aceitar qualquer numero abaixo.")
    say("Taxas identificaveis: " + (", ".join(f"{r:g}" for r in usable) or "NENHUMA"))
    if nao_ident:
        say("Taxas NAO identificaveis (declaradas, nao estimadas): "
            + ", ".join(f"{r:g}" for r in nao_ident))
    say("")
    for r, v in per_rate.items():
        a_ = v["alpha"]
        say(f"  taxa {r:>2g}: alpha = {a_['coef']:+.5f} [{a_['ic95_lo']:+.5f}, "
            f"{a_['ic95_hi']:+.5f}] p={a_['p']:.5f} -> IC "
            f"{'contem' if zero_in(a_) else 'NAO contem'} zero   (n={v['n']})")
    say("")
    if not concord:
        say("=> AS TAXAS DISCORDAM: essa e' a conclusao. Nao existe um alpha unico para a")
        say("   campanha; a medicao significa coisas diferentes em regimes diferentes de")
        say("   churn. Reportar por taxa. O agregado abaixo NAO e' interpretavel.")
    say(f"  AGREGADO TOTAL  = {m_tot['coef']:+.5f} [{m_tot['ic95_lo']:+.5f}, "
        f"{m_tot['ic95_hi']:+.5f}] p={m_tot['p']:.5f}")
    say(f"  AGREGADO DIRETO = {m_dir['coef']:+.5f} [{m_dir['ic95_lo']:+.5f}, "
        f"{m_dir['ic95_hi']:+.5f}] p={m_dir['p']:.5f}")
    say(f"  b1 (mediacao)   = {b1['coef']:+.5f} [{b1['ic95_lo']:+.5f}, {b1['ic95_hi']:+.5f}]")
    say(f"  MEDIADO = TOTAL - DIRETO = {m_tot['coef'] - m_dir['coef']:+.5f}")
    if concord:
        say("")
        if inter_flag:
            say("ATENCAO: interacao (em log) distinguivel de zero -- os rotulos H1/H2,")
            say("que pressupoem aditividade, nao se aplicam sem qualificacao.")
        if zero_in(m_dir) and b1["ic95_lo"] > 0.0 and not zero_in(m_tot):
            say("=> H2 no nivel do evento: alpha ~ 0, b1 > 0, TOTAL != 0. O piso deve ser")
            say("   reescrito como 'o pico e' invariante do evento DADO o estado pre'.")
        elif not zero_in(m_dir):
            say("=> ENUNCIADO DO PISO GEOMETRICO ERRADO: alpha != 0 mesmo com o mesmo")
            say("   estado pre-evento observavel. Nao suavizar.")
        elif zero_in(m_dir) and zero_in(m_tot) and zero_in(b1) and not inter_flag:
            say("=> H1 PURA: o metodo nao move o pico por caminho nenhum.")
        elif zero_in(m_dir) and zero_in(m_tot) and zero_in(b1) and inter_flag:
            say("=> H1 NO TERMO MEDIO, NAO PURA: sem efeito medio, com heterogeneidade")
            say("   dependente do estado pre. Nunca reportar como H1 pura.")
        else:
            say("=> PADRAO INTERMEDIARIO: reportar os tres coeficientes, sem forcar rotulo.")
    say("")
    say("Recortes (alpha):")
    for lab, t in (("janela cheia", m_full), ("M estavel", m_mst), ("M instavel", m_mun)):
        if t is None:
            say(f"  {lab:<14} nao rodado (ausencia declarada)")
        else:
            m = t.iloc[0]
            say(f"  {lab:<14} alpha = {m['coef']:+.5f} [{m['ic95_lo']:+.5f}, "
                f"{m['ic95_hi']:+.5f}] p={m['p']:.5f} -> IC "
                f"{'contem' if zero_in(m) else 'NAO contem'} zero")
    if alive_same:
        say("  (D.2: S15 deu 100% -- m_unstable e' descritivo; estes dois recortes sao")
        say("   verificacao redundante, nao ameaca.)")
    if t_arc is not None:
        m = t_arc.iloc[0]
        say("")
        say(f"Vao ABSOLUTO: d_arco = {m['coef']:+.5f} m [{m['ic95_lo']:+.5f}, "
            f"{m['ic95_hi']:+.5f}] p={m['p']:.5f}")
    flush()
    say("\nEscrito: gmax_events.csv, gmax_events_paired.csv, gmax_por_rodada.csv, LOG_gmax.txt")


if __name__ == "__main__":
    main()
