#!/usr/bin/env python
"""Metricas canonicas de E_gap pos-evento (compartilhado pelos scripts de Fase 2/3).

PRINCIPAL: t_settle (tempo de assentamento robusto, estilo controle classico) -- nao depende de
o decaimento ser exponencial, ao contrario do tau_fit. Definicao acordada:
    t_settle = menor t* (relativo a t0) tal que, para todo t >= t*,
               |E_gap(t) - E_gap_inf| <= band_frac * E_gap_peak
onde E_gap_inf = mediana de E_gap na janela final (asintota) e E_gap_peak = pico pos-t0.
Reporta o PAR (t_settle, egap_settle): t_settle = "quanto tempo"; egap_settle (=E_gap_inf) =
"deu certo?" (baixo = reconfigurou; alto = travou). Se o sinal nunca permanece na banda ate o
fim -> t_settle = inf (= nao assentou). tau_fit fica como SECUNDARIO (taxa/forma, so confiavel
com R2 alto). Para churn continuo (sem assentamento) use egap_avg.
"""
import os
import sys

import numpy as np

TAIL_FLOOR_FRAC = 0.05      # piso do ajuste exponencial (fracao do pico)
SETTLE_BAND_FRAC = 0.05     # banda do t_settle: +-5% do pico (padrao robusto de livro)
SETTLE_NOISE_K = 3.0        # piso da banda = K * late_std (3 sigma -> ~99.7% do jitter dentro)
ASYMPTOTE_WIN = 1.0         # s, janela final p/ estimar E_gap_inf
LATE_WIN = 20.0             # s, janela final p/ late_std


def exp_tau(t, e):
    """Ajuste exponencial da cauda (do pico ate 5% do pico). Retorna (tau, r2). NaN se nao cabe."""
    if t.size < 5:
        return float("nan"), float("nan")
    i_pk = int(np.nanargmax(e)); e_pk = e[i_pk]
    mask = (np.arange(e.size) >= i_pk) & (e > TAIL_FLOOR_FRAC * e_pk) & np.isfinite(e)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    tt, ee = t[mask], e[mask]
    coef = np.polyfit(tt, np.log(ee), 1)
    tau = -1.0 / coef[0] if coef[0] < 0 else float("nan")
    pred = np.polyval(coef, tt)
    ss_res = float(np.sum((np.log(ee) - pred) ** 2))
    ss_tot = float(np.sum((np.log(ee) - np.mean(np.log(ee))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(tau), float(r2)


def settling_time(t, e, band_frac=SETTLE_BAND_FRAC, asymptote_win=ASYMPTOTE_WIN,
                  late_std=None, noise_k=SETTLE_NOISE_K):
    """t_settle (relativo a t[0]) e egap_settle (=E_gap_inf). t_settle=inf se nunca permanece.

    banda = max(band_frac * pico, noise_k * late_std): o piso noise_k*late_std impede que a banda
    fique menor que ~3 sigma do jitter de regime (senao o sinal convergido espeta para fora e o
    t_settle nunca dispara). Classico: t_settle = instante apos a ULTIMA saida da banda em torno
    da assintota. Robusto a forma do decaimento; nao exige exponencial."""
    if t.size == 0:
        return float("nan"), float("nan")
    e_peak = float(np.nanmax(e))
    t_end = float(t[-1])
    win_mask = t >= (t_end - asymptote_win)
    e_inf = float(np.nanmedian(e[win_mask])) if win_mask.any() else float(e[-1])
    band = band_frac * e_peak
    if late_std is not None and np.isfinite(late_std):
        band = max(band, noise_k * float(late_std))   # piso de ruido (3 sigma)
    outside = np.abs(e - e_inf) > band
    if not outside.any():
        return 0.0, e_inf                       # sempre dentro da banda -> assentado em t0
    last_out = int(np.max(np.where(outside)[0]))
    if last_out >= e.size - 1:
        return float("inf"), e_inf              # ainda fora no fim -> nao assentou
    t_settle = float(t[last_out + 1] - t[0])
    return t_settle, e_inf


def overshoot_frac(t, e, band_frac=SETTLE_BAND_FRAC, asymptote_win=ASYMPTOTE_WIN,
                   late_std=None, noise_k=SETTLE_NOISE_K):
    """Re-excursao apos a primeira entrada na banda, como fracao da amplitude do evento.

    Definicao (controle classico adaptado ao decaimento de E_gap):
      overshoot = max(0, max(E_gap apos a 1a entrada na banda) - (E_gap_inf + banda))
                  normalizado por (E_gap_peak - E_gap_inf).
    O excesso e' medido ALEM da banda (nao da assintota): a 1a amostra dentro da
    banda fica na borda dela, entao medir a partir de E_gap_inf reportaria
    ~band_frac para um decaimento perfeitamente monotonico. Com esta definicao:
    0.0 = assentamento monotonico; > 0 = ringing/re-excursao (o sinal volta a
    SAIR da banda depois de te-la alcancado). NaN se nunca entra na banda.
    """
    if t.size == 0:
        return float("nan")
    e_peak = float(np.nanmax(e))
    t_end = float(t[-1])
    win_mask = t >= (t_end - asymptote_win)
    e_inf = float(np.nanmedian(e[win_mask])) if win_mask.any() else float(e[-1])
    band = band_frac * e_peak
    if late_std is not None and np.isfinite(late_std):
        band = max(band, noise_k * float(late_std))
    inside = np.abs(e - e_inf) <= band
    if not inside.any():
        return float("nan")                      # nunca alcancou a banda
    i_first = int(np.argmax(inside))             # primeira entrada
    amp = e_peak - e_inf
    if not np.isfinite(amp) or amp <= 0.0:
        return 0.0
    post = e[i_first:]
    return float(max(0.0, float(np.nanmax(post)) - (e_inf + band)) / amp)


def event_metrics(df, t0):
    """df: DataFrame com colunas timestamp, E_gap. Retorna dict de metricas pos-t0."""
    full = df
    sub = df[df["timestamp"] >= t0].reset_index(drop=True)
    if sub.empty:
        return {}
    t = sub["timestamp"].to_numpy(float)
    e = sub["E_gap"].to_numpy(float)
    tau, r2 = exp_tau(t, e)
    t_end = float(full["timestamp"].max())
    late = full[full["timestamp"] >= t_end - LATE_WIN]["E_gap"].to_numpy(float)
    late_std = float(np.std(late)) if late.size else float("nan")
    # t_settle com piso de ruido, reportado em 2% e 5% (robustez a P). 5% = primario.
    t_settle, egap_settle = settling_time(t, e, band_frac=0.05, late_std=late_std)
    t_settle_2pct, _ = settling_time(t, e, band_frac=0.02, late_std=late_std)
    steady = sub[sub["timestamp"] >= t0 + 10.0]["E_gap"].to_numpy(float)
    egap_avg = float(np.mean(steady)) if steady.size else float("nan")
    return {
        "tau_fit": tau, "tau_fit_r2": r2,
        "t_settle": t_settle, "t_settle_2pct": t_settle_2pct, "egap_settle": egap_settle,
        "egap_peak": float(np.nanmax(e)), "egap_final": float(e[-1]),
        "egap_avg": egap_avg, "egap_late_std": late_std,
        "overshoot_frac": overshoot_frac(t, e, late_std=late_std),
    }


def effort_metrics(agent_csv, t0=0.0, vmax=10.0):
    """Metricas de esforco/saturacao/fairness do agent_telemetry.csv (estilo M5/M6/M2).

    CHAMAR ANTES de o runner apagar o agent_telemetry.csv. Le apenas as colunas
    necessarias (o arquivo pode ter dezenas de MB).
      effort_mean_v2 : mean((velocity_norm/vmax)^2), t >= t0  (esforco de controle, M5)
      sat_frac       : Pr(velocity_norm >= vmax - tol), t >= t0 (saturacao, M6)
      fairness_p95   : P95 entre nos do P95 por-no de |e_tau_real| (fairness, M2;
                       cai para e_tau quando e_tau_real nao existe)
    Retorna {} se o arquivo nao existe/esta vazio.
    """
    import pandas as pd  # local: o resto do modulo so precisa de numpy
    if not agent_csv or not os.path.exists(agent_csv):
        return {}
    try:
        header = pd.read_csv(agent_csv, nrows=0).columns
    except Exception:
        return {}
    e_col = "e_tau_real" if "e_tau_real" in header else ("e_tau" if "e_tau" in header else None)
    usecols = [c for c in ("node_id", "timestamp", "velocity_norm", e_col) if c in header]
    if "timestamp" not in usecols or "velocity_norm" not in usecols:
        return {}
    try:
        df = pd.read_csv(agent_csv, usecols=usecols)
    except Exception:
        return {}
    df = df[df["timestamp"] >= t0]
    if df.empty:
        return {}
    v = df["velocity_norm"].to_numpy(float)
    v = v[np.isfinite(v)]
    out = {}
    if v.size and np.isfinite(vmax) and vmax > 0:
        vn = v / float(vmax)
        out["effort_mean_v2"] = float(np.mean(vn * vn))
        out["sat_frac"] = float(np.mean(v >= float(vmax) - 1e-9))
    if e_col is not None and "node_id" in df.columns:
        per_node = df.groupby("node_id")[e_col].apply(
            lambda s: float(np.percentile(np.abs(s.to_numpy(float)), 95)) if len(s) else float("nan")
        ).to_numpy(float)
        per_node = per_node[np.isfinite(per_node)]
        if per_node.size:
            out["fairness_p95"] = float(np.percentile(per_node, 95))
    return out


def run_provenance(run_dir):
    """Colunas de proveniencia (git + parametros fixados) de UMA celula do sweep.

    Le o run_manifest.json que o main.py escreveu DENTRO de run_dir -- ou seja, o
    estado resolvido do processo FILHO. Nunca chame provenance.summary_provenance()
    direto de um runner: o runner e' o processo PAI e o config_param dele carrega os
    defaults do pai, nao o env fixado para o filho.

    Uso no runner (regra 5 da campanha: toda linha carrega seed/commit/dirty):
        m.update(run_provenance(run_dir))

    Retorna {} se o manifesto nao existir (rodada antiga / main.py sem proveniencia),
    para que "nao registrado" seja distinguivel de "registrado como default".
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import provenance
    except Exception:
        return {}
    return provenance.summary_from_manifest(provenance.read_run_manifest(run_dir))


def aggregate_seeds(values):
    """Mediana, pior caso (max) e desvio entre seeds, NaN-safe.

    Para todas as metricas de campanha "maior = pior" (egap_*, t_settle, effort,
    sat_frac), o pior caso e' o maximo. Reportar mediana E pior caso evita
    conclusoes baseadas so na tendencia central (requisito da campanha).
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "worst": float("nan"), "std": float("nan")}
    return {"median": float(np.median(arr)), "worst": float(np.max(arr)),
            "std": float(np.std(arr))}


def _selftest():
    t = np.linspace(0.0, 60.0, 6001)   # dt=0.01
    t0 = 0.0

    def report(name, e):
        late = e[t >= t[-1] - LATE_WIN]
        lstd = float(np.std(late)) if late.size else float("nan")
        ts5, ei = settling_time(t, e, 0.05, late_std=lstd)
        ts2, _ = settling_time(t, e, 0.02, late_std=lstd)
        tau, r2 = exp_tau(t, e)
        print(f"  {name:20s} t_settle(5%)={ts5:7.2f} (2%)={ts2:7.2f}  egap_settle={ei:.4f}  "
              f"late_std={lstd:.4f}  tau={tau:7.2f}(R2={r2:.2f})")

    # 1) exponencial limpo: pico 0.3, tau=2 -> t_settle ~ tau*ln(peak/band)=2*ln(0.3/0.009)~7s
    report("exp(tau=2)", 0.3 * np.exp(-t / 2.0))
    # 2) exponencial lento: tau=15 -> t_settle ~ 15*ln(33)~52s (NAO dispara cedo, ao contrario de
    #    'variacao<P% numa janela', que dispararia na cauda lenta)
    report("exp(tau=15) lento", 0.3 * np.exp(-t / 15.0))
    # 3) travado-alto: decai um pouco e estaciona em 0.5 -> t_settle cedo MAS egap_settle alto
    report("travado@0.5", 0.5 + 0.0 * t)
    # 4) nunca assenta (oscila +-0.2 em torno de 0.3) -> t_settle=inf
    report("oscilante", 0.3 + 0.2 * np.sin(t * 3.0))
    # 5) decai e depois oscila pequeno (ruido 0.5% do pico) -> assenta perto de 0
    rng = np.cos(t * 50.0) * 0.0015
    report("exp+ruido_pequeno", np.maximum(0.0, 0.3 * np.exp(-t / 2.0)) + rng)


if __name__ == "__main__":
    print("Self-test metrics_util.settling_time / exp_tau:")
    _selftest()
