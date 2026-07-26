#!/usr/bin/env python
"""Inventario de reprodutibilidade: quais *_results.csv carregam proveniencia.

Varre os *_results.csv do diretorio e reporta, por arquivo, quantas linhas tem e
quais campos de proveniencia FALTAM. E' a foto do que e' e do que nao e'
reprodutivel hoje -- a lista de campos ausentes de um CSV antigo NAO e' um bug a
corrigir reescrevendo o arquivo (regra 2: nunca sobrescrever resultado), e sim o
registro de que aquele numero nao pode ser re-derivado.

Le apenas o CABECALHO + uma coluna para contar linhas (regra 7: nao carregar CSV
inteiro em memoria).

Uso:
    python experiments/scaling_law/check_provenance.py
    python experiments/scaling_law/check_provenance.py --all      # inclui _archive/
    python experiments/scaling_law/check_provenance.py --verbose  # lista campo a campo
"""
import argparse
import csv
import glob
import os

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# NUCLEO (regra 5): sem estes tres, a linha nao e' re-executavel de forma alguma.
# Cada entrada e' (nome_canonico, aliases aceitos) -- os runners historicos
# gravam "seed", que conta como experiment_seed.
CORE_FIELDS = [
    ("experiment_seed", ("experiment_seed", "seed")),
    ("git_commit", ("git_commit", "commit")),
    ("git_dirty", ("git_dirty", "dirty")),
]

# PARAMETROS fixados (regra 3). Aliases cobrem os nomes curtos que os runners ja
# usam nas suas proprias colunas (N, dt, tau_xy, T_FF, k_e_tau).
PARAM_FIELDS = [
    ("num_agents", ("num_agents", "N")),
    ("control_period", ("control_period", "dt")),
    ("k_e_tau", ("k_e_tau", "K_E_TAU")),
    ("vm_tau_xy", ("vm_tau_xy", "tau_xy")),
    ("dual_pulse_integration", ("dual_pulse_integration", "integration")),
    ("dual_pulse_delta_scale", ("dual_pulse_delta_scale", "delta_scale")),
    ("dual_pulse_t_ff", ("dual_pulse_t_ff", "T_FF")),
    ("dual_pulse_ttl_hops", ("dual_pulse_ttl_hops", "ttl", "ttl_hops")),
    ("sim_duration", ("sim_duration", "duration")),
    ("communication_delay", ("communication_delay", "delay")),
    ("communication_failure_rate", ("communication_failure_rate", "loss")),
]


def header_and_rows(path):
    """(header_list, n_rows). Le so o cabecalho e conta linhas de dados."""
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return [], 0
            n = sum(1 for _ in reader)
        return header, n
    except OSError as exc:
        print(f"  ! {os.path.basename(path)}: {exc}")
        return None, 0


def missing_fields(header, spec):
    present = {c.strip() for c in header}
    return [name for name, aliases in spec if not present.intersection(aliases)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="inclui _archive/ e subdiretorios")
    ap.add_argument("--verbose", action="store_true", help="lista os campos ausentes campo a campo")
    args = ap.parse_args()

    # figure_data*.csv nao casa com *_results*.csv mas e' resultado primario: e' o
    # que make_figures.py / make_table.py leem para as figuras da tese.
    patterns = ["*_results*.csv", "figure_data*.csv"]
    if args.all:
        patterns = ["**/" + p for p in patterns]
    paths = sorted({p for pat in patterns
                    for p in glob.glob(os.path.join(EXP_DIR, pat), recursive=args.all)})
    if not paths:
        print(f"Nenhum *_results.csv em {EXP_DIR}")
        return

    print(f"Inventario de proveniencia -- {len(paths)} arquivo(s) em {EXP_DIR}")
    print("  nucleo  = experiment_seed | git_commit | git_dirty   (regra 5)")
    print("  params  = N, dt, K_E_TAU, tau_a, integration, delta_scale, T_FF, TTL, "
          "duration, delay, loss   (regra 3)")
    print()
    print(f"{'arquivo':<46} {'linhas':>7} {'nucleo':>8} {'params':>8}  faltando (nucleo)")
    print("-" * 104)

    full_core = full_all = 0
    total_rows = 0
    rows_out = []
    for p in paths:
        header, n = header_and_rows(p)
        if header is None:
            continue
        miss_core = missing_fields(header, CORE_FIELDS)
        miss_param = missing_fields(header, PARAM_FIELDS)
        have_core = len(CORE_FIELDS) - len(miss_core)
        have_param = len(PARAM_FIELDS) - len(miss_param)
        total_rows += n
        if not miss_core:
            full_core += 1
            if not miss_param:
                full_all += 1
        rel = os.path.relpath(p, EXP_DIR)
        label = ",".join(miss_core) if miss_core else "-"
        print(f"{rel:<46} {n:>7} {have_core:>4}/{len(CORE_FIELDS):<3} "
              f"{have_param:>4}/{len(PARAM_FIELDS):<3}  {label}")
        rows_out.append((rel, miss_core, miss_param))

    print("-" * 104)
    print(f"{len(paths)} arquivos, {total_rows} linhas de resultado.")
    print(f"  com nucleo completo (seed+commit+dirty): {full_core}/{len(paths)}")
    print(f"  com nucleo + params completos:           {full_all}/{len(paths)}")
    if full_core < len(paths):
        print("\n  => Os arquivos sem nucleo completo NAO sao reprodutiveis a partir da")
        print("     propria linha. Nao reescreva-os (regra 2): re-rode e grave em arquivo")
        print("     novo, arquivando o antigo em _archive/ com nota no CAMPAIGN_LOG.md.")

    if args.verbose:
        print("\n--- detalhe por arquivo ---")
        for rel, miss_core, miss_param in rows_out:
            print(f"\n{rel}")
            print(f"  nucleo ausente: {', '.join(miss_core) if miss_core else '(nenhum)'}")
            print(f"  params ausentes: {', '.join(miss_param) if miss_param else '(nenhum)'}")


if __name__ == "__main__":
    main()
