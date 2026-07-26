#!/usr/bin/env python
"""Generate the M-mult adjacent-fault benchmark data for fig22.

Reuses diag_churn.run() (canonical B2 det-fault harness) to run the 5
deterministic scenarios (k1 / adj2 / adj3 / non2 / non3) with
DUAL_PULSE_MULTIPLICITY OFF and ON, at N=24, dt=repo default (0.05).
Writes mmult_adjacent_results.csv: scenario,k,mmult,tau_fit,t_settle,egap_final.

Expected (CAMPAIGN_LOG Ciclo 2): adj2 tau 13.6->2.2, adj3 15.5->2.2 (~6-7x);
k1/non2/non3 byte-identical (k=1 path untouched).
"""
import os
import sys

import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import diag_churn  # noqa: E402

diag_churn.N = 24
diag_churn.BUDGET = 60.0
diag_churn.RUNS = os.path.join(EXP, "mmult_runs")
os.makedirs(diag_churn.RUNS, exist_ok=True)


def main():
    rows = []
    for mm in (False, True):
        print(f"\n=== M-mult {'ON' if mm else 'OFF'} (N=24, det permanent faults) ===")
        for name, ids, k in diag_churn.DET_SCENARIOS:
            det_env = {
                "DETERMINISTIC_FAILURE_ENABLE": "True",
                "DETERMINISTIC_FAILURE_AGENT_ID": ids,
                "DETERMINISTIC_FAILURE_TIME_T0": str(diag_churn.T0),
                "DETERMINISTIC_FAILURE_OFF_TIME": "-1.0",
                "DUAL_PULSE_MULTIPLICITY": str(mm),
            }
            res = diag_churn.run(f"{name}_mm{int(mm)}", det_env,
                                 expected_N_new=diag_churn.N - k)
            rows.append({
                "scenario": name, "k": k, "mmult": bool(mm),
                "tau_fit": res.get("tau_fit"), "t_settle": res.get("t_settle"),
                "egap_final": res.get("egap_final"),
            })
    out = os.path.join(EXP, "mmult_adjacent_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {os.path.relpath(out, EXP)}")


if __name__ == "__main__":
    main()
