"""Step 7 - where do the savings come from? Walls vs setpoints.

Runs NSGA-II (best tuned hyperparameters, 10 seeds, 25,000 evaluations) on three
variants of each case study:
  * setpoints only  - walls fixed at the existing layout, zone setpoints 24-27 degC
  * walls only      - setpoints fixed at 24 degC (the policy floor), walls movable
  * walls + setpoints (the full problem)
The walls-only result is independent of the setpoint-energy coefficient BETA,
so it isolates what the movable-wall layout contributes on its own.
Outputs -> outputs/nsga2/decomposition.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obep_mlpaper.layout import T_MIN, case_study_1, case_study_2, evaluate  # noqa: E402
from obep_mlpaper.nsga import run_nsga2  # noqa: E402
from obep_mlpaper.surrogate import Surrogate  # noqa: E402

EPS = 1e-6
sur = Surrogate.load(ROOT / "outputs" / "models" / "surrogate.joblib").single_thread()
out = ROOT / "outputs" / "nsga2"
SEEDS = range(1, 11)


def variant(make, kind):
    sc = make()
    nw = sc.n_walls
    if kind == "setpoints only":
        sc.xl[:nw] = sc.x_init[1:-1]
        sc.xu[:nw] = sc.x_init[1:-1] + EPS
    elif kind == "walls only":
        sc.xl[nw:] = T_MIN
        sc.xu[nw:] = T_MIN + EPS
    return sc


def worst_abs_pmv(r):
    return np.where(r["occ"] > 0, np.abs(r["pmv"]), 0.0).max(axis=1)


rows = []
for make in (case_study_1, case_study_2):
    name = make().name
    summ = json.loads((out / f"{name}_summary.json").read_text())
    hp = summ["best_config_hyperparameters"]
    for kind in ("setpoints only", "walls only", "walls + setpoints"):
        sc = variant(make, kind)
        res = Parallel(n_jobs=2)(delayed(run_nsga2)(sc, sur, hp, s, 25_000) for s in SEEDS)
        f0 = res[0]["f0"]
        F = np.vstack([r["F"] for r in res])
        X = np.vstack([r["X"] for r in res])
        ok = worst_abs_pmv(evaluate(sc, sur, X)) <= 0.5
        rows.append(dict(
            case=name, variant=kind, hv_mean=np.mean([r["hv"] for r in res]),
            max_energy_saving_pct=100 * (1 - F[:, 0].min() / f0[0]),
            max_comfort_gain_pct=100 * (1 - F[:, 1].min() / f0[1]),
            energy_saving_with_all_pmv_ok_pct=(100 * (1 - F[ok, 0].min() / f0[0])) if ok.any() else np.nan))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out / "decomposition.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
