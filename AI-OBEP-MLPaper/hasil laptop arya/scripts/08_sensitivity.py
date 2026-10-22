"""Step 8 - sensitivity of the results to the setpoint-energy assumptions BETA and T_REF.

The setpoint effect on energy is NOT learned from ROBOD (office-1 setpoint is
constant 25 degC); it enters only through phi(T) = 1 - BETA * (T - T_REF) in
layout.evaluate(). This script re-runs NSGA-II (best tuned hyperparameters,
seeds 1-10, 25,000 evaluations) for a grid of BETA and T_REF values on the two
variants that depend on them:
  * setpoints only  - walls fixed, zone setpoints 24-27 degC
  * walls + setpoints (full problem)
The walls-only variant is independent of BETA/T_REF (setpoints fixed at 24 degC),
so it is not repeated; see decomposition.csv.

The surrogate is NOT retrained. BETA / T_REF are overridden per worker process,
so no existing file needs to be edited. T_REF also sets the air temperature at
which the surrogate is queried, so phi and the surrogate stay consistent.

Usage (from the AI-OBEP-MLPaper folder):
    python scripts/08_sensitivity.py            # full run, about 40 min on 2 cores
    python scripts/08_sensitivity.py --quick    # smoke test: 3 seeds x 3,000 evaluations
Outputs -> outputs/nsga2/sensitivity.csv, outputs/figures/fig_sensitivity_beta.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obep_mlpaper import layout as L  # noqa: E402
from obep_mlpaper.layout import T_MIN, T_MAX, case_study_1, case_study_2, evaluate  # noqa: E402
from obep_mlpaper.nsga import run_nsga2  # noqa: E402
from obep_mlpaper.surrogate import Surrogate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--jobs", type=int, default=6)
args = ap.parse_args()

EPS = 1e-6
BETA_DEFAULT, TREF_DEFAULT = L.BETA, L.T_REF            # 0.27 / 2.8, 27.4
N_SEEDS, N_EVALS = (3, 3_000) if args.quick else (10, 25_000)
SEEDS = range(1, N_SEEDS + 1)

# (beta [fraction per degC], T_ref [degC]); first row reproduces decomposition.csv
GRID = [(BETA_DEFAULT, TREF_DEFAULT)]
GRID += [(b / 100.0, TREF_DEFAULT) for b in (5.0, 7.0, 12.0)]            # BETA sweep
GRID += [(BETA_DEFAULT, t) for t in (TREF_DEFAULT - 1.0, TREF_DEFAULT + 1.0)]  # T_REF sweep

out = ROOT / "outputs" / "nsga2"
fig_dir = ROOT / "outputs" / "figures"
sur = Surrogate.load(ROOT / "outputs" / "models" / "surrogate.joblib").single_thread()


def variant(make, kind):
    sc = make()
    nw = sc.n_walls
    if kind == "setpoints only":
        sc.xl[:nw] = sc.x_init[1:-1]
        sc.xu[:nw] = sc.x_init[1:-1] + EPS
    return sc


def run_one(sc, hp, seed, beta, tref):
    """Worker: override BETA/T_REF inside this process, then run NSGA-II."""
    from obep_mlpaper import layout as Lw
    Lw.BETA, Lw.T_REF = beta, tref
    return run_nsga2(sc, sur, hp, seed, N_EVALS)


def analytic_setpoint_only(beta, tref):
    """Closed-form saving when every zone goes 24 -> 27 degC: 1 - phi(27)/phi(24)."""
    phi = lambda T: max(1.0 - beta * (T - tref), 0.0)
    return 100.0 * (1.0 - phi(T_MAX) / phi(T_MIN))


def worst_abs_pmv(r):
    return np.where(r["occ"] > 0, np.abs(r["pmv"]), 0.0).max(axis=1)


rows = []
for make in (case_study_1, case_study_2):
    name = make().name
    hp = json.loads((out / f"{name}_summary.json").read_text())["best_config_hyperparameters"]
    for beta, tref in GRID:
        for kind in ("setpoints only", "walls + setpoints"):
            sc = variant(make, kind)
            res = Parallel(n_jobs=args.jobs)(delayed(run_one)(sc, hp, s, beta, tref) for s in SEEDS)
            f0 = res[0]["f0"]
            F = np.vstack([r["F"] for r in res])
            X = np.vstack([r["X"] for r in res])

            # PMV, areas and occupants do not depend on BETA/T_REF, so evaluating X
            # with the default constants is valid for the comfort mask and the layout.
            ev = evaluate(sc, sur, X)
            ok = worst_abs_pmv(ev) <= 0.5
            row = dict(
                case=name, variant=kind, beta_pct_per_C=round(100 * beta, 2), t_ref_C=round(tref, 2),
                n_seeds=N_SEEDS, n_evals=N_EVALS, baseline_kW=f0[0],
                max_energy_saving_pct=100 * (1 - F[:, 0].min() / f0[0]),
                analytic_24to27_saving_pct=analytic_setpoint_only(beta, tref),
                saving_all_pmv_ok_pct=(100 * (1 - F[ok, 0].min() / f0[0])) if ok.any() else np.nan,
                solution_C_exists=bool(ok.any()))
            if ok.any():                                     # describe solution C
                i = np.flatnonzero(ok)[np.argmin(F[ok, 0])]
                occ_rooms = ev["occ"][i] > 0
                row.update(C_setpoint_min=ev["setpoints"][i].min(), C_setpoint_max=ev["setpoints"][i].max(),
                           C_largest_room_m2=ev["area"][i].max(),
                           C_largest_room_pax=int(ev["occ"][i][np.argmax(ev["area"][i])]),
                           C_worst_abs_pmv=np.abs(ev["pmv"][i][occ_rooms]).max())
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)   # save as we go

df = pd.DataFrame(rows)
print(df.round(2).to_string(index=False))

# ---- figure: saving versus BETA, measured by NSGA-II and closed form ---------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    beta_rows = df[np.isclose(df.t_ref_C, TREF_DEFAULT)]
    for ax, case in zip(axes, df.case.unique()):
        d = beta_rows[beta_rows.case == case]
        bs = np.linspace(2, 14, 100)
        ax.plot(bs, [analytic_setpoint_only(b / 100, TREF_DEFAULT) for b in bs], "k--", lw=1,
                label="closed form, all zones 24→27 °C")
        for kind, mk in (("setpoints only", "o"), ("walls + setpoints", "s")):
            s = d[d.variant == kind].sort_values("beta_pct_per_C")
            ax.plot(s.beta_pct_per_C, s.max_energy_saving_pct, mk + "-", label=f"{kind}: max saving")
        s = d[d.variant == "walls + setpoints"].sort_values("beta_pct_per_C")
        ax.plot(s.beta_pct_per_C, s.saving_all_pmv_ok_pct, "^-", label="walls + setpoints: all |PMV| ≤ 0.5")
        ax.set_title(case)
        ax.set_xlabel("β [% per °C]")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("energy saving vs 24 °C baseline [%]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_sensitivity_beta.png", dpi=200)
    print("figure ->", fig_dir / "fig_sensitivity_beta.png")
except Exception as e:  # plotting must never lose the CSV
    print("figure skipped:", e)