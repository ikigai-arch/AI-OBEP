"""Step 1 - surrogate baselines: ablation of the data steps, default model zoo,
and a replication of the Lyu et al. (2025) protocol.

A. Ablation (held-out test days, ExtraTrees & XGBoost defaults):
   raw 5-min -> + outlier smoothing -> + hourly aggregation -> + time-of-day/schedule
B. Default model zoo + weighted ensemble on the deployed (hourly) data
C. Paper protocol: office-1, first 12 weekdays, 5-min, smoothing, 7 features,
   random 70/15/15 row split (as AutoGluon-Tabular does by default)
Outputs -> outputs/surrogate/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obep_mlpaper.data import (FEATURES, FEATURES_PAPER, TARGET, ROOM_AREA, add_time_features,  # noqa: E402
                               day_split, five_minute_dataset, hourly_dataset, load_room)
from obep_mlpaper.surrogate import WeightedEnsemble, metrics, model_zoo  # noqa: E402

out = ROOT / "outputs" / "surrogate"
out.mkdir(parents=True, exist_ok=True)


def fit_eval(name, make, Xtr, ytr, Xte, yte, op=None):
    m = make().fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None)
    row = dict(model=name, **metrics(yte, p))
    if op is not None:
        row["R2_operating"] = metrics(yte[op], p[op])["R2"]
    return row, m, p


# ---------------------------------------------------------------- A. ablation
abl = []
steps = []
raw = add_time_features(load_room())
raw["y"] = raw["energy_kwh"] * 12 * 1000 / ROOM_AREA
steps.append(("1 raw 5-min, 7 paper features", raw, FEATURES_PAPER))
sm = five_minute_dataset(smooth=True)
sm["y"] = sm["target_kwh"] * 12 * 1000 / ROOM_AREA
steps.append(("2 + outlier smoothing (Hampel, adjacent mean)", sm, FEATURES_PAPER))
hr = hourly_dataset()
hr["y"] = hr[TARGET]
steps.append(("3 + hourly aggregation", hr, FEATURES_PAPER))
steps.append(("4 + time of day & HVAC schedule (deployed)", hr, FEATURES))
zoo = model_zoo()
for label, d, feats in steps:
    tr, te = day_split(d)
    X, y = d[feats].to_numpy(), d["y"].to_numpy()
    op = d["hvac_schedule"].to_numpy()[te] == 1
    for name in ("ExtraTrees", "XGBoost"):
        row, _, _ = fit_eval(name, lambda: model_zoo()[name], X[tr], y[tr], X[te], y[te], op)
        abl.append(dict(step=label, **row))
        print(f"{label:48s} {name:10s} R2={row['R2']:.3f}  R2_op={row['R2_operating']:.3f}  "
              f"CV(RMSE)={row['CVRMSE_pct']:.1f}%")
pd.DataFrame(abl).to_csv(out / "ablation_data_steps.csv", index=False)

# ---------------------------------------------------------------- B. zoo on deployed data
d = hourly_dataset()
tr, te = day_split(d)
X, y, g = d[FEATURES].to_numpy(), d[TARGET].to_numpy(), d["day"].to_numpy()
op = d["hvac_schedule"].to_numpy()[te] == 1
rows = []
for name in zoo:
    row, _, _ = fit_eval(name, lambda: model_zoo()[name], X[tr], y[tr], X[te], y[te], op)
    rows.append(row)
makers = {k: (lambda k=k: model_zoo()[k]) for k in ("ExtraTrees", "RandomForest", "LightGBM", "XGBoost", "KNN", "MLP")}
ens = WeightedEnsemble(makers).fit(X[tr], y[tr], g[tr])
p = np.clip(ens.predict(X[te]), 0, None)
rows.append(dict(model="WeightedEnsemble", **metrics(y[te], p), R2_operating=metrics(y[te][op], p[op])["R2"]))
zoo_res = pd.DataFrame(rows)
zoo_res.to_csv(out / "baseline_zoo_test_metrics.csv", index=False)
print("\nDefault zoo (hourly, held-out days), ensemble weights:", ens.weight_dict())
print(zoo_res.round(3).to_string(index=False))

# ---------------------------------------------------------------- C. paper protocol
p12 = five_minute_dataset(n_days=12, smooth=True)
rng = np.random.RandomState(0)
idx = rng.permutation(len(p12))
n = len(idx)
itr, iva, ite = idx[:int(.7 * n)], idx[int(.7 * n):int(.85 * n)], idx[int(.85 * n):]
Xp, yp = p12[FEATURES_PAPER].to_numpy(), p12["target_kwh"].to_numpy()
prow, models = [], {}
for name in zoo:
    row, m, _ = fit_eval(name, lambda: model_zoo()[name], Xp[itr], yp[itr], Xp[ite], yp[ite])
    prow.append(row)
    models[name] = m
# AutoGluon-style L2 weighted ensemble: weights fit on the validation rows
from scipy.optimize import nnls  # noqa: E402
members = [k for k in models if k != "Ridge"]
Pv = np.column_stack([models[k].predict(Xp[iva]) for k in members])
w, _ = nnls(Pv, yp[iva])
w = w / w.sum()
pt = np.clip(np.column_stack([models[k].predict(Xp[ite]) for k in members]) @ w, 0, None)
prow.append(dict(model="WeightedEnsemble_L2", **metrics(yp[ite], pt)))
prow.append(dict(model="Lyu et al. 2025 (AutoGluon WeightedEnsemble_L2)", R2=0.933, MAPE_pct=26.094))
prow.append(dict(model="Lyu et al. 2025 (LSTM)", R2=0.921, MAPE_pct=19.358))
paper = pd.DataFrame(prow)
paper.to_csv(out / "paper_protocol_metrics.csv", index=False)
print("\nPaper protocol (12 days, 5-min, random rows):")
print(paper[["model", "R2", "RMSE", "MAPE_pct", "CVRMSE_pct"]].round(3).to_string(index=False))
