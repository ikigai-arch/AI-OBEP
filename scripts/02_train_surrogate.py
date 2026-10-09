"""Train the room-level HVAC-energy surrogate (paper Sec. 3): 7 features -> energy.

Pools the two office rooms (3 and 4) of ROBOD, target = Wh/m2 per 5 min, day-level
train/val/test split. Reports metrics on all hours and on HVAC operating hours only.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiobep.data import FEATURES, OFFICE_ROOMS, load_rooms, split_by_day  # noqa: E402
from aiobep.surrogate import Surrogate, WeightedEnsemble, metrics, model_zoo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rooms", type=int, nargs="+", default=list(OFFICE_ROOMS))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out = ROOT / "outputs"
(out / "models").mkdir(parents=True, exist_ok=True)

df = load_rooms(args.rooms).dropna(subset=FEATURES + ["hvac_energy_wh_m2"]).reset_index(drop=True)
df["split"] = split_by_day(df, seed=args.seed)
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
print(f"rows train/val/test = {len(tr)}/{len(va)}/{len(te)}  "
      f"(days: {tr.groupby('room').day.nunique().to_dict()} / "
      f"{va.groupby('room').day.nunique().to_dict()} / {te.groupby('room').day.nunique().to_dict()})")

Xtr, ytr = tr[FEATURES].to_numpy(), tr["hvac_energy_wh_m2"].to_numpy()
models = model_zoo(args.seed)
for k, m in models.items():
    m.fit(Xtr, ytr)
ens = WeightedEnsemble({k: m for k, m in models.items() if k != "Ridge"})
print("ensemble weights:", ens.fit_weights(va[FEATURES].to_numpy(), va["hvac_energy_wh_m2"].to_numpy()))
candidates = {**models, "WeightedEnsemble": ens}

rows = []
preds = {}
for name, m in candidates.items():
    p = np.clip(m.predict(te[FEATURES].to_numpy()), 0, None)
    preds[name] = p
    for tag, mask in (("all hours", np.ones(len(te), bool)), ("operating hours", te.is_operating.to_numpy())):
        rows.append(dict(model=name, subset=tag, **metrics(te.hvac_energy_wh_m2.to_numpy()[mask], p[mask])))
res = pd.DataFrame(rows)
res.to_csv(out / "surrogate_metrics.csv", index=False)
print(res.round(3).to_string(index=False))

# Do the 7 features carry any information *inside* operating hours?  Baseline = predict the
# operating-hours training mean (R2 ~ 0 by construction).
base = tr[tr.is_operating].hvac_energy_wh_m2.mean()
print(f"\nconstant baseline RMSE (operating hours): "
      f"{np.sqrt(np.mean((te[te.is_operating].hvac_energy_wh_m2 - base) ** 2)):.3f}")

# Generalisation to an unseen room: train on one office, test on the other.
if len(args.rooms) == 2:
    loro = []
    for a, b in (args.rooms, args.rooms[::-1]):
        d_a, d_b = df[df.room == a], df[df.room == b]
        m = model_zoo(args.seed)["ExtraTrees"].fit(d_a[FEATURES].to_numpy(), d_a.hvac_energy_wh_m2.to_numpy())
        loro.append(dict(train_room=a, test_room=b,
                         **metrics(d_b.hvac_energy_wh_m2, m.predict(d_b[FEATURES].to_numpy()))))
    print("\ncross-room generalisation (ExtraTrees):\n", pd.DataFrame(loro).round(3).to_string(index=False))
    pd.DataFrame(loro).to_csv(out / "surrogate_cross_room.csv", index=False)

# Save the surrogates used by the optimiser.
meta = dict(rooms=args.rooms, target="HVAC energy [Wh/m2 per 5 min]", features=FEATURES)
Surrogate(ens, FEATURES, meta).save(out / "models" / "surrogate_ensemble.joblib")
Surrogate(models["MLP"], FEATURES, meta).save(out / "models" / "surrogate_mlp.joblib")

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
y = te.hvac_energy_wh_m2.to_numpy()
for name in ("ExtraTrees", "MLP", "WeightedEnsemble"):
    ax[0].scatter(y, preds[name], s=3, alpha=.3, label=name)
lim = [0, max(y.max(), 1)]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set(xlabel="actual [Wh/m²/5min]", ylabel="predicted", title="Test set: predicted vs actual")
ax[0].legend(markerscale=4)
piv = res.pivot(index="model", columns="subset", values="R2")
piv.plot.bar(ax=ax[1]); ax[1].set(title="R² by subset (test)", ylabel="R²"); ax[1].axhline(0, color="k", lw=.8)
ax[1].tick_params(axis="x", rotation=30)
te2 = te.assign(pred=preds["WeightedEnsemble"])
prof = te2.groupby(te2.ts.dt.hour)[["hvac_energy_wh_m2", "pred"]].mean()
prof.plot(ax=ax[2]); ax[2].set(title="Mean daily profile (test, ensemble)", xlabel="hour", ylabel="Wh/m²/5min")
ax[2].legend(["actual", "predicted"])
plt.tight_layout(); plt.savefig(out / "fig_surrogate_performance.png", dpi=130)
json.dump(dict(n_train=len(tr), n_val=len(va), n_test=len(te)), open(out / "surrogate_split.json", "w"))
