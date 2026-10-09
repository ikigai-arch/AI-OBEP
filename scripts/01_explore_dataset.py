"""Explore ROBOD: what is in it, how rooms differ, and how occupancy relates to energy.

Writes figures + tables to outputs/eda/ and prints a short text summary.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiobep.data import FEATURES, ROOM_INFO, load_rooms  # noqa: E402

out = ROOT / "outputs" / "eda"
out.mkdir(parents=True, exist_ok=True)
df = load_rooms()
df["rname"] = df.room.map(lambda r: f"R{r} {ROOM_INFO[r]['name']}")

# 1. Overview table ---------------------------------------------------------
g = df.groupby("room")
ov = pd.DataFrame({
    "name": [ROOM_INFO[r]["name"] for r in g.groups],
    "hvac": [ROOM_INFO[r]["hvac"] for r in g.groups],
    "area_m2": [ROOM_INFO[r]["area"] for r in g.groups],
    "days": g.day.nunique(), "rows": g.size(),
    # share of NaN among columns the room actually has (excludes e.g. AHU columns of FCU rooms)
    "missing_%": (g.apply(lambda d: d.drop(columns=["rname"]).dropna(axis=1, how="all").isna().mean().mean()) * 100).round(2),
    "occ_mean": g.occupant_count.mean().round(2), "occ_max": g.occupant_count.max(),
    "present_%": (g.occupant_presence.mean() * 100).round(1),
    "T_mean_C": g.air_temperature.mean().round(2), "RH_mean_%": g.indoor_relative_humidity.mean().round(1),
    "hvac_kWh_per_day": (g.apply(lambda d: d.hvac_energy_wh_m2.sum() * ROOM_INFO[d.name]["area"] / 1000 / d.day.nunique())).round(1),
    "hvac_Wh_m2_per_day": (g.hvac_energy_wh_m2.sum() / g.day.nunique()).round(1),
})
ov.to_csv(out / "room_overview.csv")
print(ov.to_string())

# 2. Mean daily profile: occupancy vs HVAC energy ---------------------------
fig, ax = plt.subplots(2, 5, figsize=(20, 6), sharex=True)
for i, r in enumerate(range(1, 6)):
    d = df[df.room == r]
    prof = d.groupby(d.ts.dt.hour)[["occupant_count", "hvac_energy_wh_m2", "air_temperature"]].mean()
    ax[0, i].plot(prof.index, prof.occupant_count, c="tab:green"); ax[0, i].set_title(f"R{r} {ROOM_INFO[r]['name']}")
    ax[1, i].plot(prof.index, prof.hvac_energy_wh_m2, c="tab:red")
    ax[1, i].set_xlabel("hour of day")
    ax[1, i].axvspan(8.5, 18.67, color="grey", alpha=.12)
ax[0, 0].set_ylabel("mean occupants"); ax[1, 0].set_ylabel("HVAC Wh/m²/5min")
fig.suptitle("Daily profiles (grey band = HVAC schedule 08:30-18:40): occupancy follows people, HVAC follows the schedule")
plt.tight_layout(); plt.savefig(out / "daily_profiles.png", dpi=120); plt.close()

# 3. Does energy respond to occupancy *within* operating hours? -------------
op = df[df.is_operating].dropna(subset=["hvac_energy_wh_m2", "occupant_count"])
fig, ax = plt.subplots(1, 5, figsize=(20, 3.8), sharey=False)
rows = []
for i, r in enumerate(range(1, 6)):
    d = op[op.room == r]
    bins = pd.cut(d.occupant_count, [-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", ">10"])
    m = d.groupby(bins, observed=True).hvac_energy_wh_m2.agg(["mean", "count"])
    ax[i].bar(m.index.astype(str), m["mean"]); ax[i].set_title(f"R{r}: HVAC vs occupants"); ax[i].set_xlabel("occupants")
    rows.append(dict(room=r, spearman_occ_vs_energy=d.occupant_count.corr(d.hvac_energy_wh_m2, method="spearman"),
                     spearman_outdoorT_vs_energy=d.dry_bulb_temp.corr(d.hvac_energy_wh_m2, method="spearman"),
                     spearman_solar_vs_energy=d.global_horizontal_solar_radiation.corr(d.hvac_energy_wh_m2, method="spearman")))
ax[0].set_ylabel("mean HVAC Wh/m²/5min")
plt.tight_layout(); plt.savefig(out / "energy_vs_occupancy_operating_hours.png", dpi=120); plt.close()
corr = pd.DataFrame(rows).round(2)
corr.to_csv(out / "operating_hours_correlations.csv", index=False)
print("\nSpearman correlation with HVAC energy, operating hours only:\n", corr.to_string(index=False))

# 4. Feature correlation heat-map for the 7 surrogate inputs (offices) -----
offc = df[df.room.isin([3, 4])].dropna(subset=FEATURES + ["hvac_energy_wh_m2"])
C = offc[FEATURES + ["hvac_energy_wh_m2"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(C))); ax.set_yticklabels(C.columns, fontsize=8)
for i in range(len(C)):
    for j in range(len(C)):
        ax.text(j, i, f"{C.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im); ax.set_title("Spearman correlation, office rooms 3+4"); plt.tight_layout()
plt.savefig(out / "feature_correlation.png", dpi=120); plt.close()

# 5. Missing data & occupancy-from-sensors check ----------------------------
miss = df.groupby("room").apply(lambda d: d.drop(columns=["rname"]).isna().mean() * 100).T
miss = miss[(miss > 0).any(axis=1)].round(2)
miss.to_csv(out / "missing_pct_by_room.csv")
print("\nColumns with missing data (%):\n", miss.to_string())

sens = df.dropna(subset=["indoor_co2", "wifi_connected_devices", "occupant_count"])
rows = [dict(room=r, corr_co2=d.indoor_co2.corr(d.occupant_count), corr_wifi=d.wifi_connected_devices.corr(d.occupant_count),
             corr_sound=d.sound_pressure_level.corr(d.occupant_count)) for r, d in sens.groupby("room")]
sc = pd.DataFrame(rows).round(2)
sc.to_csv(out / "occupancy_proxy_correlations.csv", index=False)
print("\nPearson corr. of cheap sensors with true occupant count (matters for the edge layer):\n", sc.to_string(index=False))

# 6. Time-series snapshot of one week (room 4) ------------------------------
d = df[df.room == 4]
wk = d[(d.ts >= d.ts.min() + pd.Timedelta(days=21)) & (d.ts < d.ts.min() + pd.Timedelta(days=28))]
fig, ax = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
for a, (c, lab) in zip(ax, [("occupant_count", "occupants"), ("air_temperature", "indoor T [°C]"),
                            ("hvac_energy_wh_m2", "HVAC Wh/m²/5min"), ("dry_bulb_temp", "outdoor T [°C]")]):
    a.plot(wk.ts, wk[c], lw=.8); a.set_ylabel(lab)
ax[0].set_title("Room 4 (researchers' office) - one week")
plt.tight_layout(); plt.savefig(out / "room4_week.png", dpi=120); plt.close()
print(f"\nfigures written to {out}")
