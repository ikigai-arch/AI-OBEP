"""ROBOD loading and preprocessing for the room-level HVAC surrogate.

Lyu et al. (2025, Sec. 3.4) train on ROBOD *office-1* (= ROBOD Room 3, admin office,
98.4 m2) with the 7 features below, and remove outliers by averaging adjacent
time points. This module reproduces that, and adds the hourly / time-of-day
variant used for the deployed surrogate.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("ROBOD_DIR", ROOT / "data" / "raw"))

ROOM = 3                 # ROBOD Room 3 = "office-1" in Lyu et al.
ROOM_AREA = 98.4         # m2 (ROBOD Table 1)
ENERGY_COMPONENTS = ["ceiling_fan_energy", "chilled_water_energy", "ahu_fan_energy"]
OPERATING_HOURS = (8.5, 18.67)  # AHU schedule 08:30-18:40

# The 7 surrogate inputs of Lyu et al. (Sec. 3.2).
FEATURES_PAPER = [
    "air_temperature",
    "indoor_relative_humidity",
    "occupant_count",
    "dry_bulb_temp",
    "outdoor_relative_humidity",
    "wind_speed",
    "global_horizontal_solar_radiation",
]
# Deployed surrogate: paper features + time of day (cyclic) + HVAC schedule flag.
FEATURES = FEATURES_PAPER + ["hour_sin", "hour_cos", "hvac_schedule"]
TARGET = "hvac_w_m2"     # mean HVAC power density [W/m2] = kWh per hour * 1000 / area


def load_room(room: int = ROOM, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = Path(raw_dir) / f"combined_Room{room}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/00_get_data.sh first")
    df = pd.read_csv(path)
    df.columns = [c.split(" [")[0] for c in df.columns]
    df["ts"] = pd.to_datetime(df["timestamp"].str[:16])
    df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["energy_kwh"] = df[ENERGY_COMPONENTS].sum(axis=1, min_count=len(ENERGY_COMPONENTS))
    return df.dropna(subset=FEATURES_PAPER + ["energy_kwh"]).reset_index(drop=True)


def hampel_smooth(s: pd.Series, window: int = 5, n_sigma: float = 3.0, floor_frac: float = 0.25):
    """Outlier removal by averaging adjacent points (Lyu et al. Sec. 3.4).

    A point is an outlier when it deviates from the rolling median by more than
    ``n_sigma`` robust standard deviations (MAD), with a floor of ``floor_frac``
    times the series std so that flat (HVAC-off) stretches are not flagged.
    Outliers are replaced by the mean of their two neighbours.
    Returns (smoothed series, fraction flagged).
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    dev = (s - med).abs()
    sigma = 1.4826 * dev.rolling(window, center=True, min_periods=1).median()
    thr = np.maximum(n_sigma * sigma, floor_frac * s.std())
    out_mask = dev > thr
    neighbours = (s.shift(1) + s.shift(-1)) / 2
    out = s.where(~out_mask, neighbours).fillna(s)
    return out, float(out_mask.mean())


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["hvac_schedule"] = ((df["hour"] >= OPERATING_HOURS[0]) & (df["hour"] < OPERATING_HOURS[1])).astype(int)
    return df


def five_minute_dataset(n_days=None, smooth=True) -> pd.DataFrame:
    """Paper protocol: 5-min rows, first ``n_days`` weekdays, target in kWh per 5 min."""
    df = load_room()
    if n_days is not None:
        keep = sorted(df["day"].unique())[:n_days]
        df = df[df["day"].isin(keep)].reset_index(drop=True)
    df["target_kwh"] = hampel_smooth(df["energy_kwh"])[0] if smooth else df["energy_kwh"]
    return add_time_features(df)


def hourly_dataset(smooth=True) -> pd.DataFrame:
    """Deployed protocol: all weekdays, 5-min data smoothed then averaged per hour.

    Target = mean HVAC power density [W/m2]. Hourly resolution matches the
    steady-state snapshot evaluated by the optimiser and averages out
    chilled-water valve cycling noise.
    """
    df = load_room()
    e = hampel_smooth(df["energy_kwh"])[0] if smooth else df["energy_kwh"]
    df[TARGET] = e * 12 * 1000 / ROOM_AREA          # kWh per 5 min -> W/m2
    hourly = (df.set_index("ts")[FEATURES_PAPER + [TARGET]]
              .resample("60min").mean().dropna().reset_index())
    hourly["day"] = hourly["ts"].dt.strftime("%Y-%m-%d")
    hourly["hour"] = hourly["ts"].dt.hour + 0.5     # mid-point of the hour
    return add_time_features(hourly)


def day_split(df: pd.DataFrame, test_frac: float = 0.17, seed: int = 0):
    """Hold out whole days for testing (no leakage of autocorrelated samples)."""
    days = np.array(sorted(df["day"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(days)
    n_test = max(1, int(round(test_frac * len(days))))
    test_days = set(days[:n_test])
    is_test = df["day"].isin(test_days).to_numpy()
    return ~is_test, is_test
