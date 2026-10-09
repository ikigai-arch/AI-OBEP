"""Loading and light cleaning of the ROBOD dataset (Tekler et al., 2022)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Table 1 of the ROBOD paper.
ROOM_INFO = {
    1: dict(name="Lecture room (large)", hvac="FCU", area=118.6, seats=40),
    2: dict(name="Lecture room (small)", hvac="FCU", area=53.7, seats=40),
    3: dict(name="Office (admin staff)", hvac="AHU/VAV", area=98.4, seats=15),
    4: dict(name="Office (researchers)", hvac="AHU/VAV", area=141.9, seats=25),
    5: dict(name="Library", hvac="AHU/VAV", area=182.8, seats=36),
}
OFFICE_ROOMS = (3, 4)

# The 7 input features used by the surrogate in the ML paper (Sec. 3.2).
FEATURES = [
    "air_temperature",
    "indoor_relative_humidity",
    "occupant_count",
    "dry_bulb_temp",
    "outdoor_relative_humidity",
    "wind_speed",
    "global_horizontal_solar_radiation",
]
# HVAC-related energy end-uses forming the target (Sec. 3.2): ceiling fans,
# chilled water, and the AHU (rooms 3-5) / FCU (rooms 1-2) fan.
ENERGY_COMPONENTS = ["ceiling_fan_energy", "chilled_water_energy", "fan_energy"]
OPERATING_HOURS = (8.5, 18.67)  # 08:30-18:40, AHU schedule of rooms 3-5


def load_room(room: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one ``combined_Room<k>.csv``; strip units from column names.

    Adds: ``ts`` (naive local time), ``day``, ``hour``, ``room``, ``fan_energy``
    (AHU or FCU fan, whichever the room has) and ``hvac_energy_wh_m2`` -
    the surrogate target in Wh per m2 per 5 minutes.
    """
    path = Path(raw_dir) / f"combined_Room{room}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - unzip SupplementaryData.zip into {raw_dir}"
        )
    df = pd.read_csv(path)
    df.columns = [c.split(" [")[0] for c in df.columns]
    df["ts"] = pd.to_datetime(df["timestamp"].str[:16])  # drop the +08:00 suffix
    df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["room"] = room
    df["fan_energy"] = df["ahu_fan_energy"] if "ahu_fan_energy" in df else df["fcu_fan_energy"]
    kwh = df[ENERGY_COMPONENTS].sum(axis=1, min_count=len(ENERGY_COMPONENTS))
    df["hvac_energy_wh_m2"] = kwh * 1000.0 / ROOM_INFO[room]["area"]
    df["is_operating"] = (df["hour"] >= OPERATING_HOURS[0]) & (df["hour"] < OPERATING_HOURS[1])
    return df


def load_rooms(rooms=range(1, 6), raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.concat([load_room(r, raw_dir) for r in rooms], ignore_index=True)


def split_by_day(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed: int = 0):
    """Random split at the *day* level (per room) so that neighbouring 5-min
    samples of one day never end up in both train and test (no leakage)."""
    keys = np.array(sorted((df["room"].astype(str) + "_" + df["day"]).unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_tr, n_va = int(fracs[0] * n), int(fracs[1] * n)
    which = dict.fromkeys(keys[:n_tr], "train")
    which.update(dict.fromkeys(keys[n_tr:n_tr + n_va], "val"))
    which.update(dict.fromkeys(keys[n_tr + n_va:], "test"))
    k = df["room"].astype(str) + "_" + df["day"]
    return k.map(which).to_numpy()
