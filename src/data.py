"""Loading and feature engineering for the AI4I 2020 predictive maintenance data."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"

TARGET = "failure"
# The five failure-mode flags are what the target is built from. They are
# never used as model inputs: doing so would be label leakage.
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RENAME = {
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": TARGET,
}

BASE_FEATURES = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
TYPE_FEATURES = ["type_L", "type_M", "type_H"]
POWER_FEATURES = ["power_w", "temp_diff_k"]
WEAR_FEATURES = ["wear_torque"]

FEATURE_SETS = {
    "raw": BASE_FEATURES + TYPE_FEATURES,
    "engineered": BASE_FEATURES + TYPE_FEATURES + POWER_FEATURES,
    "engineered+wear": BASE_FEATURES + TYPE_FEATURES + POWER_FEATURES + WEAR_FEATURES,
}


def load(path=DATA_PATH):
    """Read the CSV and give the columns short names."""
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=RENAME)
    return df.drop(columns=["UDI"])


def add_features(df):
    """Return a copy with the derived columns used by the models.

    power_w      mechanical power in watts: torque (Nm) * angular speed (rad/s)
    temp_diff_k  process temperature minus air temperature
    wear_torque  tool wear * torque, the quantity behind overstrain failures
    """
    out = df.copy()
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    for t in ("L", "M", "H"):
        out[f"type_{t}"] = (out["type"] == t).astype(int)
    return out


def audit(df):
    """Counts that describe how consistent the failure labels are."""
    any_mode = df[MODE_COLS].sum(axis=1) > 0
    return {
        "rows": int(len(df)),
        "failures": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_product_ids": int(df["product_id"].duplicated().sum()),
        "failure_without_any_mode_flag": int(((df[TARGET] == 1) & ~any_mode).sum()),
        "mode_flag_without_failure": int(((df[TARGET] == 0) & any_mode).sum()),
        "mode_counts": {c: int(df[c].sum()) for c in MODE_COLS},
    }
