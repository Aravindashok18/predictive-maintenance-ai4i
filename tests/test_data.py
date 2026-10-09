import numpy as np
import pandas as pd

from src import data


def tiny():
    return pd.DataFrame({
        "product_id": ["L1", "M2", "H3"],
        "type": ["L", "M", "H"],
        "air_temp_k": [298.0, 300.0, 299.0],
        "process_temp_k": [308.0, 310.5, 309.0],
        "rpm": [1500, 1200, 2000],
        "torque_nm": [40.0, 60.0, 20.0],
        "tool_wear_min": [10, 200, 0],
        "failure": [0, 1, 0],
        "TWF": [0, 0, 0], "HDF": [0, 1, 0], "PWF": [0, 0, 0], "OSF": [0, 0, 0], "RNF": [0, 0, 0],
    })


def test_power_is_torque_times_angular_speed():
    out = data.add_features(tiny())
    expected = 40.0 * 1500 * 2 * np.pi / 60
    assert abs(out.loc[0, "power_w"] - expected) < 1e-9


def test_temperature_difference_and_wear_torque():
    out = data.add_features(tiny())
    assert out["temp_diff_k"].tolist() == [10.0, 10.5, 10.0]
    assert out["wear_torque"].tolist() == [400.0, 12000.0, 0.0]


def test_type_dummies_are_one_hot():
    out = data.add_features(tiny())
    assert (out[["type_L", "type_M", "type_H"]].sum(axis=1) == 1).all()


def test_no_feature_set_contains_label_columns():
    forbidden = set(data.MODE_COLS) | {data.TARGET}
    for name, cols in data.FEATURE_SETS.items():
        assert not forbidden & set(cols), name


def test_add_features_does_not_modify_input():
    df = tiny()
    before = df.copy()
    data.add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_real_dataset_shape_and_audit():
    df = data.load()
    a = data.audit(df)
    assert a["rows"] == 10000
    assert a["failures"] == 339
    assert a["missing_values"] == 0
    assert set(df["type"]) == {"L", "M", "H"}
