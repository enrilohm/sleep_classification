import pandas as pd
from glob import glob
import os

data_dir = "data/physionet.org/files/sleep-accel/1.0.0"


def load_heart_rate(patient_id):
    hr_df = pd.read_csv(
        f"{data_dir}/heart_rate/{patient_id}_heartrate.txt", header=None
    )
    hr_df = hr_df.drop_duplicates(keep="first").sort_values(by=[0]).set_index(0)
    hr_df.rename(columns={1: "hr"}, inplace=True)
    assert abs(len(hr_df) - hr_df.index.nunique()) <= 1
    return hr_df


def load_acceleration(patient_id):
    acc_df = pd.read_csv(
        f"{data_dir}/motion/{patient_id}_acceleration.txt", sep=" ", header=None
    )
    acc_df = acc_df.drop_duplicates(keep="first").sort_values(by=[0]).set_index(0)
    acc_df.columns = ["acc_x", "acc_y", "acc_z"]
    assert len(acc_df) == acc_df.index.nunique()
    return acc_df


def load_labels(patient_id):
    label_df = pd.read_csv(
        f"{data_dir}/labels/{patient_id}_labeled_sleep.txt", sep=" ", header=None
    )
    label_df = label_df.drop_duplicates(keep="first").sort_values(by=[0]).set_index(0)
    assert len(label_df) == label_df.index.nunique()
    return label_df


def get_valid_epochs(hr_df, acc_df, label_df):
    """Using valid epoch logic for comparability with https://github.com/ojwalch/sleep_classifiers"""
    valid_data = (
        label_df.index.isin(acc_df.index.map(lambda x: x - x % 30))
        & label_df.index.isin(hr_df.index.map(lambda x: x - x % 30))
        & (label_df[1] != -1)
    )
    valid_data.iloc[0] = False
    valid_data.iloc[-1] = False
    return valid_data


def read_data(patient_id):
    hr_df = load_heart_rate(patient_id)
    acc_df = load_acceleration(patient_id)
    label_df = load_labels(patient_id)

    assert hr_df.index.min() < label_df.index.min()
    assert hr_df.index.max() > label_df.index.max()
    assert acc_df.index.min() < label_df.index.min()
    assert acc_df.index.max() > label_df.index.max()

    label_df["valid_data"] = get_valid_epochs(hr_df, acc_df, label_df)

    adjust_adjust_units(hr_df, acc_df, label_df)
    return hr_df, acc_df, label_df


def adjust_adjust_units(hr_df, acc_df, label_df):
    """
    index from seconds to nano seconds for compatibility with pd.to_datetime
    convert acceleration data to SI units
    """
    hr_df.index *= 10 ** 9
    acc_df.index *= 10 ** 9
    label_df.index *= 10 ** 9
    acc_df *= 9.81


def get_patient_ids():
    return [os.path.basename(x).split("_")[0] for x in glob(f"{data_dir}/labels/*.txt")]
