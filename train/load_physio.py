import shelve
from tqdm import tqdm
import pandas as pd
from glob import glob
import os
from sleep_classification.preprocess import preprocess_data
from sleep_classification.feature_engineering import (
    get_heart_feature,
    get_activity_counts,
)
import random
from train.utils import random_flip, random_crop
import numpy as np
import tensorflow as tf

data_dir = "data/physionet.org/files/sleep-accel/1.0.0"
shelve_path = "data/physionet.org/shelve"


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


def trim(data, value=0):
    start_index = (data.iloc[:, 0] != value).argmax()
    end_index = len(data) - (data.iloc[:, 0][::-1] != value).argmax()
    return data.iloc[start_index:end_index]


patient_id = "844359"


def read_data(patient_id):
    hr_df = load_heart_rate(patient_id)
    acc_df = load_acceleration(patient_id)
    label_df = load_labels(patient_id)
    label_df[label_df[1] != -1]
    label_df = trim(label_df, value=-1)
    if label_df.iloc[0, 0] == 0:
        start = label_df.index[0]
        label_df = pd.concat(
            (pd.DataFrame(index=range(start - 240*30, start, 30), data={1: -2}), label_df)
        )
    if label_df.iloc[-1, 0] == 0:
        start = label_df.index[-1]
        label_df = pd.concat(
            (label_df,pd.DataFrame(index=range(start+30, start + 240*30, 30), data={1: -2}))
        )
    # assert hr_df.index.min() < label_df.index.min()
    # assert hr_df.index.max() > label_df.index.max()
    # assert acc_df.index.min() < label_df.index.min()
    # assert acc_df.index.max() > label_df.index.max()

    # ex.line(label_df)
    # ex.line(hr_df)
    # ex.line(acc_df)
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


def read_all_data():
    print("preprocessing data:")
    patient_ids = get_patient_ids()
    data_dict = {}
    for patient_id in tqdm(patient_ids):
        print(f"processing {patient_id}")
        data_dict[patient_id] = {}

        hr_df, acc_df, label_df = read_data(patient_id)
        label_df.loc[~label_df.valid_data, 1] = -1.0
        label_df.drop(columns="valid_data", inplace=True)
        hr_arr, acc_arr, label_arr = preprocess_data(hr_df, acc_df, label_df)
        for key, value in zip(["hr", "acc", "label"], [hr_arr, acc_arr, label_arr]):
            data_dict[patient_id][key] = value
        for key, value in zip(
            ["hr_df", "acc_df", "label_df"], [hr_df, acc_df, label_df]
        ):
            data_dict[patient_id][key] = value
    return data_dict


def load_physio():
    with shelve.open(shelve_path, "r") as shelf:
        d = dict(shelf)
    return d


def add_features(data_dict):
    for patient_id, data in tqdm(data_dict.items()):
        data["heart_feature"] = get_heart_feature(data["hr"])
    for patient_id, data in tqdm(data_dict.items()):
        data["activity_feature"] = get_activity_counts(data["acc"])


def get_batch(data_dict, features, batch_size, sequence_len, train):
    batch = []
    for sample in range(batch_size):
        patient_id = random.choice(list(data_dict.keys()))
        data = {
            k: v for k, v in data_dict[patient_id].items() if k in features + ["label"]
        }
        if train:
            data = random_crop(data, features, sequence_len)
            random_flip(data)
        batch.append(data)
    labels = [d.pop("label") for d in batch]
    labels = np.stack(labels, axis=0)
    weights = (labels != -1).astype(int)
    labels += labels == -1
    labels += (labels == -2) * 2

    inputs = [np.stack(x, axis=0) for x in zip(*[d.values() for d in batch])]
    inputs = dict(zip(features, inputs))
    return inputs, labels, weights

class PhysioDataLoader(tf.keras.utils.Sequence):
    def __init__(
        self, data_dict, features, batch_size, n_batches, sequence_len=100, train=True
    ):
        self.n_batches = n_batches
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.train = train
        self.features = features
        self.sequence_len = sequence_len

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        f, l, w = get_batch(self.data_dict, self.features, self.batch_size, self.sequence_len, self.train)
        f["activity_feature"] = np.log(1 + f["activity_feature"])
        return list(f.values()), l, w

if __name__ == "__main__":
    data_dict = read_all_data()
    add_features(data_dict)
    with shelve.open(shelve_path, "c") as shelf:
        shelf.update(data_dict)

    get_batch(data_dict, ["hr"], 10, True)
    dl = PhysioDataLoader(data_dict, ["hr"], 10, 10, True)
    dl.__getitem__(100)

    data_dict = load_physio()
    data_dict["844359"]["acc_df"]

    acc = list(data_dict.values())[1]["acc"] / 9.81
    np.abs(acc).max(axis=0)
    np.abs(acc).mean(axis=0)
    np.abs(acc).max(axis=0) / 9.81

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    moving_average(np.abs(acc[:, 1])).max()


    data_dict.keys()
    import plotly.express as ex
    p="8173033"
    ex.line(data_dict[p]["label_df"])
    ex.line(data_dict[p]["hr_df"])
    ex.line(data_dict[p]["acc_df"])
    data_dict["844359"]["label_df"]
