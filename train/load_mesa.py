import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import shelve
import random

from train.utils import random_crop, random_flip
from sleep_classification.preprocess import preprocess_data
from sleep_classification.feature_engineering import (
    get_heart_feature,
)


shelve_path = "data/mesa/shelve"


def get_mesa_dict():
    heart_rate_dir = "data/mesa/heart_rate"
    psg_dir = "data/mesa/psg"
    data_dict = {}
    for i in tqdm(range(10_000)):
        data_id = str(i).zfill(4)
        heart_rate_path = os.path.join(heart_rate_dir, data_id + ".pkl")
        psg_path = os.path.join(psg_dir, data_id + ".pkl")
        if os.path.exists(heart_rate_path) and os.path.exists(psg_path):
            patient_dict = {}
            data_dict[data_id] = patient_dict

            hr_df = pd.read_pickle(os.path.join(heart_rate_dir, data_id + ".pkl"))
            hr_df.index = [i * 10 ** 9 for i in range(len(hr_df))]
            patient_dict["hr_df"] = trim_zeros(hr_df)
            label_df = pd.read_pickle(os.path.join(psg_dir, data_id + ".pkl"))
            label_df.index = [i * 30 * 10 ** 9 for i in range(len(label_df))]
            patient_dict["label_df"] = trim_zeros(label_df)
    return data_dict


def trim_zeros(data):
    start_index = (data.iloc[:, 0] != 0).argmax()
    end_index = len(data) - (data.iloc[:, 0][::-1] != 0).argmax()
    return data.iloc[start_index:end_index]


def get_batch(data_dict, features, batch_size, sequence_len, train):
    supported_features = ["hr", "heart_feature"]
    input_dict = {}
    if "activity_feature" in features:
        input_dict["activity_feature"] = -np.ones((batch_size, sequence_len, 1))

    features = [f for f in features if f in supported_features]
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

    inputs = [np.stack(x, axis=0) for x in zip(*[d.values() for d in batch])]
    inputs = dict(zip(features, inputs))
    input_dict.update(inputs)
    return input_dict, labels


def load_mesa():
    with shelve.open(shelve_path, "r") as shelf:
        d = dict(shelf)
    return d


class MesaDataLoader(tf.keras.utils.Sequence):
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
        inputs, l = get_batch(
            self.data_dict,
            self.features,
            self.batch_size,
            self.sequence_len,
            self.train,
        )
        if "hr" in inputs:
            inputs["hr"] /= 100
        return (
            tuple([inputs[f] for f in self.features]),
            l,
            np.ones((self.batch_size, self.sequence_len, 1)),
        )


def add_features(data_dict):
    for patient_id, data in tqdm(data_dict.items()):
        data["heart_feature"] = get_heart_feature(data["hr"])


if __name__ == "__main__":
    data_dict = load_mesa()
    list(data_dict.values())[0].keys()
    data_dict = get_mesa_dict()

    for data_id, data in tqdm(data_dict.items()):
        hr_df = data["hr_df"]
        label_df = data["label_df"]
        hr_arr, label_arr = preprocess_data(hr_df=hr_df, acc_df=None, label_df=label_df)
        data["hr"] = hr_arr
        data["label"] = label_arr

    with shelve.open(shelve_path, "c") as shelf:
        shelf.update(data_dict)

    add_features(data_dict)

    get_batch(data_dict, ["hr"], 10, True)
    dl = MesaDataLoader(data_dict, ["hr"], 10, 10, True)
    dl.__getitem__(100)
