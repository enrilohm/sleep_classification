from sklearn import metrics
import numpy as np
from sleep_classification.preprocess import preprocess_data
from sleep_classification.feature_engineering import (
    get_heart_feature,
    get_activity_counts,
)
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import layers as L
from train.data_load import read_data, get_patient_ids


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
    return data_dict


def random_crop(data, features, crop_size):
    label_arr = data["label"].copy()
    cropped = {}
    y_boundary = label_arr.shape[0] - crop_size
    y_start = np.random.randint(0, y_boundary)
    for feature in features:
        feature_array = data[feature]
        assert len(feature_array) % len(label_arr) == 0, feature
        r = len(feature_array) // len(label_arr)
        cropped_feature = feature_array[y_start * r : (y_start + crop_size) * r, :]
        cropped[feature] = cropped_feature
    cropped["label"] = label_arr[y_start : y_start + crop_size, :]
    return cropped


def random_flip(data, prob=0.5):
    if random.random() <= prob:
        data = {k: np.flip(v, 0) for k, v in data.items()}


def get_batch(data_dict, features, batch_size, train):
    batch = []
    for sample in range(batch_size):
        patient_id = random.choice(list(data_dict.keys()))
        data = {
            k: v for k, v in data_dict[patient_id].items() if k in features + ["label"]
        }
        if train:
            data = random_crop(data, features, 100)
            random_flip(data)
        batch.append(data)
    labels = [d.pop("label") for d in batch]
    labels = np.stack(labels, axis=0)
    weights = (labels != -1).astype(int)
    labels += labels == -1

    inputs = [np.stack(x, axis=0) for x in zip(*[d.values() for d in batch])]
    inputs = dict(zip(features, inputs))
    return inputs, labels, weights


def hybrid_pooling(x, kernel, strides):
    max = L.MaxPooling1D(kernel, strides=strides)(x)
    avg = L.AveragePooling1D(kernel, strides=strides)(x)
    result = L.Concatenate(axis=2)([max, avg])
    return result


def create_model():
    hr_in = L.Input(shape=(None, 1))
    acc_in = L.Input(shape=(None, 1))

    acc = acc_in
    hr = hr_in

    x = L.Concatenate(axis=2)([hr_in, acc_in])
    x = L.Conv1D(8, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(128, 5, padding="same")(x)
    x = L.Dropout(0.1)(x)
    x = L.Activation("relu")(x)
    x = L.Dense(6, activation="softmax")(x)

    output = x

    model = tf.keras.models.Model(inputs=(hr_in, acc_in), outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


class StandardDataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_dict, features, batch_size, steps, train):
        self.steps = steps
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.train = train
        self.features = features

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        f, l, w = get_batch(self.data_dict, self.features, self.batch_size, self.train)
        f["activity_feature"] = np.log(1 + f["activity_feature"])
        return list(f.values()), l, w


def save_data(data_dict):
    with shelve.open("data/processed/data_dict") as d:
        for k, v in data_dict.items():
            d[k] = v


def load_data():
    with shelve.open("data/processed/data_dict") as d:
        data_dict = dict(d)
    return data_dict


def add_features(data_dict):
    for patient_id, data in tqdm(data_dict.items()):
        data["heart_feature"] = get_heart_feature(data["hr"])
    for patient_id, data in tqdm(data_dict.items()):
        data["activity_feature"] = get_activity_counts(data["acc"])


if __name__ == "__main__":
    create_model().summary()

    data_dict = read_all_data()
    feature_dict = data_dict.copy()
    add_features(feature_dict)
    # import shelve

    # save_data(feature_dict)
    # feature_dict = load_data()

    aucs = []
    for test_patient in feature_dict:
        print(f"\n\ntraining model for patient {test_patient}")
        train_dict = feature_dict.copy()
        test_dict = {test_patient: train_dict.pop(test_patient)}
        dl_train = StandardDataLoader(
            train_dict,
            ["heart_feature", "activity_feature"],
            batch_size=100,
            steps=100,
            train=True,
        )
        dl_test = StandardDataLoader(
            test_dict,
            ["heart_feature", "activity_feature"],
            batch_size=1,
            steps=1,
            train=False,
        )
        for i in range(1):
            model = create_model()
            predictions = []
            for i in range(20):
                model.fit(dl_train, epochs=1, verbose=0)
                model.evaluate(dl_test, verbose=0)

                prediction = 1 - model.predict(dl_test, verbose=0)[0, :, 0]

                (_, _), labels, weights = next(iter(dl_test))
                labels = (labels[0, :, 0] > 0).astype(int)
                weights = weights[0, :, 0]

                labels = labels[np.where(weights)]
                prediction = prediction[np.where(weights)]
                auc = metrics.roc_auc_score(labels, prediction)
                print(f"\tepoch: {i} --- test-auc: {str(auc)}")
            predictions.append(prediction)
        predictions = sum(predictions) / len(predictions)
        auc = metrics.roc_auc_score(labels, predictions)
        print("new auc " + str(auc))
        aucs.append(auc)
        print("new mean auc: " + str(np.mean(aucs)))
        model.save(f"models/{test_patient}.h5")
