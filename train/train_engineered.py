from sklearn import metrics
import numpy as np
from tqdm import tqdm
import random
from tensorflow.keras import layers as L
from train.load_physio import get_patient_ids, load_physio, PhysioDataLoader
from train.utils import hybrid_pooling
import tensorflow as tf
from train.load_dalia import load_dalia
from train.load_stress import load_stress
from train.load_mondy import load_mondy
from train.load_mesa import load_mesa

dalia_dict = load_dalia()
stress_dict = load_stress()
mondy_dict = load_mondy()
mesa_dict = load_mesa()
physio_dict = load_physio()

sample = list(stress_dict.values())[0]
np.sqrt((sample["acc_df"] ** 2).sum(axis=1))
len(dalia_dict)
len(stress_dict)
len(mondy_dict)
len(physio_dict)
len(mesa_dict)


def create_model():
    hr_in = L.Input(shape=(None, 1))
    acc_in = L.Input(shape=(None, 1))

    acc = acc_in
    hr = hr_in

    x = L.Concatenate(axis=2)([hr_in, acc_in])
    # x=hr_in
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
    # model = tf.keras.models.Model(inputs=(hr_in), outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


def save_data(data_dict):
    with shelve.open("data/processed/data_dict") as d:
        for k, v in data_dict.items():
            d[k] = v


def load_data():
    with shelve.open("data/processed/data_dict") as d:
        data_dict = dict(d)
    return data_dict


if __name__ == "__main__":
    result_dict = {}
    create_model().summary()

    data_dict = load_physio()
    feature_dict = data_dict.copy()
    list(feature_dict.values())[0]
    # import shelve

    # save_data(feature_dict)
    # feature_dict = load_data()

    features = ["heart_feature", "activity_feature"]
    # features = ["heart_feature"]
    aucs = []
    for test_patient in feature_dict:
        print(f"\n\ntraining model for patient {test_patient}")
        train_dict = feature_dict.copy()
        test_dict = {test_patient: train_dict.pop(test_patient)}
        assert test_patient not in train_dict
        dl_train = PhysioDataLoader(
            train_dict,
            features,
            batch_size=100,
            n_batches=100,
            sequence_len=100,
            train=True,
        )
        dl_test = PhysioDataLoader(
            test_dict,
            features,
            batch_size=1,
            sequence_len=1,
            n_batches=1,
            train=False,
        )
        for i in range(1):
            model = create_model()
            predictions = []
            for i in range(10):
                model.fit(dl_train, epochs=1, verbose=0)
                model.evaluate(dl_test, verbose=1)

                prediction = 1 - model.predict(dl_test, verbose=0)[0, :, 0]

                _, labels, weights = next(iter(dl_test))
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
        result_dict[test_patient] = {"labels": labels, "predictions": prediction}
        model.save(f"models/{test_patient}.h5")
        break

import plotly.express as pe

pe.line(labels)
pe.line(prediction)
pe.line(test_dict[test_patient]["heart_feature"])
pe.line(test_dict[test_patient]["hr_df"])
pe.line(md["hr_df"])
pe.line(md["acc_df"][:10000])

pe.line(md["heart_feature"][:1000])
pe.line(test_dict[test_patient]["activity_feature"])
pe.line(md["activity_feature"][:1000])
pe.line(mp[:, 0, 0][:1000])
list(mondy_dict.keys())[4]
md = mondy_dict["18ad66be3a0-165a2619-01ef-455a-ac6d-dd109bdaec39"]  # looks good!
md = mondy_dict["18b6629a6da-830ea28f-f989-4c06-b5c7-26f699da3c93"]
md = mondy_dict["18b669e6c81-b1a6fcac-cc50-4e65-83b2-428c832d45ed"]
md = mondy_dict['18b8fdd2eb3-bfb83b6e-27e1-464c-a560-bdd25ea5131c']
pe.line(md["heart_feature"])
pe.line(md["activity_feature"])
mp.shape
mp = model.predict(
    (np.expand_dims(md["heart_feature"], 0), np.expand_dims(md["activity_feature"], 0))
)
np.mean(mp[0, :, 0] > 0.9)
pe.line(md["hr_df"])
pe.line(md["activity_feature"])
pe.line(mp[0, :, 0])
