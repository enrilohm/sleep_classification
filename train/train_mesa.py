from sklearn import metrics
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import layers as L
from train.utils import random_flip, random_crop, hybrid_pooling
import shelve
from train.load_mesa import load_mesa, MesaDataLoader
from train.load_physio import load_physio, PhysioDataLoader

mesa_dict = load_mesa()


def create_model():
    hr_in = L.Input(shape=(None, 1))
    # acc_in = L.Input(shape=(None, 1))
    hr = hr_in
    # acc = acc_in

    # x = L.Concatenate(axis=2)([hr, acc])
    x = hr

    x = L.Conv1D(8, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    # x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    x = L.MaxPooling1D(3)(x)
    x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    # x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    x = L.MaxPooling1D(5)(x)
    x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    # x = L.Conv1D(16, 5, activation="relu", padding="same")(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(128, 5, activation="relu", padding="same")(x)
    # x = L.Conv1D(128, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(128, 5, padding="same")(x)
    x = L.Dropout(0.1)(x)
    x = L.Activation("relu")(x)
    x = L.Dense(6, activation="softmax")(x)

    output = x

    model = tf.keras.models.Model(inputs=(hr_in,), outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


features = ["hr"]
# features = ["heart_feature"]

dl_train_mesa = MesaDataLoader(
    mesa_dict,
    features,
    batch_size=400,
    n_batches=100,
    sequence_len=150,
    train=True,
)
# dl_train_mesa.__getitem__(0)


def get_dataset_from_dl(dl):
    def gen():
        for x in dl:
            yield x

    ds = tf.data.Dataset.from_generator(
        gen, output_types=(((tf.float64, tf.float64), tf.int64, tf.int64))
    )
    return ds


# ds = get_dataset_from_dl(dl_train_mesa)
# len(next(iter(dl_train_mesa)))
# next(iter(ds))
# base_model = model


# base_model.summary()
# base_model.evaluate(test_dl)
# base_model.evaluate(dl_train_mesa)
physio_dict = load_physio()


def get_predictions_and_labels(test_patient):
    train_dict = physio_dict.copy()
    test_dict = {test_patient: train_dict.pop(test_patient)}
    dl_test = PhysioDataLoader(
        test_dict,
        features,
        batch_size=1,
        n_batches=1,
        train=False,
    )
    prediction = 1 - base_model.predict(dl_test, verbose=0)[0, :, 0]
    (_), labels, weights = next(iter(dl_test))
    labels = (labels[0, :, 0] > 0).astype(int)
    weights = weights[0, :, 0]
    labels = labels[np.where(weights)]
    prediction = prediction[np.where(weights)]
    return prediction, labels


test_dl = PhysioDataLoader(
    physio_dict,
    features,
    batch_size=1,
    n_batches=1,
    train=False,
)

len(test_dl)


base_model = create_model()
tf.keras.backend.set_value(base_model.optimizer.learning_rate, 0.0001)
for i in range(100):
    base_model.fit(
        dl_train_mesa,
        validation_data=test_dl,
        epochs=1,
    )
    aucs = []
    for test_patient in physio_dict:
        prediction, labels = get_predictions_and_labels(test_patient)
        auc = metrics.roc_auc_score(labels, prediction)
        aucs.append(auc)
        break
    auc = np.mean(aucs)
    print(f"\ttest-auc: {str(auc)}")

list(range(0, 30, 3))
import plotly.graph_objects as go


def plot_results(hr, prediction, labels):
    fig = go.Figure()

    # Add first trace
    fig.add_trace(
        go.Scatter(
            x=list(range(0, len(prediction) * 30, 30)),
            y=prediction,
            mode="lines+markers",
            name="Series 1",
            marker=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(0, len(labels) * 30, 30)),
            y=labels,
            mode="lines+markers",
            name="Series 2",
            marker=dict(color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(hr) * 30)),
            y=hr[:, 0] / 100,
            mode="lines+markers",
            name="Series 3",
            marker=dict(color="green"),
        )
    )
    fig.show()


test_patients = list(physio_dict)
test_patient = test_patients[5]
hr = physio_dict[test_patient]["hr"]
prediction, labels = get_predictions_and_labels(test_patient)
plot_results(hr, prediction, labels)
import plotly.express as px

px.line(labels)
px.line(prediction)


# Create a figure


# Add second trace


# tf.keras.backend.set_value(base_model.optimizer.learning_rate, 0.00001)
prediction
base_model.optimizer.learning_rate

data_dict = load_physio()
feature_dict = data_dict.copy()

aucs = []
for test_patient in feature_dict:
    print(f"\n\ntraining model for patient {test_patient}")
    train_dict = feature_dict.copy()
    test_dict = {test_patient: train_dict.pop(test_patient)}
    dl_train_physio = PhysioDataLoader(
        train_dict,
        features,
        batch_size=100,
        steps=100,
        train=True,
    )
    ds_physio = get_dataset_from_dl(dl_train_physio)
    ds_mesa = get_dataset_from_dl(dl_train_mesa)

    def get_mixed():
        return (
            tf.data.experimental.sample_from_datasets(
                (ds_mesa.unbatch(), ds_physio.unbatch()),
                weights=None,
                seed=None,
                stop_on_empty_dataset=False,
            )
            .shuffle(200)
            .batch(100)
            .as_numpy_iterator()
        )

    dl_test = PhysioDataLoader(
        test_dict,
        features,
        batch_size=1,
        steps=1,
        train=False,
    )
    for i in range(1):
        model = create_model()
        tf.keras.backend.set_value(model.optimizer.learning_rate, 0.00005)
        # model = tf.keras.models.clone_model(base_model)
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # )
        predictions = []
        for i in range(50):
            model.fit(get_mixed(), epochs=1, verbose=1)
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
    model.save(f"models/{test_patient}.h5")


def convert_to_arrays(x):
    if isinstance(x, tuple):
        return tuple((convert_to_arrays(e) for e in x))
    else:
        return x.numpy()
