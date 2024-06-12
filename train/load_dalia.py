from sleep_classification.feature_engineering import (
    get_heart_feature,
    get_activity_counts,
)
from sleep_classification.preprocess import preprocess_data
import shelve
from tqdm import tqdm
import pickle
from glob import glob
import pandas as pd

def load_dalia():
    with shelve.open("data/ppg+dalia/processed", "r") as shelf:
        d = dict(shelf)
    return d

def add_features(data_dict):
    for patient_id, data in tqdm(data_dict.items()):
        data["heart_feature"] = get_heart_feature(data["hr_arr"])
    for patient_id, data in tqdm(data_dict.items()):
        data["activity_feature"] = get_activity_counts(data["acc_arr"])

if __name__=="__main__":
    patient_ids = ["S" + str(i) for i in range(1, 16)]
    data_dict = {}
    for patient_id in tqdm(patient_ids):
        path = f"data/ppg+dalia/data/PPG_FieldStudy/{patient_id}/{patient_id}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        acc_df = pd.DataFrame(data["signal"]["wrist"]["ACC"])
        acc_df.index = [i / 32 * 10 ** 9 for i in range(len(acc_df))]
        hr_df = pd.DataFrame(data["label"].astype(int).reshape(-1, 1))
        hr_df.index = [i * 2 * 10 ** 9 for i in range(len(hr_df))]
        label_df = pd.DataFrame(index=[i * 30 * 10 ** 9 for i in range(len(hr_df) * 15)], data={"label":0})
        data_dict["dalia_" + patient_id] = {"hr_df": hr_df, "acc_df": acc_df, "label_df":label_df}

    for patient_id in tqdm(data_dict):
        patient_data = data_dict[patient_id]
        hr_df = patient_data["hr_df"]
        acc_df = patient_data["acc_df"]
        label_df = patient_data["label_df"]
        hr_arr, acc_arr, label_arr = preprocess_data(hr_df, acc_df, label_df)
        patient_data["hr_arr"] = hr_arr
        patient_data["acc_arr"] = acc_arr
        patient_data["label_arr"] = label_arr

    add_features(data_dict)

    with shelve.open("data/ppg+dalia/processed", "c") as shelf:
        shelf.update(data_dict)


    import plotly.express as pe
    patient_id="dalia_S6"
    pe.line(data_dict[patient_id]["hr_arr"])
    pe.line(data_dict[patient_id]["heart_feature"])
    pe.line(data_dict[patient_id]["activity_feature"])
