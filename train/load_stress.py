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


def add_features(data_dict):
    for patient_id, data in tqdm(data_dict.items()):
        data["heart_feature"] = get_heart_feature(data["hr_arr"])
    for patient_id, data in tqdm(data_dict.items()):
        data["activity_feature"] = get_activity_counts(data["acc_arr"])


def load_stress():
    with shelve.open("data/stress/processed", "r") as shelf:
        d = dict(shelf)
    return d


if __name__ == "__main__":
    import plotly.express as pe

    data_dict = {}
    patient_ids = ["S" + str(i) for i in range(1, 11)]
    for patient_id in tqdm(patient_ids):
        for exam in ["Final", "Midterm 1", "Midterm 2"]:
            acc_path = f"data/stress/Data/{patient_id}/{exam}/ACC.csv"
            acc_df = pd.read_csv(acc_path, header=None)
            assert acc_df.iloc[0].min() == acc_df.iloc[0].max()
            assert all(acc_df.iloc[1] == 32)
            acc_start = acc_df.iloc[0, 0]
            acc_df = acc_df.iloc[2:] / 64 * 9.81

            hr_path = f"data/stress/Data/{patient_id}/{exam}/HR.csv"
            hr_df = pd.read_csv(hr_path, header=None)
            assert hr_df.iloc[1, 0] == 1
            hr_start = hr_df.iloc[0, 0]
            hr_df = hr_df[2:]

            acc_df.index = [
                acc_start * 10 ** 9 + i / 32 * 10 ** 9 for i in range(len(acc_df))
            ]
            hr_df.index = [
                hr_start * 10 ** 9 + i * 1 * 10 ** 9 for i in range(len(hr_df))
            ]
            label_df = pd.DataFrame(
                index=[
                    min(hr_start, acc_start) * 10 ** 9 + i * 30 * 10 ** 9
                    for i in range(len(hr_df) * 30)
                ],
                data={"label": 0},
            )
            data_dict["stress_" + patient_id + "_" + exam] = {
                "hr_df": hr_df,
                "acc_df": acc_df,
                "label_df": label_df,
            }
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

    with shelve.open("data/stress/processed", "c") as shelf:
        shelf.update(data_dict)

    patient_id = "stress_S6_Final"
    # pe.line(data_dict[patient_id]["hr_df"])
    # pe.line(data_dict[patient_id]["heart_feature"])
    # pe.line(data_dict[patient_id]["activity_feature"])
