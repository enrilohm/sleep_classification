from sleep_classification.feature_engineering import (
    get_heart_feature,
    get_activity_counts,
)
from sleep_classification.preprocess import preprocess_data
import shelve
from tqdm import tqdm

def add_features(data_dict):
    for patient_id, data in tqdm(data_dict.items()):
        data["heart_feature"] = get_heart_feature(data["hr_arr"])
    for patient_id, data in tqdm(data_dict.items()):
        data["activity_feature"] = get_activity_counts(data["acc_arr"])

def load_mondy():
    with shelve.open("data/mondy/processed", "r") as shelf:
        d=dict(shelf)
    return d


if __name__=="__main__":
    with shelve.open("data/mondy/test_data", "r") as shelf:
        data_dict=dict(shelf)

    for patient_id in tqdm(data_dict):
        patient_data = data_dict[patient_id]
        hr_df = patient_data["hr_df"]
        hr_df = hr_df[hr_df["hr"]!=0]
        patient_data["hr_df"]=hr_df
        acc_df = patient_data["acc_df"]
        hr_arr, acc_arr,_ = preprocess_data(hr_df, acc_df)
        patient_data["hr_arr"] = hr_arr
        patient_data["acc_arr"] = acc_arr

    add_features(data_dict)

    with shelve.open("data/mondy/processed", "c") as shelf:
        shelf.update(data_dict)

    # import plotly.express as px
    # px.line(list(data_dict.values())[0]["hr_df"])
