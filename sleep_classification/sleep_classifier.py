import os
import pandas as pd
import numpy as np
import tensorflow as tf

from typing import List
from glob import glob

from sleep_classification.data import get_data_path
from sleep_classification.preprocess import preprocess_data
from sleep_classification.feature_engineering import (
    get_heart_feature,
    get_activity_counts,
)


def load_models() -> List[tf.keras.Model]:
    model_dir = get_data_path("models")
    model_paths = glob(os.path.join(model_dir, "*.h5"))
    models = [tf.keras.models.load_model(model_path) for model_path in model_paths]
    return models


class SleepClassifier:
    def __init__(self, gpu=False):
        if not gpu:
            tf.config.set_visible_devices([], "GPU")
        self.models = load_models()

    def predict(self, hr_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create predictions.

        Parameters
        ----------
        hr_df :
            DataFrame containing a single heart rate column.
            Index should be time information compatible with pandas pd.to_datetime

        acc_df :
            DataFrame containing 3 accelerometer columns (X,Y,Z) in SI units.
            Index should be time information compatible with pandas pd.to_datetime

        Returns
        -------
        A DataFrame with awake predictions and datetime index
        """
        hr_df = hr_df.copy()
        acc_df = acc_df.copy()
        hr_arr, acc_arr, time_grid = preprocess_data(hr_df, acc_df)

        hr_fe = get_heart_feature(hr_arr)
        acc_fe = get_activity_counts(acc_arr)

        predictions = [
            model((np.expand_dims(hr_fe, 0), np.expand_dims(hr_fe, 0)))
            for model in self.models
        ]
        prediction = sum(predictions) / len(predictions)
        wake_prediction = prediction[0][:, 0]
        prediction = time_grid.set_index(0)
        prediction["prediction"] = wake_prediction
        return prediction
