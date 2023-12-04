from sleep_classification.data import get_data_path
from sleep_classification.preprocess import preprocess_data
from sleep_classification.feature_engineering import get_heart_feature, get_activity_counts
import tensorflow as tf
from glob import glob
import os
import numpy as np

def load_models():
    model_dir = get_data_path("models")
    model_paths = glob(os.path.join(model_dir, "*.h5"))
    models =  [tf.keras.models.load_model(model_path) for model_path in model_paths]
    return models



class SleepClassifier:
    def __init__(self):
        self.models = load_models()
    def predict(self, hr_df, acc_df):
        hr_arr, acc_arr = preprocess_data(hr_df, acc_df)
        hr_fe = get_heart_feature(hr_arr)
        acc_fe = get_activity_counts(acc_arr)

        predictions = [model.predict((np.expand_dims(hr_fe,0), np.expand_dims(hr_fe,0))) for model in self.models]
        prediction = sum(predictions)/len(predictions)
        wake_prediction = prediction[0][:,0]
        return wake_prediction
