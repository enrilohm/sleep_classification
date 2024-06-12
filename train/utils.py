import random
import numpy as np
from tensorflow.keras import layers as L


def random_flip(data, prob=0.5):
    if random.random() <= prob:
        data = {k: np.flip(v, 0) for k, v in data.items()}

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

def hybrid_pooling(x, kernel, strides):
    max = L.MaxPooling1D(kernel, strides=strides)(x)
    avg = L.AveragePooling1D(kernel, strides=strides)(x)
    result = L.Concatenate(axis=2)([max, avg])
    return result
