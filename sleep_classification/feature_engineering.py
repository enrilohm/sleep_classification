# taken from https://github.com/ojwalch/sleep_classifiers/
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def convolve_with_dog(y, box_pts):
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2)
        )

    y = np.insert(
        y, 0, np.flip(y[0 : int(box_pts / 2)])
    )  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2) :]))
    y_smooth = np.convolve(y, box, mode="valid")

    return y_smooth


def get_heart_feature(hr_arr):
    interpolated_hr = convolve_with_dog(hr_arr.flatten(), 285)
    scalar = np.percentile(np.abs(interpolated_hr), 90)
    interpolated_hr = interpolated_hr / scalar

    hr_feature = []
    for i in range(len(hr_arr) // 30):
        start = max(0, i * 30 - 300)
        end = i * 30 + 300
        hr_feature.append(np.std(interpolated_hr[start:end]))
    return np.expand_dims(np.array(hr_feature), 1)


def get_activity_counts(data):
    fs = 50
    z_data = data[:, 2] / 9.81

    cf_low = 3
    cf_hi = 11
    order = 5
    w1 = cf_low / (fs / 2)
    w2 = cf_hi / (fs / 2)
    pass_band = [w1, w2]
    b, a = butter(order, pass_band, "bandpass")

    z_filt = filtfilt(b, a, z_data)
    z_filt = np.abs(z_filt)
    top_edge = 5
    bottom_edge = 0
    number_of_bins = 128

    bin_edges = np.linspace(bottom_edge, top_edge, number_of_bins + 1)
    binned = np.digitize(z_filt, bin_edges)
    epoch = 15
    counts = max2epochs(binned, fs, epoch)
    counts = (counts - 18) * 3.07
    counts[counts < 0] = 0
    # counts = np.expand_dims(counts, axis=1)

    label_length = len(counts) // 2
    counts = np.pad(counts, [0, 100])
    i = 13
    count_feature = []
    for i in range(label_length):
        start = max(0, i * 2 - 20)
        end = i * 2 + 20
        window_counts = counts[start:end]
        smoothed = smooth_gauss(window_counts)
        count_feature.append(smoothed)
    count_feature = np.expand_dims(count_feature, 1)
    return count_feature


def smooth_gauss(y):
    y = np.pad(y, [40 - len(y), 0])
    box = np.ones(len(y)) / len(y)
    mu = int(len(y) / 2.0)
    sigma = 50 / 15  # seconds
    gauss = np.exp(-1 / 2 * (((np.arange(0, len(y), 1) - mu) / sigma) ** 2))
    gauss = gauss / np.sum(gauss)

    sum_value = np.sum(np.multiply(gauss, y))
    return sum_value


def max2epochs(data, fs, epoch):
    data = data.flatten()

    seconds = int(np.floor(np.shape(data)[0] / fs))
    data = np.abs(data)
    data = data[0 : int(seconds * fs)]

    data = data.reshape(fs, seconds, order="F").copy()

    data = data.max(0)
    data = data.flatten()
    N = np.shape(data)[0]
    num_epochs = int(np.floor(N / epoch))
    data = data[0 : (num_epochs * epoch)]

    data = data.reshape(epoch, num_epochs, order="F").copy()
    epoch_data = np.sum(data, axis=0)
    epoch_data = epoch_data.flatten()

    return epoch_data
