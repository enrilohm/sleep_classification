import pandas as pd
import numpy as np
import datetime


def process_input_hr(hr_df):
    hr_df.index = pd.to_datetime(hr_df.index)
    hr_df = hr_df.astype(float)
    assert len(hr_df.columns) == 1, "exactly 1 column needed for heart_rate!"
    return hr_df


def process_input_acc(acc_df):
    acc_df.index = pd.to_datetime(acc_df.index)
    acc_df = acc_df.astype(float)
    assert len(acc_df.columns) == 3, "exactly 3 columns needed for acceleration!"
    return acc_df


def align_sequences(hr_df, acc_df, label_df):
    min_time_labels = max(
        [x.index.min() for x in [hr_df, acc_df]]
    ) + datetime.timedelta(seconds=14.5)
    max_time_labels = min(
        [x.index.max() for x in [hr_df, acc_df]]
    ) - datetime.timedelta(seconds=14.5)

    if label_df is None:
        time_grid = pd.date_range(
            start=min_time_labels, end=max_time_labels, freq="30S"
        )
    else:
        label_df.index = pd.to_datetime(label_df.index)
        time_grid = [
            x for x in label_df.index if min_time_labels <= x <= max_time_labels
        ]

    min_time_labels = time_grid[0]
    max_time_labels = time_grid[-1]

    hr_grid = pd.date_range(
        start=time_grid[0] - datetime.timedelta(seconds=14.5),
        end=time_grid[-1] + datetime.timedelta(seconds=14.5),
        freq="1S",
    )
    acc_grid = pd.date_range(
        start=time_grid[0] - datetime.timedelta(seconds=14.99),
        end=time_grid[-1] + datetime.timedelta(seconds=14.99),
        freq="20L",
    )

    hr_interpolation = np.interp(hr_grid, hr_df.index, hr_df.to_numpy()[:, 0])
    hr_interpolation = np.expand_dims(hr_interpolation, -1)

    acc_interpolations = [
        np.interp(acc_grid, acc_df.index, acc_df.to_numpy()[:, i]) for i in range(3)
    ]
    acc_interpolation = np.concatenate(
        [
            np.expand_dims(acc_interpolation, -1)
            for acc_interpolation in acc_interpolations
        ],
        axis=1,
    )

    if label_df is not None:
        labels = label_df.loc[time_grid].to_numpy()
    else:
        labels = pd.DataFrame(time_grid)
    return hr_interpolation, acc_interpolation, labels


def preprocess_data(hr_df, acc_df, label_df=None):
    hr_df = process_input_hr(hr_df)
    acc_df = process_input_acc(acc_df)

    hr_interpolation, acc_interpolation, labels = align_sequences(
        hr_df, acc_df, label_df
    )

    return hr_interpolation, acc_interpolation, labels
