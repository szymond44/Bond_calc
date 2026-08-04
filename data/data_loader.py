import pandas as pd
import os

DEFAULT_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def load_data(file_names, cutoff_date, base_path=None):
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    data_frames = []

    for name in file_names:
        file_path = os.path.join(base_path, f"{name}.csv")

        df = pd.read_csv(file_path, index_col='Data', parse_dates=['Data'])

        df = df[['Zamkniecie']] / 100
        df = df.rename(columns={'Zamkniecie': name})
        df = df[df.index >= pd.to_datetime(cutoff_date)]
        data_frames.append(df)

    combined_df = pd.concat(data_frames, axis=1)
    combined_df = combined_df.sort_index()

    combined_df = combined_df.ffill().dropna()

    return combined_df
