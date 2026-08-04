import pandas as pd
import os

def load_data(file_names, cutoff_date, base_path="C:/Users/szymo/Desktop/Nowy folder/"):
    data_frames = [] 

    for name in file_names:
        file_path = os.path.join(base_path, f"{name}.csv")

        df = pd.read_csv(file_path, index_col='Data')

        df = df[['Zamkniecie']] / 100 
        df = df.rename(columns={'Zamkniecie': name})
        df = df[df.index >= cutoff_date]
        data_frames.append(df)

    combined_df = pd.concat(data_frames, axis=1)

    combined_df = combined_df.ffill().dropna()

    return combined_df