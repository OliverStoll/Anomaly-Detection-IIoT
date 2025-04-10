import os
import pandas as pd


def restructure_data_columns(datasets_dir_path: str, round_digits=4) -> None:
    """ Restructure all datasets in the dir to be a single column per feature. """
    dataset_files = [f for f in os.listdir(datasets_dir_path) if f.endswith("2_1000.csv")]
    for filename in dataset_files:
        dataset_path = f"{datasets_dir_path}/{filename}"
        use_columns = [0, 2, 4, 6] if filename.endswith("1_1000.csv") else [0, 1, 2, 3]
        data_df = pd.read_csv(dataset_path, usecols=use_columns)
        vibration_data = []
        for i in range(data_df.shape[1]):
            vibration_data.extend(data_df.iloc[:, i])
        vibration_data = [round(vibration_entry, round_digits) for vibration_entry in vibration_data]
        with open(f"{datasets_dir_path}/{filename.replace('_', '_COL-')}", "w") as f:
            f.write("Vibration\n")
            for i in vibration_data:
                f.write(str(i) + ",\n")
