import os
import pandas as pd


class DatasetResampler:
    def __init__(self, directory_path: str, new_sampling_rate: int, feature_cols_tuple: tuple[int, int]) -> None:
        self.directory_path = directory_path
        self.new_sampling_rate = new_sampling_rate
        self.feature_cols_tuple = feature_cols_tuple
        self.dataset_type  = 'kbm' if 'kbm' in self.directory_path else 'bearing'
        self.old_sampling_rate = 20480 if self.dataset_type == 'bearing' else 800

    def resample_all_csv_in_directory(self) -> None:
        """
        Resample all files in a folder to a new sampling rate.

        The files need to be all of the same type (bearing or kbm) and have the same column names.

        :param new_sampling_rate: number of samples to resample to
        """

        for file in os.listdir(self.directory_path):
            if file.endswith("_full.csv"):
                print(f"Resampling {file} Rate: {self.new_sampling_rate}")
                self.resample_csv(file_path=f"{self.directory_path}/{file.replace('_full.csv', '')}")

    def resample_csv(self, file_path: str) -> pd.DataFrame:
        save_path = f"{file_path}_{self.new_sampling_rate}.csv"
        data_df = pd.read_csv(f"{file_path}_full.csv")
        if self.dataset_type == 'kbm':
            data_df = data_df.iloc[:-(len(data_df) % self.old_sampling_rate)]
        if os.path.exists(save_path):
            os.remove(save_path)
        temp_df = data_df.iloc[:,:-1]
        resampling_factor = self.old_sampling_rate // self.new_sampling_rate
        resampled_df = temp_df.groupby(data_df.index // resampling_factor)
        resampled_df = resampled_df.mean()
        resampled_df.to_csv(save_path, index=False)
        return resampled_df


def _resample_all_to_defaults(resample_rates: list[int]) -> None:
    for i in resample_rates:
        kbm_resampler = DatasetResampler(directory_path='data/kbm', new_sampling_rate=i, feature_cols_tuple=(2, 6))
        bearing_resampler = DatasetResampler(directory_path='data/bearing', new_sampling_rate=i, feature_cols_tuple=(0, 4))
        kbm_resampler.resample_all_csv_in_directory()
        bearing_resampler.resample_all_csv_in_directory()