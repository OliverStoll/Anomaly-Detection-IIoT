import pandas as pd
import os
import datetime


class DataPipeKBM:
    anomalies = {'stabilus': ["17/06/19 10:22:00", "17/06/19 11:22:00", "17/06/19 12:24:00", "17/06/19 13:24:00"],
                     'stadtkehl': ["22/05/19 06:39:59", "22/05/19 07:40:36", "22/05/19 08:41:13", "22/05/19 09:41:51"],
                     'pumpe-v2': ["2/16/18 17:59", "2/16/18 18:59"],
                     'pumpe-v3': ["5/20/18 7:35", "5/25/18 18:35"]
                 }
    file_names = ["stabilus", "stadtkehl", "wasserwerke-a", "wasserwerke-b", "piaggio", "gummipumpe", "pumpe-v2", "pumpe-v3"]
    column_names = ['vibration-x', 'vibration-y', 'vibration-z', 'temperature', 'time', 'time_sec']
    file_ending = '_full.csv'

    def __init__(self, import_path: str, export_path: str):
        self.import_path = import_path
        self.export_path = export_path

    def prepare_data(self, sep: str = ',', split_temperature_from_tags:bool = True):
        """
        Clean and preprocess KBM CSV files.

        :param sep: delimiter of the csv files
        :param split_temperature_from_tags: if True, the tags are split into separate columns
        """
        for file in os.scandir(self.import_path):
            df = pd.read_csv(file, sep=sep)
            if split_temperature_from_tags:
                df[['tags', 'temperature']] = df['tags'].str.split('temperature=', expand=True)
                df['temperature'] = df['temperature'].str.split(' ', expand=True)[0]
            df = df.sort_values(by=['time'])
            df['time_sec'] = df['time'].apply(lambda x: x.split('.')[0])
            df = df[self.column_names]
            df.to_csv(f"{self.export_path}/{file.name}{self.file_ending}", index=False)

    def check_data_quality(self):
        """Log time discontinuities between samples."""
        for file in os.scandir(self.import_path):
            path = f"{self.import_path}/{file.name}{self.file_ending}"
            df = pd.read_csv(path)
            previous_time = datetime.datetime.min
            for time_str in df['time_sec'].unique():
                current_time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                time_difference = current_time - previous_time
                previous_time = current_time
                if time_difference < datetime.timedelta(seconds=0):
                    print(file.name, current_time, time_difference)
                elif time_difference == datetime.timedelta(seconds=1):
                    print(file.name, current_time, time_difference)
