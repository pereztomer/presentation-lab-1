from torch.utils.data import Dataset
from glob import glob
import pandas as pd
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, folder_path, seq_length):
        files_paths = glob(f'{folder_path}/**.psv')
        # files_paths = files_paths[:200]
        assert len(files_paths) > 0, 'No available files'
        self.files_path = files_paths
        self.x = [None] * len(files_paths)
        self.samples_length = [None] * len(files_paths)
        self.y = [None] * len(files_paths)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        if self.x[idx] is None or self.y[idx] is None:
            patient_path = self.files_path[idx]
            patient_dataframe = pd.read_csv(patient_path, sep='|')
            if len(patient_dataframe[patient_dataframe.SepsisLabel == 1]) == 0:
                label = 0
            else:
                label = 1
                label_row = patient_dataframe[patient_dataframe.SepsisLabel == 1].index[0]
                patient_dataframe = patient_dataframe.iloc[:label_row + 1]

            patient_dataframe = patient_dataframe.drop(columns=['SepsisLabel'], inplace=False)

            # one_hundred_precent_null = ['Alkalinephos',
            #                             'PaCO2',
            #                             'FiO2',
            #                             'Bilirubin_total',
            #                             'BaseExcess',
            #                             'SaO2',
            #                             'pH',
            #                             'AST',
            #                             'TroponinI',
            #                             'Bilirubin_direct',
            #                             'Fibrinogen',
            #                             'Unit1',
            #                             'Unit2',
            #                             'EtCO2',
            #                             ]
            # patient_dataframe = patient_dataframe.drop(columns=one_hundred_precent_null, inplace=False)
            self.samples_length[idx] = patient_dataframe.shape[0]

            # padding the df:
            if patient_dataframe.shape[0] >= self.seq_length:
                # max_prev_vals = patient_dataframe.iloc[:-self.seq_length + 1].max().to_frame().T
                mean_prev_vals = patient_dataframe.iloc[:-self.seq_length + 1].max().to_frame().T
                patient_dataframe = patient_dataframe.iloc[-self.seq_length + 1:]
                patient_dataframe = pd.concat([mean_prev_vals, patient_dataframe], ignore_index=True)
                patient_dataframe = patient_dataframe.fillna(patient_dataframe.mean(numeric_only=True).round(1),
                                                             inplace=False)
            else:
                zero_padding = np.zeros((self.seq_length - patient_dataframe.shape[0], patient_dataframe.shape[1]))
                padding_df = pd.DataFrame(zero_padding, columns=patient_dataframe.columns)
                patient_dataframe = patient_dataframe.fillna(patient_dataframe.mean(numeric_only=True).round(1),
                                                             inplace=False)
                patient_dataframe = pd.concat([padding_df, patient_dataframe], ignore_index=True)

            patient_dataframe = patient_dataframe.fillna(0, inplace=False)

            self.x[idx] = patient_dataframe.to_numpy(dtype=np.float32)
            self.y[idx] = np.array(label).astype(np.float32)

        return self.x[idx], self.y[idx]


def equalize_ds(folder_path):
    files_paths = glob(f'{folder_path}/**.psv')
    new_ds_path = 'new_data/train'
    for file_path in files_paths:
        patient_dataframe = pd.read_csv(file_path, sep='|')
        if len(patient_dataframe[patient_dataframe.SepsisLabel == 1]) > 0:
            for index in range(1, 14):
                new_file_name = file_path.replace('.psv', '') + f'_{index}.psv'
                patient_dataframe.to_csv(new_file_name, sep='|', index=False)


def plot_graph(lst_1, lst_2, lst_1_name, lst_2_name, graph_name):
    import matplotlib.pyplot as plt

    # Create a list of index numbers for the x-axis
    index = list(range(len(lst_1)))

    # Plot both lists on the same graph
    plt.plot(index, lst_1, label=lst_1_name)
    plt.plot(index, lst_2, label=lst_2_name)

    # Add a legend to the graph
    plt.legend()

    # Add a title to the graph
    plt.title(graph_name)

    # Show the graph
    plt.show()
