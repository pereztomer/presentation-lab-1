from torch.utils.data import DataLoader, Dataset
from glob import glob
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, folder_path):
        files_paths = glob(f'{folder_path}/**.psv')
        assert len(files_paths) > 0, 'No available files'
        self.files_path = files_paths
        self.x = [0]*len(files_paths)
        self.y = [0]*len(files_paths)

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        if self.x[idx] == 0 or self.y[idx] == 0:
            patient_path = self.files_path[idx]
            patient_dataframe = pd.read_csv(patient_path, sep='|')
            if len(patient_dataframe[patient_dataframe.SepsisLabel == 1]) == 0:
                label = 0
            else:
                label = 1
                label_row = patient_dataframe[patient_dataframe.SepsisLabel == 1].index[0]
                patient_dataframe = patient_dataframe.iloc[:label_row + 1]

            patient_dataframe = patient_dataframe.fillna(patient_dataframe.mean(numeric_only=True).round(1), inplace=False)
            patient_dataframe = patient_dataframe.fillna(0, inplace=False)
            patient_dataframe = patient_dataframe.drop(columns=['SepsisLabel'], inplace=False)
            self.x[idx] = patient_dataframe.to_numpy()
            self.y[idx] = label

        return self.x[idx], self.y[idx]

