import pandas as pd
from tqdm import tqdm
from data_utils import CustomDataset
from torch.utils.data import DataLoader

def main():
    max_seq_length = 336 # This is the maximun seq length from the train and test sets
    train_path = 'data/train'
    train_batch_size = 32
    train_shuffle = False
    train_ds = CustomDataset(folder_path=train_path, seq_length=50)
    # for val in tqdm(train_ds):
    #     print(val[0].shape)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle)

    for val in train_data_loader:
        print(val)

if __name__ == '__main__':
    main()
