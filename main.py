import pandas as pd
from tqdm import tqdm
from data_utils import CustomDataset, equalize_ds
from torch.utils.data import DataLoader


def main():
    max_seq_length = 336  # This is the maximun seq length from the train and test sets
    train_path = 'data/train'
    equalize_ds(folder_path=train_path)
    exit()
    train_batch_size = 500
    train_shuffle = False
    train_ds = CustomDataset(folder_path=train_path, seq_length=50)
    for val in train_ds:
        if val[0].shape[1] != 41:
            print(val[0].shape)
    exit()
    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle)

    one_class = 0
    zero_class = 0
    for val in tqdm(train_data_loader):
        one_class += val[1].tolist().count(1)
        zero_class += val[1].tolist().count(0)

    print(one_class)
    print(zero_class)


if __name__ == '__main__':
    main()
