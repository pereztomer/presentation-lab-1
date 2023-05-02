import pandas as pd
from tqdm import tqdm
from data_utils import CustomDataset


def main():
    max_seq_length = 336 # This is the maximun seq length from the train and test sets
    train_path = 'data/train'
    train_ds = CustomDataset(folder_path=train_path, seq_length=50)
    for val in tqdm(train_ds):
        print(val[0].shape)

    exit()

    # train_ds = CustomDatasetAug(sentences=train_sentences_idx_padded,
    #                             tags=train_y_padded,
    #                             positions=train_pos_idx_padded,
    #                             d_tags=train_d_tags_idx_padded,
    #                             seq_len_vals=train_sentences_real_len)
    #
    # train_data_loader = DataLoader(dataset=train_ds,
    #                                batch_size=train_batch_size,
    #                                shuffle=train_shuffle)


if __name__ == '__main__':
    main()
