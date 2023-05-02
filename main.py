import pandas as pd

from data_utils import CustomDataset


def main():
    train_path = 'data/train'
    train_ds = CustomDataset(folder_path=train_path)
    max_shape = 0
    for idx, val in enumerate(train_ds):
        if max_shape < val[0].shape[0]:
            max_shape = val[0].shape[0]

    test_path = 'data/test'
    test_ds = CustomDataset(folder_path=test_path)
    for idx, val in enumerate(test_ds):
        if max_shape < val[0].shape[0]:
            max_shape = val[0].shape[0]

    print(max_shape)
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
