from glob import glob

import numpy as np
import pandas as pd


def extract_sample(patient_path):
    patient_dataframe = pd.read_csv(patient_path, sep='|')
    if len(patient_dataframe[patient_dataframe.SepsisLabel == 1]) == 0:
        label = 0
    else:
        label = 1
        label_row = patient_dataframe[patient_dataframe.SepsisLabel == 1].index[0]
        patient_dataframe = patient_dataframe.iloc[:label_row + 1]

    patient_dataframe = patient_dataframe.drop(columns=['SepsisLabel'], inplace=False)
    return patient_dataframe, label


def calc_nulls_percentage(ds_path):
    files_paths = glob(f'{ds_path}/**.psv')
    zero_counter = 0
    one_counter = 0
    total_df_one = None
    total_df_zero = None
    zero_seq_len_list = []
    one_seq_len_list = []
    for idx, sample_path in enumerate(files_paths):
        if idx % 1000 == 0:
            print(f'{idx}/{len(files_paths)}')
        # if idx == 50:
        #     break
        sample_df, sample_label = extract_sample(sample_path)
        if sample_label == 0:
            zero_counter += 1
        else:
            one_counter += 1

        if sample_label == 0:
            zero_seq_len_list.append(len(sample_df))
            if total_df_zero is None:
                total_df_zero = sample_df
            else:
                total_df_zero = pd.concat([total_df_zero, sample_df], axis=0)
        elif sample_label == 1:
            one_seq_len_list.append(len(sample_df))
            if total_df_one is None:
                total_df_one = sample_df
            else:
                total_df_one = pd.concat([total_df_one, sample_df], axis=0)

    set_type = ds_path.split('/')[-1]
    # total df statistics:
    total_df = pd.concat([total_df_one, total_df_zero], axis=0)
    total_df_mean = total_df.mean().to_frame()
    total_df_mean.columns = [f'{set_type}.total mean']

    total_df_std = total_df.std().to_frame()
    total_df_std.columns = [f'{set_type}.total std']

    total_df_null = (total_df.isnull().sum() * 100 / len(total_df)).to_frame()
    total_df_null.columns = [f'{set_type}.total null']

    total_df_stats = pd.concat([total_df_mean,
                                total_df_std,
                                total_df_null],
                               axis=1)

    # class statistics:
    df_one_mean = total_df_one.mean().to_frame()
    df_one_mean.columns = [f'{set_type}.label one mean']

    df_zero_mean = total_df_zero.mean().to_frame()
    df_zero_mean.columns = [f'{set_type}.label zero mean']

    df_one_std = total_df_one.std().to_frame()
    df_one_std.columns = [f'{set_type}.label one std']

    df_zero_std = total_df_zero.std().to_frame()
    df_zero_std.columns = [f'{set_type}.label zero std']

    df_null_one = (total_df_one.isnull().sum() * 100 / len(total_df_one)).to_frame()
    df_null_one.columns = [f'{set_type}.label one null']

    df_null_zero = (total_df_zero.isnull().sum() * 100 / len(total_df_zero)).to_frame()
    df_null_zero.columns = [f'{set_type}.label zero null']

    labels_df = pd.concat([df_one_mean,
                           df_zero_mean,
                           df_one_std,
                           df_zero_std,
                           df_null_one,
                           df_null_zero],
                          axis=1)
    print(np.average(zero_seq_len_list), np.average(one_seq_len_list))
    print(np.std(zero_seq_len_list), np.std(one_seq_len_list))
    total_seq_len = zero_seq_len_list + one_seq_len_list

    print(np.average(total_seq_len), np.std(total_seq_len))

    return total_df_stats, labels_df


def main():
    train_path = 'original_data/train'
    test_path = 'original_data/test'
    train_total_df, train_labels_df = calc_nulls_percentage(train_path)
    test_total_df, test_labels_df = calc_nulls_percentage(test_path)

    # total_df = pd.concat([train_total_df, test_total_df], axis=1)
    # total_df.to_csv('total_stats.csv')
    #
    # class_df = pd.concat([train_labels_df, test_labels_df], axis=1)
    # class_df.to_csv('classes_stats.csv')

    print('hi')


if __name__ == '__main__':
    main()
