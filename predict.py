import argparse
import glob
import numpy as np
import re
import pandas as pd
from xgboost import XGBClassifier

def crop_if_sick(df):
    first_reported_sick_index = df[df.SepsisLabel == 1]
    if not first_reported_sick_index.empty:
        first_reported_sick_index = first_reported_sick_index.index[0]
        df = df.loc[:first_reported_sick_index]
    return df

def data_imputation(arr):
    # calculate the column-wise mean ignoring NaN values
    col_mean = np.nanmean(arr, axis=0)

    # create a boolean mask for NaN values
    nan_mask = np.isnan(arr)

    # replace NaN values with the column-wise mean
    arr[nan_mask] = np.take(col_mean, nan_mask.nonzero()[1])
    return arr

def create_ds(source):
    patient_list = []
    files = sorted([(int(re.findall(r'\d+', s)[-1]),s) for s in glob.glob('/'+source+'/*')])
    files = [s[1] for s in files]
    getlen = len(pd.read_csv(files[0], sep='|').iloc[0])
    ds = np.zeros(
        (len(files), getlen))
    for i, file in enumerate(files):
        patient_list.append(file.split('/')[-1][:-4])
        df = pd.read_csv(file, sep='|')
        df = crop_if_sick(df)
        df=df.drop(['SepsisLabel'],errors='ignore', axis=1)
        num_rows = df.shape[0]
        # add a new column with the number of rows
        df.insert(0, 'num_rows', num_rows)
        
        last_row = df.fillna(method='ffill').iloc[[-1]]
        ds[i] = np.asarray(last_row)
    ds= data_imputation(ds)
    return ds,patient_list



def XGboost_filter(X):
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    nthread=1)
    xgb.load_model('model.json')
    predicted_y = xgb.predict(X)
    np.savetxt('output2.txt', predicted_y)
    return predicted_y
    




    

def save_predictions_to_csv(ids, predictions, filename):
    # create a DataFrame with the two lists
    df = pd.DataFrame({'id': ids, 'prediction': predictions})
    # save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def main(args):
    #/home/student/HW1/data/test
    source_folder = args.input
    print("exptracting ds Train...")
    X, patient_list = create_ds(
        source=source_folder)
    Y= XGboost_filter(X)
    save_predictions_to_csv(patient_list,Y,'prediction.csv')
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('input', help='Input file path')
    args = parser.parse_args()
    main()
