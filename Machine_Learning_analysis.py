import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier


def evaluate_pred(pred, y_val):
    accuracy = metrics.accuracy_score(y_val, y_pred=pred)
    precision = metrics.precision_score(y_val, y_pred=pred)
    recall = metrics.recall_score(y_val, y_pred=pred)
    F1 = 2 * ((precision*recall)/(precision+recall))
    print("acuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

def Over_sample(arr):  
    np.random.seed(123)
    # find rows where the last feature is equal to 1
    rows_to_duplicate = np.where(arr[:, -1] == 1)[0]

    # create a new array with 13 copies of the rows to duplicate
    duplicated_rows = np.repeat(arr[rows_to_duplicate], 13, axis=0)

    # stack the original array with the duplicated rows
    new_arr = np.vstack([arr, duplicated_rows])
    np.random.shuffle(new_arr)
    return (new_arr)


def tree_fitter(x_train, y_train, x_val, y_val):
    print("Xgboost train...")
    params = {
        'min_child_weight': [0.5, 1, 5, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4,
                        12.8, 25.6, 51.2, 102.4, 200],
        'reg_lambda': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 200],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    nthread=1)
    folds = 3
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4,
                                       cv=skf.split(x_train, y_train), random_state=1001)

    start_time = timer(None)

    random_search.fit(x_train, y_train)
    timer(start_time)

    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    predicted_y = random_search.best_estimator_.predict(x_val)
    print("Tree:")
    evaluate_pred(pred=predicted_y, y_val=y_val)
    print("...................")
    return predicted_y


def crop_if_sick(df, filename):
    first_reported_sick_index = df[df.SepsisLabel == 1]
    if not first_reported_sick_index.empty:
        first_reported_sick_index = first_reported_sick_index.index[0]
        df = df.loc[:first_reported_sick_index]
        # print(filename)
        # print (df)
    return df


def data_imputation(arr):
    # calculate the column-wise mean ignoring NaN values
    col_mean = np.nanmean(arr, axis=0)

    # create a boolean mask for NaN values
    nan_mask = np.isnan(arr)

    # replace NaN values with the column-wise mean
    arr[nan_mask] = np.take(col_mean, nan_mask.nonzero()[1])
    return arr



def show_Statistics(df):
    st = df.describe()
    print (st.to_string())
    hist = df.hist(bins=100, ylabelsize=10, xlabelsize=10,figsize=(30, 18))
    # plt.savefig('my_plot.png')
    print("stats")


def create_ds(source, Get_Y=False,OverSample=False):
    df_list = []
    files = sorted(glob.glob(source))
    getlen = len(pd.read_csv(files[0], sep='|').iloc[0])+1
    training_ds = np.zeros(
        (len(files), getlen))
    for i, file in enumerate(files):
        df = pd.read_csv(file, sep='|')
        df = crop_if_sick(df, os.path.basename(file))
        last_row = df.fillna(method='ffill').iloc[[-1]]
        last_row = last_row.reset_index()
        last_row.insert(0, 'number_of_rows', last_row.pop('index'))
        # print(last_row)
        # df_list.append(last_row)
        training_ds[i] = np.asarray(last_row)[0]
    # big_table = pd.concat(df_list, axis=0)
    # show_Statistics(big_table)
    training_ds= data_imputation(training_ds)
    if OverSample:
      training_ds= Over_sample(training_ds)
    X = training_ds[:, :-1]
    Y = training_ds[:, -1]
    if Get_Y == True:
        return X, Y
    return X

def Random_Forest(x_train, y_train, x_val, y_val):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    pred = rf.predict(x_val)
    print("Random_Forest :")
    evaluate_pred(pred=pred, y_val=y_val)
    print("...................")
    return pred



if __name__ == '__main__':
    print("exptracting ds Train...")
    X_train, Y_train = create_ds(
        source="/home/student/HW1/data/train/*", Get_Y=True, OverSample=False)
    print(len(X_train)," rows were loaded")
    print(np.sum(Y_train)/len(X_train)," of them belong to class sick ..")
    print("exptracting ds Test...")
    X_Test, Y_test = create_ds(
        source="/home/student/HW1/data/test/*", Get_Y=True)
    
    Random_Forest(x_train=X_train,y_train=Y_train, x_val=X_Test,y_val=Y_test)
    tree_fitter(x_train=X_train,y_train=Y_train, x_val=X_Test,y_val=Y_test)
    

