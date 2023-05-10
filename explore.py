import re
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




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay


def stats(y_true,y_scores):
    # Generate example data
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Plot ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("2.png")
    # Calculate confusion matrix
    y_pred = y_scores > 0.5
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    # initialize using the raw 2D confusion matrix
    # and output labels (in our case, it's 0 and 1)
    display = ConfusionMatrixDisplay(conf_mat, display_labels=[0, 1])

    # set the plot title using the axes object
    # ax.set(title='Confusion Matrix for the Diabetes Detection Model')

    # show the plot.
    # Pass the parameter ax to show customizations (ex. title)
    display.plot()
    plt.savefig("1.png")

    # fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    #
    # # initialize using the raw 2D confusion matrix
    # # and output labels (in our case, it's 0 and 1)
    # display = ConfusionMatrixDisplay(test_cm, display_labels=model.classes_)
    #
    # # set the plot title using the axes object
    # ax.set(title='Confusion Matrix for the Diabetes Detection Model')
    #
    # # show the plot.
    # # Pass the parameter ax to show customizations (ex. title)
    # display.plot(ax=ax)
    # plt.show()

def plot_roc_curve(model, X_test, y_test):
    # Make predictions on the test data
    y_pred_proba = model.predict_proba(X_test)

    # Calculate the FPR, TPR, and threshold values
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("figfig")

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

def drop_columns_by_number(arr, columns_to_drop):
    # Drop the specified columns from the NumPy array
    arr = np.delete(arr, columns_to_drop, axis=1)

def tree_fitter(x_train, y_train, x_val, y_val):
    print("Xgboost train...")
    # params = {
    #     'min_child_weight': [0.5, 1, 5],
    #     'gamma': [0.1, 0.5, 1, 1.5, 2],
    #     'subsample': [0.4, 0.6, 0.8, 1.0],
    #     'reg_alpha': [0, 0.1, 0.2],
    #     'reg_lambda': [0, 0.1, 0.2],
    #     'colsample_bytree': [0.6, 0.8],
    #     'max_depth': [5,6]
    # }
    # params = {
    #     'min_child_weight': [1,5],
    #     'gamma': [2],
    #     'subsample': [0.6,0.8],
    #     'reg_alpha': [0.1],
    #     'reg_lambda': [0, 0.1],
    #     'colsample_bytree': [0.6],
    #     'max_depth': [5,6]
    # }

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    nthread=1)
    # folds = 3
    # param_comb = 5

    # skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    # random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=4,
    #                                    cv=skf.split(x_train, y_train), random_state=1001)

    # start_time = timer(None)

    # random_search.fit(x_train, y_train)
    # timer(start_time)

    # results = pd.DataFrame(random_search.cv_results_)
    # results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    # print(random_search.best_params_)


    # random_search.best_estimator_.save_model("model73.json")

    # predicted_y = random_search.best_estimator_.predict(x_val)
    xgb.load_model('model.json')
    # plot_roc_curve(random_search.best_estimator_,x_val,y_val)
    predicted_y =xgb.predict(x_val)
    # np.savetxt('eval_123.txt', predicted_y)
    print("XGBOOST :")
    evaluate_pred(pred=predicted_y, y_val=y_val)
    print("...................")
    # get_feature_importance_order(random_search.best_estimator_)
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

def get_feature_importance_order(clf):
    # Get the feature importance from the XGBoost classifier
    importance = clf.feature_importances_

    # Create a list of tuples with feature names and importance scores
    feature_importance = [(f'f{i}', importance_score) for i, importance_score in enumerate(importance)]

    # Sort the feature importance scores in descending order
    feature_importance_sorted = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Get a list of feature names in order of their importance scores
    feature_names_ordered = [feature[0] for feature in feature_importance_sorted]
    print (feature_names_ordered)
    

def create_aggregation_dict(df):
    agg_dict={}
    cols= df.columns.tolist()
    for col in cols:
        if col =="Age" or col == "Gender":
            agg_dict[col]= 'median'
        if col == "SepsisLabel" or col == "ICULOS":
            agg_dict[col]= lambda x: x.dropna().iloc[-1]
        else: 
            agg_dict[col]='mean'
    return agg_dict


def show_Statistics(df):
    st = df.describe()
    print (st.to_string())
    hist = df.hist(bins=100, ylabelsize=10, xlabelsize=10,figsize=(30, 18))
    # plt.savefig('my_plot.png')
    print("stats")


def create_ds(source, Get_Y=False,OverSample=False,tit=False):
    df_list = []
    files = sorted([(int(re.findall(r'\d+', s)[-1]),s) for s in glob.glob(source)])
    files = [s[1] for s in files]
    # if tit:
    #     # files = sorted([(int(re.findall(r'\d+', s)[-1]),s) for s in glob.glob(source)])
    #     # files = [s[1] for s in files]
    #     files = sorted(glob.glob(source))
    #     testfiles= sorted(glob.glob('/home/student/HW1/data/test/*'))
    #     files= files + testfiles
    getlen = len(pd.read_csv(files[0], sep='|').iloc[0])+1
    training_ds = np.zeros(
        (len(files), getlen))
    for i, file in enumerate(files):
        df = pd.read_csv(file, sep='|')
        df = crop_if_sick(df, os.path.basename(file))
                # get the number of rows
        # df.drop("Calcium").drop()
        num_rows = df.shape[0]
        # add a new column with the number of rows
        df.insert(0, 'num_rows', num_rows)
        
        last_row = df.fillna(method='ffill').iloc[[-1]]
        # last_row = df.median(skipna=True).to_frame().transpose()
        # agg_dict = create_aggregation_dict(df)
        # last_row = df.agg(agg_dict, axis=0, skipna=True).to_frame().transpose()
        # last_row = last_row.reset_index()
        # print(last_row)
        # df_list.append(last_row)
        training_ds[i] = np.asarray(last_row)
    # big_table = pd.concat(df_list, axis=0)
    # delme= [np.argpartition(np.isnan(training_ds).sum(axis=0), -3)[-3:]]# show_Statistics(big_table)
    
    #training_ds= data_imputation(training_ds)
    
    if OverSample:
      training_ds= Over_sample(training_ds)
    X = training_ds[:, :-1]
    Y = training_ds[:, -1]
    # X=(X-np.mean(X,axis=0))/(np.std(X,axis=0)**2+0.0001)
    # X = np.delete(X, delme, axis=1)
    
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
        source="/home/student/HW1/data/train/*", Get_Y=True, OverSample=False,tit=True)
    print(len(X_train)," rows were loaded")
    print(np.sum(Y_train)/len(X_train)," of them belong to class sick ..")
    print("exptracting ds Test...")
    X_Test, Y_test = create_ds(
        source="/home/student/HW1/data/test/*", Get_Y=True)
    
    # Random_Forest(x_train=X_train,y_train=Y_train, x_val=X_Test,y_val=Y_test)
    y_preds= tree_fitter(x_train=X_train,y_train=Y_train, x_val=X_Test,y_val=Y_test)
    stats(Y_test,y_preds)
