import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn import preprocessing


def preprocess_data(df):
    # remove fields that are not used
    X = df.iloc[:, 5:32]
    X.drop('links', axis=1, inplace=True)
    X.drop('link-tags', axis=1, inplace=True)
    X.drop('published', axis=1, inplace=True)
    X.drop('modified', axis=1, inplace=True)
    X.drop('description', axis=1, inplace=True)
    # convert time strings into time
    tp = df['published']
    tm = df['modified']
    size = len(tp)
    time_pub = np.zeros(size)
    time_mod = np.zeros(size)
    for i in range(0, size):
        s = tp[i]
        t = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
        timestamp = datetime.timestamp(t)
        time_pub[i] = timestamp
    for i in range(0, size):
        s = tm[i]
        t = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
        timestamp = datetime.timestamp(t)
        time_mod[i] = timestamp
    X.insert(0, 'published', time_pub)
    X.insert(0, 'modified', time_mod)
    # convert objects into numbers (categorical encoding)
    X = pd.get_dummies(X, X.columns[X.dtypes == 'object'])
    # normalize data
    x = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)
    return X


if __name__ == '__main__':
    # import train dataset
    df = pd.read_csv("train.csv", sep=",")
    X_train = preprocess_data(df)
    y_train = df['label']
    # train model
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train, y_train)

    # import test dataset
    df_test = pd.read_csv("test.csv", sep=",")
    X_test = preprocess_data(df_test)
    # make predictions
    predictions = xgb_classifier.predict(X_test)
    # output csv file
    ids = df_test['id']
    arr = pd.DataFrame()
    arr['id'] = ids
    arr['label'] = predictions
    arr.to_csv("labels.csv", index=False)
