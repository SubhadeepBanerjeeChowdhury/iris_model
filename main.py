import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import os
import json


def get_data(base_dir):
    data_file_names = [x for x in os.listdir(base_dir) if x.endswith('.csv')]
    data = {}
    for name in data_file_names:
        path_file = os.path.join(base_dir, name)
        data = pd.read_csv(r"C:\Users\Subhadeep\PycharmProjects\pythonProject6\model\iris.csv")
    return data


def split_data(data, test_size=0.2, random_state=42):
    df = data['iris.csv']
    X = df.drop('variety', axis=1)
    y = df['variety']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def train_model(X_train, y_train):
    log_reg = LogisticRegression()
    model = log_reg.fit(X_train, y_train)
    return model


def save_model(model):
    dump(model, "../model/iris_prediction.joblib")
    print("Model saved")



if __name__ == "__main__":
    base_dir = r'C:\Users\Subhadeep\PycharmProjects\pythonProject6\model\iris.csv'
    data = get_data(base_dir)

    split_data = split_data(data['iris.csv'])

    m = train_model(split_data['X_train'], split_data['y_train'])
    save_model(m)
