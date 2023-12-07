#Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump 
import os
import json

#Function to retrieve data from CSV files in a specified directory
def get_data(base_dir):
    data_file_names = [x for x in os.listdir(base_dir) if x.endswith('.csv')]
    data = {}
    for name in data_file_names:
        path_file = os.path.join(base_dir, name)
# Read the CSV file into a DataFrame and store it in the 'data' dictionary
        data = pd.read_csv(r"C:\Users\Subhadeep\PycharmProjects\pythonProject6\model\iris.csv")
    return data

# Function to split data into training and testing sets
def split_data(data, test_size=0.2, random_state=42):
    df = data['iris.csv']
    X = df.drop('variety', axis=1)
    y = df['variety']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Return a dictionary containing the split data
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

# Function to train a logistic regression model
def train_model(X_train, y_train):
    log_reg = LogisticRegression()
    model = log_reg.fit(X_train, y_train)
    return model

# Function to save the trained model to a joblib file
def save_model(model):
    dump(model, "../model/iris_prediction.joblib")
    print("Model saved")


# Main block to execute the code
if __name__ == "__main__":
    base_dir = r'C:\Users\Subhadeep\PycharmProjects\pythonProject6\model\iris.csv'
    data = get_data(base_dir)

    split_data = split_data(data['iris.csv'])

    m = train_model(split_data['X_train'], split_data['y_train'])
    save_model(m)
