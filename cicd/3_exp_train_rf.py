import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
import boto


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#path = "/home/cdsw/data/pump_processed.csv"
path = 'data/pump_sensors_processed.csv'

mlflow.set_experiment("rf_scores")

def eval_metrics(actual, pred):
    f1_macro = f1_score(actual, pred, average='macro')
    acc_score = accuracy_score(actual, pred)
    return f1_macro, acc_score


def normalize_df(df_train, df_test):
    """
    Function to normalize the data using minimax scaler
    """
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train.values)
    scaled_test = scaler.transform(df_test.values)
    train = pd.DataFrame(data=scaled_train, 
                         columns=df_train.columns, 
                         index=df_train.index)
    test = pd.DataFrame(data=scaled_test, 
                        columns=df_test.columns, 
                        index=df_test.index)
    return train, test


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    try:
        data = pd.read_csv(path)        
        
    except Exception as e:
        logger.exception(
            "Something went wrong", e
        )
        
    with mlflow.start_run():

        # Let's track some params
        mlflow.log_param("data_path", path)
        mlflow.log_param("input_rows", data.shape[0]) 
        mlflow.log_param("input_cols", data.shape[1] -1 )

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "machine_status", 0 Broker, 1 Normal
        train_x = train.drop(["machine_status"], axis=1)
        test_x = test.drop(["machine_status"], axis=1)
        train_y = train[["machine_status"]]
        test_y = test[["machine_status"]]

        #logs articats: columns used for modelling
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv("data/features.csv", header=False, index=False)
        mlflow.log_artifact("data/features.csv")

        cols_y = pd.DataFrame(list(train_y.columns))
        cols_y.to_csv("data/targets.csv", header=False, index=False)
        mlflow.log_artifact("data/targets.csv")

        # MODELLING
        # Scaling data
        train_x, test_x = normalize_df(train_x, test_x)
        
        # rf parameters --> taking params from gridsearch cv
        
        n_estimators = int(sys.argv[1])  if len(sys.argv) > 1 else 3
        max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        # declate rf and fit
        rf = RandomForestClassifier(n_estimators=n_estimators, 
                                    max_depth=max_depth,
                                    criterion='gini', 
                                    random_state=21, 
                                    n_jobs=-1)
        rf.fit(train_x, train_y)
        
        # predict
        pred = rf.predict(test_x)

        (f1, acc) = eval_metrics(test_y, pred)

        print("Random Forrest (n_estimators=%f, max_depth=%f):" % (n_estimators, max_depth))
        print("  F1 Score: %s" % f1)
        print("  Accuracy: %s" % acc)

        #Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        #Log metrics
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Accuracy", acc)
        # Log model
        mlflow.sklearn.log_model(rf, "model")

        mlflow.end_run()