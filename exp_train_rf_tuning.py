import utils

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
from sklearn.model_selection import TimeSeriesSplit


from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get url from DVC

path = "data/pump_sensors_processed.csv"


mlflow.set_experiment("train_rf")

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
        mlflow.log_param("input_cols", data.shape[1])

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "machine_status", 0 Broker, 1 Normal
        train_x = train.drop(["machine_status"], axis=1)
        test_x = test.drop(["machine_status"], axis=1)
        train_y = train[["machine_status"]]
        test_y = test[["machine_status"]]

        #logs articats: columns used for modelling
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv("features.csv", header=False, index=False)
        mlflow.log_artifact("features.csv")

        cols_y = pd.DataFrame(list(train_y.columns))
        cols_y.to_csv("targets.csv", header=False, index=False)
        mlflow.log_artifact("targets.csv")

        # MODELLING
        # Scaling data
        train_x, test_x = normalize_df(train_x, test_x)
        
        # time splitting of the data for cross validation
        folds = TimeSeriesSplit(n_splits=5)
        
        
        # rf parameters
        param_grid = {'n_estimators': [10, 25, 50, 100,150, 200],
                      'max_depth': [1, 3, 5, 10, 20, 30, 50] }
        
        # declate rf and fit
        rf_clf = RandomForestClassifier(criterion='gini', 
                                              random_state=21, 
                                              n_jobs=-1)
        
        rf_gridcv = GridSearchCV(estimator=rf_clf,
                                param_grid=param_grid,
                                cv=folds,
                                scoring='f1_macro',
                                n_jobs=-1,
                                return_train_score=True)

        rf_gridcv.fit(train_x, train_y)
        
        # PREDICT
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