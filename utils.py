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

import mlflow
import sklearn
import warnings
import tempfile
from sklearn.model_selection import GridSearchCV
import os
from datetime import datetime
import pandas as pd


def log_run(gridsearch, experiment_name: str, model_name: str, run_index: int, conda_env, tags={}):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
        tags (dict): Dictionary of extra data and tags (usually features)
    """
    
    cv_results = gridsearch.cv_results_
    
    with mlflow.start_run(run_name=str(run_index)) as run:  

        mlflow.log_param("k-folds", gridsearch.cv.n_splits)

        print("Logging parameters")
        params = list(gridsearch.param_grid.keys())
        for param in params:
            mlflow.log_param(param, cv_results["param_%s" % param][run_index])

        print("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][run_index])
            mlflow.log_metric(score_name.replace("mean","std"), cv_results[score_name.replace("mean","std")][run_index])

        print("Logging model")        
        mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, conda_env=conda_env)
        
        print("Logging extra data related to the experiment")
        mlflow.set_tags(tags) 

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()
        
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run_id)

        
def log_results(gridsearch, experiment_name, model_name, tags={}, log_only_best=False):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        tags (dict): Dictionary of extra tags
        log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """
    conda_env = {
            'name': 'mlflow-env',
            'channels': ['defaults'],
            'dependencies': [
                'python=3.8.13',
                'scikit-learn>=1.1.2',
                {'pip': ['xgboost==1.0.1']}
            ]
        }


    best = gridsearch.best_index_

    #mlflow.set_tracking_uri("http://kubernetes.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    if(log_only_best):
        log_run(gridsearch, experiment_name, model_name, best, conda_env, tags)
    else:
        for i in range(len(gridsearch.cv_results_['params'])):
            log_run(gridsearch, experiment_name, model_name, i, conda_env, tags)