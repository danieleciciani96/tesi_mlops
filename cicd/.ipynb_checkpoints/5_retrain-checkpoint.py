import sys
import json
import pandas as pd
import cdsw, time, os
import mlflow
import mlflow.sklearn

from cmlapi.utils import Cursor
import cmlapi
import string
import random
import json


from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection

path_processed = "/home/cdsw/data/pump_processed.csv"

df = pd.read_csv(path_processed)
df['target'] = df['machine_status']
df = df.drop(["machine_status"], axis=1)

from_period = "2018-08-01 00:00:00"
#filter_period = sys.argv[1] if len(sys.argv) > 1 else None



reference = df.loc[df.timestamp < from_period].drop("timestamp", axis=1).sample(10000, random_state=2022)

# in a real world use case, production dataset is the dataset collected from predictions logs metrics.
production = df.loc[df.timestamp >= from_period].drop("timestamp", axis=1).sample(10000, random_state=2022)

#"""
### Read last production metrics for db

model_deployment_crn = "crn:cdp:ml:us-west-1:558bc1d2-8867-4357-8524-311d51259233:workspace:de20f039-2b2a-48ea-b388-1be9ffd49da1/492d014a-d103-4f7e-a8f7-ca8cb4509794"

current_timestamp_ms = int(time.time()) * 1000

known_metrics = cdsw.read_metrics(model_deployment_crn=model_deployment_crn,
                                  start_timestamp_ms=0,
                                  end_timestamp_ms=current_timestamp_ms)  

production = pd.io.json.json_normalize(known_metrics["metrics"])

# rename columns
colums_mapping = {'metrics.data.sensor_00': 'sensor_00',
                  'metrics.data.sensor_02': 'sensor_02',
                  'metrics.data.sensor_04': 'sensor_04',
                  'metrics.data.sensor_06': 'sensor_06',
                  'metrics.data.sensor_07': 'sensor_07',
                  'metrics.data.sensor_08': 'sensor_08',
                  'metrics.data.sensor_09': 'sensor_09',
                  'metrics.data.sensor_10': 'sensor_10',
                  'metrics.data.sensor_11': 'sensor_11',
                  'metrics.data.sensor_51': 'sensor_51',
                  'metrics.prediction' : 'target'
                 }
production.rename(columns=colums_mapping, inplace=True)

# drop unused colums
cols_to_keep = ['sensor_00', 'sensor_02', 'sensor_04', 'sensor_06', 'sensor_07', 
                'sensor_08', 'sensor_09', 'sensor_10', 'sensor_11', 'sensor_51', 'target']

# select only required cols
production = production[cols_to_keep]

# cast sensors value to numeric
cols_to_keep.remove('target')
production[cols_to_keep] = production[cols_to_keep].astype(float)
#"""

reference = reference.reset_index(drop=True)




# Evidently columns mapping
categorical_features = ['target']
numerical_features = ['sensor_00','sensor_02', 'sensor_04', 'sensor_06',  'sensor_07', 'sensor_08',
                     'sensor_09', 'sensor_10', 'sensor_11', 'sensor_51']

column_mapping = ColumnMapping()
column_mapping.target = 'target'
column_mapping.numerical_features = numerical_features

## create dashboard for data drift & Cat Target Drift
dash = Dashboard(tabs=[
                    DataDriftTab(verbose_level=1),
                    CatTargetDriftTab(verbose_level=1)
                    ])

dash.calculate(reference, production)

dash.save("/home/cdsw/reports/report_data_target_drift.html".format(from_period))

### get profiles and track it on MLFlow
mlflow.set_experiment("drift_detection")

with mlflow.start_run():
    
    mlflow.log_param("From Period", from_period)
    
    # TARGET DRIFT PROFILE
    pump_target_drift_profile = Profile(sections=[CatTargetDriftProfileSection()])
    pump_target_drift_profile.calculate(reference, production, column_mapping=column_mapping) 
    target_drift_score = json.loads(pump_target_drift_profile.json())["cat_target_drift"]['data']['metrics']['target_drift']
    
    mlflow.log_metric("Target Drift score", target_drift_score)
   

    # DATA DRIFT PROFILE
    pump_data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    pump_data_drift_profile.calculate(reference, production, column_mapping=column_mapping) 
    data_drift_bool = json.loads(pump_data_drift_profile.json())['data_drift']['data']['metrics']['dataset_drift']
    
    mlflow.log_metric("Data Drift Detected", int(data_drift_bool))
 
    # track the single drift for each sensors
    for feature in numerical_features:
        score = json.loads(pump_data_drift_profile.json())['data_drift']['data']['metrics'][feature]['drift_score']
        mlflow.log_metric(feature, score)

    mlflow.end_run()
     
"""
# Retrain the CICD pipeline with CML API
client = cmlapi.default_client()

if target_drift_score > 0 or data_drift_bool == True:
    project_id = "py8u-dlpy-yq5n-a5vr"
    preprocessing_job_id = "727h-bc97-nfg8-uhy6"

    jobrun_body = cmlapi.CreateJobRunRequest(project_id, preprocessing_job_id)
    job_run = client.create_job_run(jobrun_body, project_id, preprocessing_job_id)
    run_id = job_run.id
"""