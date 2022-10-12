import requests
import sys
import json
import pandas as pd
import cdsw, time, os
import json

path_processed = "/home/cdsw/data/pump_processed.csv"
df = pd.read_csv(path_processed)
df = df.drop(["machine_status"], axis=1)

filter_period = "2018-08-01 00:00:00"
#filter_period = sys.argv[1] if len(sys.argv) > 1 else None

df_new = df.loc[df.timestamp >= filter_period].drop("timestamp", axis=1)
print("Len df_new: {}".format(len(df_new)))

df_new = df_new.head(5000)

def perform_requests(x):
    curr_row = x.to_json()
    r = requests.post('https://modelservice.ml-1096e3e8-eda.ps-sandb.a465-9q4k.cloudera.site/model', 
                       data='{"aaccessKey":"mil0x0azw6yujrnkpxxketb0zsrhdyyn","request": ' + curr_row + '}',
                       headers={'Content-Type': 'application/json'})

df_new.apply(perform_requests, axis=1)