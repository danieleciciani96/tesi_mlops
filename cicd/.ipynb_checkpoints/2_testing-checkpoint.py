#The shape of your data
import unittest
import pytest
import pandas as pd
 
from evidently.pipeline.column_mapping import ColumnMapping

from evidently.report import Report
from evidently.metric_preset import DataDrift, NumTargetDrift
 
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQuality, DataStability
from evidently.tests import *



path_processed = "/home/cdsw/data/pump_processed.csv"
data = pd.read_csv(path_processed)
#data.drop("Unnamed: 0", axis=1, inplace=True)

print(data.head())
    
def test_n_columns(data):
    n_cols = data.shape[1]
    assert n_cols == 12
    
    
def test_min_max_s4(s4):
    min_allowed = -1
    max_allowed = 100000000
    
    assert s4.describe().max() <= max_allowed
    assert s4.describe().min() >= min_allowed
    
def test_target(target):
    assert target.unique()[0] == 1
    assert target.unique()[1] == 0
    
    
test_n_columns(data)
test_min_max_s4(data.sensor_04)
test_target(data.machine_status)

"""
#to do: pylint quality check

from pylint.lint import Run
import sys
import json

def quality_check(pyfile):

    THRESHOLD = 5
    if len(sys.argv) < 2:
        print("Module to evaluate need to be the first argument")

    run = Run(['--errors-only', '--output-format=json', pyfile], do_exit=False)
    score = run.linter.stats['global_note']"

    outputjson = run.linter.stats['by_module']
    outputjson['score'] = score

    with open('log_error.json', 'w') as outfile:
        json.dump(outputjson, outfile)

    if score < THRESHOLD:
        sys.exit

quality_check(sys.argv[1])

"""