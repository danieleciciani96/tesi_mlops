#The shape of your data
import unittest
import pytest
import pandas as pd

path_processed = "/home/cdsw/data/pump_processed.csv"
        
data = pd.read_csv(path_processed)
data.drop("Unnamed: 0", axis=1, inplace=True)

def test_n_columns(data):
    n_cols = data.shape[1]
    assert n_cols == 10
    
    
def test_min_max_s4(s4):
    min_allowed = 0
    max_allowed = 1000000
    
    assert s4.describe().max() <= max_allowed
    assert s4.describe().min() > min_allowed
    
def test_target(target):
    assert target.unique()[0] == 1
    assert target.unique()[1] == 0
    
    

test_n_columns(data)
test_min_max_s4(data.sensor_04)
test_target(data.machine_status)


"""
def test_loss():
  
  in_tensor = tf.placeholder(tf.float32, (None, 3))
  
  labels = tf.placeholder(tf.int32, None, 1))
  
  model = Model(in_tensor, labels)
  
  sess = tf.Session()
  loss = sess.run(model.loss, feed_dict={
    in_tensor:np.ones(1, 3),
    labels:[[1]]
  })
  
  assert loss != 0
"""
  
  