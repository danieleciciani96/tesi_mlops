import mlflow
import pandas as pd
import cdsw, numpy, sklearn
from cmlapi.utils import Cursor

logged_model = '/home/cdsw/.experiments/n03k-3b0z-nbdp-wz8l/94k7-jivb-s3j1-g7r9/artifacts/model'

"""
data = {
    
  'sensor_04' : '4219',
  'sensor_19' : '31294',
  'sensor_20' : '421',
  'sensor_21' : '645',
  'sensor_38' : '664',
  'sensor_39' : '7654',
  'sensor_40' : '12',
  'sensor_41' : '1321',
  'sensor_42' : '3124',
}
"""

@cdsw.model_metrics
def predict(data):
    
    df = pd.DataFrame(data, index=[0])
    df.columns = ['sensor_04', 'sensor_19', 'sensor_20', 'sensor_21', 'sensor_38', 'sensor_39', 'sensor_40', 'sensor_41', 'sensor_42']

    #data = args.get('input')
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
 
    # Predict on a Pandas DataFrame.
    pred = loaded_model.predict(df)
    
    cdsw.track_metric("prediction", str(pred))
    cdsw.track_metric("data", data)
   
    return {'input_data': str(data), 'pred': str(pred[0])}
