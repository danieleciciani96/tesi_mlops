import mlflow
import pandas as pd
import cdsw, numpy, sklearn
from cmlapi.utils import Cursor

# path miglior modello MLFlow preso come argomento
#logged_model = sys.argv[1]

logged_model = '/home/cdsw/.experiments/yjjt-ep2e-zurn-9njl/9eb6-g9r6-n5fo-x0b1/artifacts/model'

"""
data = {
    
  'sensor_00' : '4219',
  'sensor_02' : '31294',
  'sensor_04' : '421',
  'sensor_06' : '645',
  'sensor_07' : '664',
  'sensor_08' : '7654',
  'sensor_09' : '12',
  'sensor_10' : '1321',
  'sensor_11' : '3124',
  'sensor_51' : '643',
}
"""

@cdsw.model_metrics
def predict(data):
    
    df = pd.DataFrame(data, index=[0])
    df.columns = ['sensor_00', 'sensor_02', 'sensor_04', 'sensor_06', 'sensor_07',
                  'sensor_08', 'sensor_09', 'sensor_10', 'sensor_11', 'sensor_51']
    
    #data = args.get('input')
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
 
    # Predict on a Pandas DataFrame.
    pred = loaded_model.predict(df)
    
    cdsw.track_metric("prediction", pred[0])
    cdsw.track_metric("data", data)
   
    return {'input_data': str(data), 'pred': str(pred[0])}
