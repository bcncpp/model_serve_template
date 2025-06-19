from typing import List
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from pydantic import BaseModel
from datetime import datetime
class SensorData(BaseModel):
    timestamp: datetime
    machine_id: str
    failure: int
    val1: int
    val2: int
    val3: int
    val4: int
    field7: int
    val5: int
    val6: int
    val7: float

#records = [SensorData(**item) for item in data]

def detect_anomalies(data: List[SensorData]):
    """
    Detects anomalies in the 'val1' time series using Prophet.
    Returns a DataFrame of anomalies with timestamps and values.
    """
    df = pd.DataFrame([r.dict() for r in data])
    df_prophet = df[['timestamp', 'val1']].rename(columns={'timestamp': 'ds', 'val1': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    df_merged = df_prophet.copy()
    df_merged['yhat'] = forecast['yhat']
    df_merged['yhat_lower'] = forecast['yhat_lower']
    df_merged['yhat_upper'] = forecast['yhat_upper']
    # Flag anomalies
    df_merged['anomaly'] = (df_merged['y'] < df_merged['yhat_lower']) | (df_merged['y'] > df_merged['yhat_upper'])
    anomalies = df_merged[df_merged['anomaly'] == True]
    return anomalies

