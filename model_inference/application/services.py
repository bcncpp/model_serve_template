from typing import List
import pandas as pd
from model_inference.domain.sensor import SensorData
class AnomalyDetectorService:
    async def detect(self, data: List[SensorData]) -> List[SensorData]:
        return []


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
