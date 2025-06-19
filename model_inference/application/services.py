from typing import List
import pandas as pd
import bentoml
from model_inference.domain.sensor import SensorData

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 60}
)
class AnomalyDetectorService:
    def __init__(self, model_path: str = "models/prophet_anomaly_model.json", serialization_method: str = "json"):
        """
        Initialize the service
        
        Args:
            model_path: Path to the saved model
            serialization_method: 'json' (recommended), 'joblib', or 'pickle'
        """
        self.model_manager = ProphetModelManager(model_path, serialization_method)
        self._model = None
    
    def _load_model(self) -> Prophet:
        """Lazy load the model"""
        if self._model is None:
            self._model = self.model_manager.load_model()
        return self._model
    
    def _preprocess_data(self, data: List[SensorData]) -> pd.DataFrame:
        """Optimized data preprocessing"""
        # Convert to DataFrame more efficiently
        data_dicts = [{"ds": item.timestamp, "y": item.val1} for item in data]
        df = pd.DataFrame(data_dicts)
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('ds').drop_duplicates(subset=['ds']).reset_index(drop=True)
        
        # Handle missing values
        df['y'] = df['y'].fillna(df['y'].median())
        
        return df
    
    @bentoml.api(input=JSON(pydantic_model=List[SensorData]), output=JSON())
    async def detect_anomalies(self, data: List[SensorData]) -> List[AnomalyResult]:
        """
        Detects anomalies in the 'val1' time series using pre-trained Prophet model.
        Returns a list of all data points with anomaly flags.
        """
        if not data or len(data) < 2:
            return []
        
        try:
            # Load the pre-trained model
            model = self._load_model()
            
            # Preprocess data
            df_prophet = self._preprocess_data(data)
            
            if len(df_prophet) < 2:
                return []
            
            # Make predictions using the pre-trained model
            forecast = model.predict(df_prophet)
            
            # Merge results efficiently
            df_merged = df_prophet.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                on='ds', 
                how='left'
            )
            
            # Vectorized anomaly detection
            df_merged['anomaly'] = (
                (df_merged['y'] < df_merged['yhat_lower']) | 
                (df_merged['y'] > df_merged['yhat_upper'])
            )
            
            # Convert back to response format
            results = []
            for _, row in df_merged.iterrows():
                results.append(AnomalyResult(
                    timestamp=row['ds'].isoformat(),
                    val1=row['y'],
                    yhat=row['yhat'],
                    yhat_lower=row['yhat_lower'],
                    yhat_upper=row['yhat_upper'],
                    anomaly=bool(row['anomaly'])
                ))
            
            return results
            
        except Exception as e:
            # Log error and return empty result
            print(f"Error in anomaly detection: {str(e)}")
            return []
    
    @bentoml.api(input=JSON(pydantic_model=List[SensorData]), output=JSON())
    async def get_anomalies_only(self, data: List[SensorData]) -> List[AnomalyResult]:
        """
        Returns only the anomalous data points.
        """
        all_results = await self.detect_anomalies(data)
        return [result for result in all_results if result.anomaly]
    
    @bentoml.api(input=JSON(pydantic_model=List[SensorData]), output=JSON())
    async def train_model(self, training_data: List[SensorData]) -> dict:
        """
        Train and save a new Prophet model with the provided data.
        """
        try:
            self.model_manager.train_and_save_model(training_data, force_retrain=True)
            # Reset cached model to force reload
            self._model = None
            return {"status": "success", "message": "Model trained and saved successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Training failed: {str(e)}"}
    
    @bentoml.api(input=JSON(), output=JSON())
    async def health_check(self) -> dict:
        """Health check endpoint with model info"""
        model_info = self.model_manager.get_model_info()
        return {
            "status": "healthy", 
            "model_loaded": self._model is not None,
            "model_info": model_info
        }

    @bentoml.api(input=JSON(), output=JSON())
    async def model_info(self) -> dict:
        """Get detailed model information"""
        return self.model_manager.get_model_info()

class AnomalyDetectorService:
    def __init__(self, model_path: str):
        pass
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
