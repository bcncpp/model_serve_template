from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import List, Any, Optional


class AnomalyType(Enum):
    NORMAL = "normal"
    HIGH_VALUE = "high_value"
    LOW_VALUE = "low_value"


class AnomalyPrediction(BaseModel):
    """Domain entity for anomaly prediction results"""

    prediction_id: str
    timestamp: datetime
    actual_value: float
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    anomaly_type: AnomalyType
    is_anomaly: bool
    confidence_score: float
    severity_score: float


class PredictionResult(BaseModel):
    """Domain entity for batch prediction results"""

    predictions: List[AnomalyPrediction]
    model_version: Optional[str]
    total_predictions: int
    anomalies_count: int
    processing_time_ms: float
    metadata: Optional[dict]


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


class PredictionInput(BaseException):
    data: SensorData | Any
