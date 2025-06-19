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


class AnomalyResult(BaseModel):
    timestamp: str
    val1: float
    yhat: float
    yhat_lower: float
    yhat_upper: float
    anomaly: bool
