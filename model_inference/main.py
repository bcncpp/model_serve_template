from fastapi import FastAPI
from api.controllers.anomaly_detector import AnomalyDetectorController

app = FastAPI()
anomaly_detector = AnomalyDetectorController()
app.include_router(anomaly_detector.router, prefix="/api")
