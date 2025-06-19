from fastapi import FastAPI
from api.routes import sensor_anomalies

app = FastAPI()
app.include_router(sensor_anomalies.router, prefix="/api")
