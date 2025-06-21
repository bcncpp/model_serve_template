from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List
from models import SensorData  # adjust import as needed
from model_inferencs.applicatonservices import (
    AnomalyDetectorService,
    get_anomaly_detector_service,
)  # adjust import as needed


class AnomalyDetectorController:
    def __init__(self):
        self.router = APIRouter(prefix="/anomaly", tags=["anomaly"])
        self.router.add_api_route("/", self.get_anomaly, methods=["POST"])
        # Removed /{user_id} since it's not used in the function

    async def get_anomaly(
        self,
        input_sensors: List[SensorData] = Body(...),
        service: AnomalyDetectorService = Depends(get_anomaly_detector_service),
    ):
        user = await service.detect(data=input_sensors)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
