from fastapi import APIRouter, Depends, HTTPException, Body
from model_inference.domain.entities import PredictionInput
from model_inference.domain.services.anomaly import AnomalyDetectionService
from model_inference.api.deps import get_anomaly_detection_service


class AnomalyDetectorController:
    def __init__(self):
        self.router = APIRouter(prefix="/anomaly", tags=["anomaly"])
        self.router.add_api_route("/", self.get_anomaly, methods=["POST"])

    @classmethod
    def create(cls) -> "AnomalyDetectorController":
        return cls()

    async def get_anomaly(
        self,
        input_sensors: PredictionInput = Body(...),
        service: AnomalyDetectionService = Depends(get_anomaly_detection_service),
    ):
        result = await service.predict(data=input_sensors)
        if not result:
            raise HTTPException(status_code=404, detail="No anomaly detected")
        return result
