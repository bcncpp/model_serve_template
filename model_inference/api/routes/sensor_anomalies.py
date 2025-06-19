from fastapi import APIRouter, Depends, HTTPException
from model_inference.application.services import AnomalyDetectorService
from model_inference.api.deps import get_anomaly_detector_service
from model_inference.domain.sensor import SensorData

router = APIRouter()


@router.get("/sensor")
async def get_user(
    input_sensors: list[SensorData],
    service: AnomalyDetectorService = Depends(get_anomaly_detector_service),
):
    user = await service.detect(data=input_sensors)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.__dict__
