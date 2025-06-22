from model_inference.domain.services.anomaly import AnomalyDetectionService


def get_anomaly_detection_service() -> AnomalyDetectionService:
    service = AnomalyDetectionService()
    return service
