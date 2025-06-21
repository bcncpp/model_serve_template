from model_inference.services.anomaly import AnomalyDetectorService


def get_anomaly_detector_service() -> AnomalyDetectorService:
    user_repo = AnomalyDetectorService()
    return user_repo
