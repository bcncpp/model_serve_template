import asyncio
from datetime import datetime
from typing import Optional, Callable, List
from model_inference.domain.entities import (
    PredictionInput,
    AnomalyPrediction,
    AnomalyType,
    PredictionResult,
    SensorData,
)
from model_inference.domain.services import InferenceService
from model_inference.common.logger import LoggingMixin


class AnomalyDetectionError(Exception): ...


class AnomalyDetectionService(InferenceService, LoggingMixin):
    def __init__(self, model_loader: Optional[Callable] = None, load_from_vertex=False):
        if model_loader:
            self.model = model_loader()
        elif load_from_vertex:
            self.model = self._load_model_from_vertex_ai()
        else:
            self.model = self._load_model_from_disk()

        self.model_version = getattr(self.model, "version", "unknown")
        self.log.info("Model loaded with version: %s", self.model_version)

    async def predict(
        self, prediction: PredictionInput, model_version: Optional[str] = None
    ) -> Optional[PredictionResult]:
        prediction_result = await self._run_model(prediction)
        return PredictionResult(
            predictions=[prediction_result],
            model_version=model_version,
            anomalies_count=0,
            total_predictions=1,
            processing_time_ms=0,
            metadata={},
        )

    async def predict_batch(
        self, prediction: List[PredictionInput], model_version: Optional[str] = None
    ) -> Optional[PredictionResult]:
        return None

    async def _run_model(self, data: PredictionInput) -> AnomalyPrediction:
        sensor: SensorData = data.data
        features = [
            sensor.val1,
            sensor.val2,
            sensor.val3,
            sensor.val4,
        ]

        loop = asyncio.get_running_loop()
        try:
            # Run the blocking model prediction in a thread pool
            score = await loop.run_in_executor(
                None,  # Use default thread pool
                lambda: self.model.predict_proba([features])[0][1],
            )
        except Exception as e:
            raise AnomalyDetectionError("Error running prediction") from e

        is_anomaly = score > 0.8
        return AnomalyPrediction(
            is_anomaly=is_anomaly,
            score=score,
            model_version=self.model_version,
            timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.OUTLIER,  # Replace as needed
        )

    def _load_model_from_vertex_ai(self):
        raise NotImplementedError()

    def _load_model_from_disk(self):
        return None
