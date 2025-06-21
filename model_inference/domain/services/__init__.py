from abc import ABC
from typing import List, Optional
from model_inference.domain.entities import PredictionInput, BatchPredictionResult


class InferenceService(ABC):
    async def predict(
        self, prediction: PredictionInput, model_version: Optional[str] = None
    ) -> Optional[BatchPredictionResult]: ...
    async def predict_batch(
        self, prediction: List[PredictionInput], model_version: Optional[str] = None
    ) -> Optional[BatchPredictionResult]: ...
