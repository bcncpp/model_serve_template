from abc import ABC
from typing import List, Optional
from model_inference.domain.entities import PredictionInput, PredictionResult

from typing import Generic, TypeVar

T = TypeVar("T")


class InferenceService(Generic[T]):
    async def predict(
        self, prediction: PredictionInput, model_version: Optional[str] = None
    ) -> Optional[PredictionResult]: ...
    async def predict_batch(
        self, prediction: List[PredictionInput], model_version: Optional[str] = None
    ) -> Optional[PredictionResult]: ...
