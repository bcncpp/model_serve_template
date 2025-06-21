from model_inference.common.logger import LoggingMixin
from model_inference.domain.services import InferenceService
import bentoml


@bentoml.service(
    name="anomaly_detection_inference",
    resources={"cpu": "2", "memory": "2Gi"},
    traffic={"timeout": 120},
)
class AnomalyDetectionService(InferenceService, LoggingMixin):
    """BentoML service for anomaly detection inference"""

    def __init__(self, model_loader=None):
        # Load model on service initialization
        self.model = None
        self.model_version = None
        self.feature_columns = [
            "value",
            "rolling_mean_7d",
            "rolling_std_7d",
            "lag_1h",
            "lag_24h",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
        ]

    def _run_prophet_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Prophet model inference"""
        try:
            # Prepare data for Prophet
            prophet_df = df[["timestamp", "value"]].rename(
                columns={"timestamp": "ds", "value": "y"}
            )
            forecast = self.model.predict(prophet_df[["ds"]])

            # Merge results
            df["yhat"] = forecast["yhat"].values
            df["yhat_lower"] = forecast["yhat_lower"].values
            df["yhat_upper"] = forecast["yhat_upper"].values

            return df

        except Exception as e:
            logger.error(f"Prophet inference failed: {e}")
            # Fallback to dummy predictions
            df["yhat"] = df["value"] * 0.95
            df["yhat_lower"] = df["yhat"] - 5
            df["yhat_upper"] = df["yhat"] + 5
            return df

    def _classify_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify anomalies based on predictions"""
        # Determine anomalies
        df["is_anomaly"] = (df["value"] < df["yhat_lower"]) | (
            df["value"] > df["yhat_upper"]
        )

        # Classify anomaly types
        def get_anomaly_type(row):
            if not row["is_anomaly"]:
                return "normal"
            elif row["value"] > row["yhat_upper"]:
                return "high_value"
            elif row["value"] < row["yhat_lower"]:
                return "low_value"
            else:
                return "normal"

        df["anomaly_type"] = df.apply(get_anomaly_type, axis=1)

        # Calculate confidence scores
        def calculate_confidence(row):
            if not row["is_anomaly"]:
                return 0.95

            interval_width = row["yhat_upper"] - row["yhat_lower"]
            if row["value"] > row["yhat_upper"]:
                deviation = row["value"] - row["yhat_upper"]
            else:
                deviation = row["yhat_lower"] - row["value"]

            if interval_width > 0:
                normalized_deviation = deviation / interval_width
                confidence = max(0.1, min(0.99, 1.0 - (normalized_deviation * 0.3)))
            else:
                confidence = 0.5

            return confidence

        df["confidence_score"] = df.apply(calculate_confidence, axis=1)

        # Calculate severity scores
        def calculate_severity(row):
            if not row["is_anomaly"]:
                return 0.0

            interval_width = row["yhat_upper"] - row["yhat_lower"]
            if row["value"] > row["yhat_upper"]:
                deviation = row["value"] - row["yhat_upper"]
            else:
                deviation = row["yhat_lower"] - row["value"]

            if interval_width > 0:
                severity = min(deviation / (interval_width * 2), 1.0)
            else:
                severity = 0.5

            return severity * row["confidence_score"]

        df["severity_score"] = df.apply(calculate_severity, axis=1)

        return df

    @bentoml.api(
        input=JSON(pydantic_model=BatchInferenceInput),
        output=JSON(pydantic_model=BatchInferenceOutput),
    )
    async def predict_batch(
        self, input_data: BatchInferenceInput
    ) -> BatchInferenceOutput:
        """Batch anomaly detection inference"""
        start_time = datetime.utcnow()

        try:
            # Load specific model version if requested
            if (
                input_data.model_version
                and input_data.model_version != self.model_version
            ):
                await self._load_model_version(input_data.model_version)

            # Extract features
            df = self._extract_features(input_data.sensor_data)

            # Run inference
            df = self._run_prophet_inference(df)

            # Classify anomalies
            df = self._classify_anomalies(df)

            # Convert to output format
            predictions = []
            for i, (_, row) in enumerate(df.iterrows()):
                prediction = AnomalyOutput(
                    id=f"pred_{i}_{int(start_time.timestamp())}",
                    timestamp=row["timestamp"].isoformat(),
                    val1=row["value"],
                    predicted_value=row["yhat"],
                    confidence_lower=row["yhat_lower"],
                    confidence_upper=row["yhat_upper"],
                    anomaly_type=row["anomaly_type"],
                    is_anomaly=bool(row["is_anomaly"]),
                    confidence_score=row["confidence_score"],
                    severity_score=row["severity_score"],
                    model_version=self.model_version,
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds()
                    * 1000,
                )
                predictions.append(prediction)

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return BatchInferenceOutput(
                predictions=predictions,
                model_version=self.model_version,
                total_predictions=len(predictions),
                anomalies_count=sum(1 for p in predictions if p.is_anomaly),
                processing_time_ms=processing_time,
                metadata={
                    "feature_extraction_enabled": True,
                    "model_type": "prophet",
                    "batch_size": len(input_data.sensor_data),
                },
            )

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return error response
            return BatchInferenceOutput(
                predictions=[],
                model_version=self.model_version or "unknown",
                total_predictions=0,
                anomalies_count=0,
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds()
                * 1000,
                metadata={"error": str(e), "status": "failed"},
            )

    @bentoml.api(
        input=JSON(pydantic_model=SensorInput),
        output=JSON(pydantic_model=AnomalyOutput),
    )
    async def predict_single(self, sensor_input: SensorInput) -> AnomalyOutput:
        """Single point anomaly detection"""
        batch_input = BatchInferenceInput(sensor_data=[sensor_input])
        batch_output = await self.predict_batch(batch_input)

        if batch_output.predictions:
            return batch_output.predictions[0]
        else:
            # Return error prediction
            return AnomalyOutput(
                id="error",
                timestamp=sensor_input.timestamp,
                val1=sensor_input.val1,
                predicted_value=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                anomaly_type="error",
                is_anomaly=False,
                confidence_score=0.0,
                severity_score=0.0,
                model_version=self.model_version or "unknown",
                processing_time_ms=0.0,
            )

    @bentoml.api(input=JSON(), output=JSON())
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the inference service"""
        return {
            "status": "healthy",
            "model_version": self.model_version,
            "model_loaded": self.model is not None,
            "timestamp": datetime.utcnow().isoformat(),
            "service_name": "anomaly_detection_inference",
        }

    @bentoml.api(input=JSON(), output=JSON())
    async def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            if self.model_version and self.model_version != "dummy:latest":
                model_ref = bentoml.models.get(self.model_version)
                return {
                    "version": self.model_version,
                    "creation_time": model_ref.creation_time.isoformat(),
                    "metadata": model_ref.info.metadata,
                    "labels": model_ref.info.labels,
                    "size_bytes": model_ref.info.size_bytes,
                }
            else:
                return {
                    "version": self.model_version,
                    "type": "dummy_model",
                    "status": "development",
                }
        except Exception as e:
            return {"error": str(e)}

    async def _load_model_version(self, model_version: str):
        """Load a specific model version"""
        try:
            self.model = bentoml.models.load_model(model_version)
            self.model_version = model_version
            logger.info(f"Switched to model version: {model_version}")
        except Exception as e:
            logger.error(f"Failed to load model version {model_version}: {e}")
            raise
