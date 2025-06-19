import json
import numpy as np
import triton_python_backend_utils as pb_utils
from anomaly import SensorData, detect_anomalies
from typing import List, Dict, Any
import pandas as pd

class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the model.
        """
        pass

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """
        Perform inference on a batch of requests.
        """
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            # Get the input JSON string
            input_tensor: pb_utils.Tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_json: str = input_tensor.as_numpy()[0].decode("utf-8")
            records: List[Dict[str, Any]] = json.loads(input_json)

            # Convert to SensorData
            sensor_data_list: List[SensorData] = [SensorData(**row) for row in records]

            # Run anomaly detection
            anomalies_df: pd.DataFrame = detect_anomalies(sensor_data_list)

            # Convert anomalies to JSON string for output
            anomalies_json: str = anomalies_df.to_json(orient="records", date_format="iso")

            # Create output tensor
            output_tensor: pb_utils.Tensor = pb_utils.Tensor("OUTPUT", np.array([anomalies_json.encode("utf-8")], dtype=object))

            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses 