import numpy as np
import triton_python_backend_utils as pb_utils
from anomaly import SensorData, detect_anomalies

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get all input tensors
            batch_size = pb_utils.get_input_tensor_by_name(request, "val1").as_numpy().shape[0]
            sensor_data_list = []
            for i in range(batch_size):
                sensor = SensorData(
                    timestamp=pb_utils.get_input_tensor_by_name(request, "timestamp").as_numpy()[i].decode("utf-8"),
                    machine_id=pb_utils.get_input_tensor_by_name(request, "machine_id").as_numpy()[i].decode("utf-8"),
                    failure=int(pb_utils.get_input_tensor_by_name(request, "failure").as_numpy()[i]),
                    val1=int(pb_utils.get_input_tensor_by_name(request, "val1").as_numpy()[i]),
                    val2=int(pb_utils.get_input_tensor_by_name(request, "val2").as_numpy()[i]),
                    val3=int(pb_utils.get_input_tensor_by_name(request, "val3").as_numpy()[i]),
                    val4=int(pb_utils.get_input_tensor_by_name(request, "val4").as_numpy()[i]),
                    field7=int(pb_utils.get_input_tensor_by_name(request, "field7").as_numpy()[i]),
                    val5=int(pb_utils.get_input_tensor_by_name(request, "val5").as_numpy()[i]),
                    val6=int(pb_utils.get_input_tensor_by_name(request, "val6").as_numpy()[i]),
                    val7=float(pb_utils.get_input_tensor_by_name(request, "val7").as_numpy()[i]),
                )
                sensor_data_list.append(sensor)
            # Run anomaly detection
            anomalies_df = detect_anomalies(sensor_data_list)
            # Prepare output tensor (indices of anomalies)
            indices = anomalies_df.index.to_numpy().astype(np.int32)
            output_tensor = pb_utils.Tensor("anomaly_indices", indices)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses 