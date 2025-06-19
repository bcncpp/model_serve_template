# Galemind

This directory contains the model repositories for the anomaly detection models designed to be served using NVIDIA Triton Inference Server with a Python backend.

## Models Included

*   `anomaly_detector_json`: Accepts a JSON string as input and outputs a JSON string of detected anomalies.
*   `anomaly_detector_numpy`: Accepts individual NumPy arrays for each feature as input and outputs NumPy arrays.

## Serving with NVIDIA Triton

To serve these models using Triton, you will need to build a custom Docker image that includes the necessary Python dependencies (like pandas, prophet, etc.) and then run the Triton server container, mounting this `model/` directory.

### 1. Build the Custom Triton Docker Image

Navigate to your project root directory in the terminal where `Dockerfile.triton` is located. Build the Docker image:

```bash
docker build -f Dockerfile.triton -t tritonserver-custom:24.01 .
```

This command reads `Dockerfile.triton`, pulls the base Triton image, and installs the required Python libraries into a new image tagged `tritonserver-custom:24.01`.

### 2. Run the Triton Inference Server Container

In your terminal (from the project root), run the custom Docker image, mounting this `model/` directory into the container's `/models` directory. This allows Triton to find and load your models.

```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/model:/models tritonserver-custom:24.01 \
  tritonserver --model-repository=/models
```

*   `--rm`: Automatically remove the container when it exits.
*   `-p8000:8000 -p8001:8001 -p8002:8002`: Maps the container's HTTP (8000), gRPC (8001), and metrics (8002) ports to your host machine.
*   `-v $(pwd)/model:/models`: Mounts your local `./model` directory to `/models` inside the container.
*   `tritonserver --model-repository=/models`: The command to start the Triton server, specifying the model repository location.

Keep this terminal window open; the Triton server will run here.

Check the output for messages indicating that your models (`anomaly_detector_json` and `anomaly_detector_numpy`) are loading successfully and are marked as `READY`.

## Testing the Anomaly Detection Model (JSON Input)

Once the Triton server is running, you can test the `anomaly_detector_json` model using the `test_triton.py` Python script and sample data (`2020-03-19/S1.csv`).

### 1. Install Test Script Dependencies

Open a **new** terminal window (leaving the Triton server running in the first one) and navigate to your project root. Install the required Python libraries for the test script:

```bash
pip install tritonclient[http] pandas
```

### 2. Ensure Data File Exists

Make sure the sample data file `2020-03-19/S1.csv` exists in your project at that relative path from the **project root**.

### 3. Run the Test Script

Execute the test script from your project root:

```bash
python test_triton.py
```

### Expected Output

The script will connect to the running Triton server, send the data from `2020-03-19/S1.csv` to the `anomaly_detector_json` model, and print the anomaly status for each data point. You should see output similar to this:

```
Reading data from 2020-03-19/S1.csv...
Sending inference request to model anomaly_detector_json version 1...

Anomaly Status for all data points:
           timestamp        val1  anomaly
0  2020-03-19 00:00:00    270749    False
1  2020-03-19 00:02:00  28232562    False
...
(more non-anomalous data)
...
123 2020-03-19 04:22:00  95889516     True
124 2020-03-19 04:28:00  18331882    False
125 2020-03-19 04:30:00  13663442    False
126 2020-03-19 04:34:00  14307017     True
...
(more anomalous and non-anomalous data)
```

Each row shows the timestamp, the original `val1` value, and whether the model detected an `anomaly` (`True` or `False`).



