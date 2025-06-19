import typer
from typing import Final
import typer
from src.data_generation.syntetic import generate_syn_data
import pandas as pd
from src.model.anomaly import SensorData, detect_anomalies
from datetime import datetime
import os

app = typer.Typer()
#Get cvs 2020-03-19/S1.csv




@app.command("print")
def print_csv(csv_path: str = "2020-03-19/S1.csv"):
    """
    Print the contents of a CSV file. If the file is not found, print a clear error message and return immediately.
    This prevents recursive error messages, file name too long errors, and invalid file path recursion.
    """
    # Guard clause: prevent recursion if an error message is passed as the file path
    if "File not found" in csv_path:
        print("Invalid file path provided.")
        return
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    try:
        df = pd.read_csv(csv_path)
        print(df)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

    

@app.command()
def generate(path: str):
    if not path:
        raise ValueError(f"Path invalid {path}")
    generate_syn_data(path=path)

@app.command()
def parse_sensor_data(
    csv_path: str = "2020-03-19/S1.csv",
    timestamp_key: str = "timestamp",
    machine_id_key: str = "machine_id",
    failure_key: str = "failure",
    val1_key: str = "val1",
    val2_key: str = "val2",
    val3_key: str = "val3",
    val4_key: str = "val4",
    field7_key: str = "field7",
    val5_key: str = "val5",
    val6_key: str = "val6",
    val7_key: str = "val7",
):
    """
    Reads the specified CSV file, parses each row into a SensorData object,
    and prints the number of records. This prepares the data for anomaly detection.
    """
    import csv
    sensor_data_list = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert types as required by SensorData
            sensor = SensorData(
                timestamp=datetime.strptime(row[timestamp_key], "%Y-%m-%d %H:%M:%S"),
                machine_id=row[machine_id_key],
                failure=int(row[failure_key]),
                val1=int(row[val1_key]),
                val2=int(row[val2_key]),
                val3=int(row[val3_key]),
                val4=int(row[val4_key]),
                field7=int(row[field7_key]),
                val5=int(row[val5_key]),
                val6=int(row[val6_key]),
                val7=float(row[val7_key]),
            )
            sensor_data_list.append(sensor)
    print(f"Parsed {len(sensor_data_list)} SensorData records from {csv_path}")
    return sensor_data_list

@app.command("detect-anomaly")
def detect_anomaly(csv_path: str = "2020-03-19/S1.csv"):
    """
    Runs anomaly detection on the specified CSV file using Prophet.
    Prints the detected anomalies (timestamp and value).
    """
    # Parse the CSV into SensorData objects
    sensor_data_list = parse_sensor_data(csv_path)
    if not sensor_data_list:
        print("No sensor data found or failed to parse CSV.")
        return
    # Run anomaly detection
    anomalies = detect_anomalies(sensor_data_list)
    if anomalies.empty:
        print("No anomalies detected.")
    else:
        print("Anomalies detected:")
        print(anomalies[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']])

if __name__ == "__main__":
    app()
