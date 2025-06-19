import os
import csv
import random
from datetime import datetime, timedelta

# Configuration
start_date = datetime.strptime("12/2/2020", "%d/%m/%Y")
end_date = datetime.strptime("5/10/2025", "%d/%m/%Y")
measurement_interval = timedelta(seconds=120)
records_per_day = 720  # 24h * 60min / 2
device_prefixes = ["Z2", "S1", "Y1"]

thresholds = {
    "val1": 105234552,
    "val2": 392,
    "val3": 24929,
    "val4": 529,
    "val5": 3,
    "val6": 4,
    "val7": 70.0,
}


def _generate_row(timestamp, machine_id, failure):
    if failure:
        val1 = random.randint(thresholds["val1"] + 1, 250_000_000)
        val2 = random.randint(thresholds["val2"] + 1, 10000)
        val3 = random.randint(thresholds["val3"] + 1, 50000)
        val4 = random.randint(thresholds["val4"] + 1, 1000)
        val5 = random.randint(thresholds["val5"] + 1, 8)
        val6 = random.randint(thresholds["val6"] + 1, 10)
        val7 = round(random.uniform(thresholds["val7"] + 0.1, 100.0), 1)
    else:
        val1 = random.randint(10_000_000, thresholds["val1"])
        val2 = random.randint(0, thresholds["val2"])
        val3 = random.randint(0, thresholds["val3"])
        val4 = random.randint(0, thresholds["val4"])
        val5 = random.randint(0, thresholds["val5"])
        val6 = random.randint(0, thresholds["val6"])
        val7 = round(random.uniform(0.0, thresholds["val7"]), 1)

    field7 = random.randint(9, 12)

    return [
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        machine_id,
        failure,
        val1,
        val2,
        val3,
        val4,
        field7,
        val5,
        val6,
        val7,
    ]


def _write_csv_for_device(day_dir, device_prefix, machine_id, day_start):
    failure_indices = set(random.sample(range(records_per_day), k=random.randint(2, 6)))
    file_path = os.path.join(day_dir, f"{device_prefix}.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "machine_id",
                "failure",
                "val1",
                "val2",
                "val3",
                "val4",
                "field7",
                "val5",
                "val6",
                "val7",
            ]
        )
        for i in range(records_per_day):
            timestamp = day_start + i * measurement_interval
            failure = 1 if i in failure_indices else 0
            row = _generate_row(timestamp, machine_id, failure)
            writer.writerow(row)


def generate_syn_data(path: str):
    os.makedirs(path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(path)
    current_date = start_date
    machine_counter = 1
    while current_date <= end_date:
        day_str = current_date.strftime("%Y-%m-%d")
        day_dir = day_str
        os.makedirs(day_dir, exist_ok=True)
        day_start = datetime(current_date.year, current_date.month, current_date.day)
        for prefix in device_prefixes:
            machine_id = f"{prefix}{machine_counter:04d}"
            _write_csv_for_device(day_dir, prefix, machine_id, day_start)
            machine_counter += 1
        print(f"{day_str} - CSVs written")
        current_date += timedelta(days=1)
        os.chdir(cwd)
