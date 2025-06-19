import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec, step, Parameter, card
from model.predictive_model import LSTMModel


class PredictiveMaintenanceFlow(FlowSpec):
    data_path = Parameter("data_path", default="sensor_data.csv")
    seq_len = Parameter("seq_len", default=20)
    epochs = Parameter("epochs", default=10)

    @step
    def start(self):
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        self.next(self.preprocess)

    @step
    def preprocess(self):
        print("Preprocessing...")

        self.df.sort_values(by=["machine_id", "timestamp"], inplace=True)
        self.df["machine_id"] = self.df["machine_id"].astype("category").cat.codes

        features = [
            "machine_id",
            "val1",
            "val2",
            "val3",
            "val4",
            "val5",
            "val6",
            "val7",
        ]
        target = "failure"

        scaler = MinMaxScaler()
        self.df[features] = scaler.fit_transform(self.df[features])

        def create_sequences(data, labels, seq_length):
            Xs, ys = [], []
            for i in range(len(data) - seq_length):
                Xs.append(data[i : i + seq_length])
                ys.append(labels[i + seq_length])
            return np.array(Xs), np.array(ys)

        self.X_all, self.y_all = [], []
        for _, group in self.df.groupby("machine_id"):
            X_seq, y_seq = create_sequences(
                group[features].values, group[target].values, self.seq_len
            )
            self.X_all.append(X_seq)
            self.y_all.append(y_seq)

        X = np.concatenate(self.X_all)
        y = np.concatenate(self.y_all)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).unsqueeze(1)

        self.input_size = self.X_train.shape[2]
        self.next(self.train)

    @step
    def train(self):
        print("Training...")
        self.model = LSTMModel(input_size=self.input_size)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(self.X_train)
            loss = self.loss_fn(pred, self.y_train)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.next(self.evaluate)

    @step
    @card  # Add a simple results card
    def evaluate(self):
        print("Evaluating...")
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.X_test)
            acc = ((pred > 0.5).float() == self.y_test).float().mean()
            print(f"Test Accuracy: {acc:.2f}")
            self.accuracy = acc.item()
        self.next(self.end)

    @step
    def end(self):
        print(f"Final Accuracy: {self.accuracy:.2f}")


if __name__ == "__main__":
    PredictiveMaintenanceFlow()
