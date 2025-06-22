import polars as pl
import torch
import torch.nn as nn
import numpy as np
import glob
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=32):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim, batch_first=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = self.relu(h_n.squeeze(0))
        return out


class Decoder(nn.Module):
    def __init__(self, embedding_dim=32, seq_len=20, n_features=8):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True
        )
        self.fc = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


class Autoencoder(nn.Module):
    def __init__(self, seq_len=20, n_features=8, embedding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim)
        self.decoder = Decoder(embedding_dim, seq_len, n_features)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_and_prepare_csvs(files: List[str], feature_cols: List[str], seq_len: int):
    dfs = [pl.read_csv(file) for file in files]
    df = pl.concat(dfs)
    df = df.sort(["timestamp", "machine_id"])
    df_features = df.select(feature_cols)
    data_np = df_features.to_numpy().astype(np.float32)
    num_sequences = len(data_np) // seq_len
    data_np = data_np[: num_sequences * seq_len]
    data_np = data_np.reshape(num_sequences, seq_len, len(feature_cols))
    return torch.tensor(data_np)


def df_to_tensor(df, seq_len=20, feature_cols=None):
    if feature_cols is None:
        feature_cols = [
            "val1",
            "val2",
            "val3",
            "val4",
            "field7",
            "val5",
            "val6",
            "val7",
        ]
    data = df[feature_cols].values.astype(np.float32)
    num_sequences = len(data) // seq_len
    data = data[: num_sequences * seq_len]
    data = data.reshape(num_sequences, seq_len, len(feature_cols))
    return torch.tensor(data)


def main():
    files = glob.glob("./data/*.csv")
    feature_cols = ["val1", "val2", "val3", "val4", "field7", "val5", "val6", "val7"]
    seq_len = 20
    x = load_and_prepare_csvs(files, feature_cols, seq_len).to(device)

    n_features = len(feature_cols)
    model = Autoencoder(seq_len=seq_len, n_features=n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, x)
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item()}")

    torch.save(model.state_dict(), "autoencoder_lstm.pth")
    print("Model saved as autoencoder_lstm.pth")


if __name__ == "__main__":
    main()
