import torch
import torch.nn as nn

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, 20, 1]
        out, (h_n, c_n) = self.lstm(x)  # h_n: [1, batch_size, 32]
        out = self.relu(h_n.squeeze(0))  # [batch_size, 32]
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 20)

    def forward(self, x):
        # x: [batch_size, 32]
        x = x.unsqueeze(1).repeat(1, 1, 1)  # [batch_size, 1, 32]
        out, _ = self.lstm(x)              # [batch_size, 1, 32]
        out = self.fc(out.squeeze(1))     # [batch_size, 20]
        return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and move model to device
model = Autoencoder().to(device)
# Instantiate model
model = Autoencoder().to(device)

# Example input
x = torch.randn(64, 20, 1).to(device)

# Training step (example)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

model.train()
output = model(x)
loss = criterion(output, x.squeeze(-1))
loss.backward()
optimizer.step()
optimizer.zero_grad()
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
torch.save(model, "autoencoder_full.pt")
