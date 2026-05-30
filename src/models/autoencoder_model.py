# src/models/autoencoder_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation.metrics import evaluate, log_result
from src.visualization.plots import save_all
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim: int, epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.train_losses = []

    def fit(self, X_clean) -> None:
        """Train ONLY on non-fraud data."""
        self.model = Autoencoder(input_dim=self.input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        data = X_clean.values if hasattr(X_clean, "values") else np.array(X_clean)
        X_tensor = torch.FloatTensor(data)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

    def anomaly_scores(self, X) -> np.ndarray:
        """Reconstruction error per sample (MSE)."""
        self.model.eval()
        data = X.values if hasattr(X, "values") else np.array(X)
        X_tensor = torch.FloatTensor(data)
        with torch.no_grad():
            recon = self.model(X_tensor)
        errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
        return errors

    def save_training_curve(self) -> None:
        os.makedirs("outputs/plots", exist_ok=True)
        plt.figure()
        plt.plot(self.train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Autoencoder Training Loss")
        plt.tight_layout()
        plt.savefig("outputs/plots/autoencoder_training_loss.png", dpi=150)
        plt.close()

def run_autoencoder(split):
    X_train_clean = split.X_train[split.y_train == 0]
    print(f"Training autoencoder on {len(X_train_clean)} non-fraud samples...")

    detector = AutoencoderAnomalyDetector(input_dim=split.X_train.shape[1], epochs=50)
    detector.fit(X_train_clean)
    detector.save_training_curve()

    y_score_val = detector.anomaly_scores(split.X_val)
    y_score_test = detector.anomaly_scores(split.X_test)

    val_result = evaluate(split.y_val.values, y_score_val, model_name="autoencoder_val")
    result = evaluate(split.y_test.values, y_score_test,
                      model_name="autoencoder", threshold=val_result.threshold)
    log_result(result)
    save_all(split.y_test.values, y_score_test, result.confusion_matrix, "autoencoder")
    return result
