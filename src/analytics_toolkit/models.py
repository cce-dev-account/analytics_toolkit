"""Machine learning models and PyTorch utilities."""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""

    def __init__(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class SimpleNN(nn.Module):
    """Simple fully connected neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "tanh":
                self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ModelTrainer:
    """Utility class for training PyTorch models."""

    def __init__(self, model: nn.Module, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        criterion: nn.Module | None = None,
        optimizer_class: Any = optim.Adam,
        early_stopping_patience: int | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model."""

        if criterion is None:
            criterion = nn.MSELoss()  # Default for regression

        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(
                            self.device
                        )
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                self.history["val_loss"].append(avg_val_loss)

                # Early stopping
                if early_stopping_patience is not None:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        return self.history

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)
