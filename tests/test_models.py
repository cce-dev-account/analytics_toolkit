"""
Tests for models module.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from analytics_toolkit.models import ModelTrainer, SimpleNN, TabularDataset
from torch.utils.data import DataLoader


class TestTabularDataset:
    """Test suite for TabularDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        return X, y

    @pytest.fixture
    def pandas_data(self):
        """Create pandas DataFrame and Series."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randn(50))
        return X, y

    def test_numpy_input_with_target(self, sample_data):
        """Test TabularDataset with numpy arrays."""
        X, y = sample_data
        dataset = TabularDataset(X, y)

        assert len(dataset) == 100
        assert dataset.X.shape == (100, 5)
        assert dataset.y.shape == (100,)
        assert isinstance(dataset.X, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.float32

    def test_numpy_input_without_target(self, sample_data):
        """Test TabularDataset without target."""
        X, _ = sample_data
        dataset = TabularDataset(X)

        assert len(dataset) == 100
        assert dataset.X.shape == (100, 5)
        assert dataset.y is None

    def test_pandas_input(self, pandas_data):
        """Test TabularDataset with pandas input."""
        X, y = pandas_data
        dataset = TabularDataset(X, y)

        assert len(dataset) == 50
        assert dataset.X.shape == (50, 3)
        assert dataset.y.shape == (50,)
        assert isinstance(dataset.X, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)

    def test_getitem_with_target(self, sample_data):
        """Test __getitem__ method with target."""
        X, y = sample_data
        dataset = TabularDataset(X, y)

        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], torch.Tensor)
        assert item[0].shape == (5,)
        assert item[1].shape == ()

    def test_getitem_without_target(self, sample_data):
        """Test __getitem__ method without target."""
        X, _ = sample_data
        dataset = TabularDataset(X)

        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)

    def test_dataloader_integration(self, sample_data):
        """Test integration with PyTorch DataLoader."""
        X, y = sample_data
        dataset = TabularDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        for batch_X, batch_y in dataloader:
            assert batch_X.shape == (10, 5)
            assert batch_y.shape == (10,)
            break


class TestSimpleNN:
    """Test suite for SimpleNN class."""

    def test_initialization(self):
        """Test SimpleNN initialization."""
        model = SimpleNN(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1,
            dropout_rate=0.3,
            activation="relu",
        )

        assert isinstance(model, nn.Module)
        assert len(model.layers) == 7  # 2 linear + 2 activation + 2 dropout + 1 output
        assert isinstance(model.layers[0], nn.Linear)
        assert isinstance(model.layers[1], nn.ReLU)
        assert isinstance(model.layers[2], nn.Dropout)

    def test_initialization_with_tanh(self):
        """Test SimpleNN with tanh activation."""
        model = SimpleNN(
            input_size=5, hidden_sizes=[32], output_size=1, activation="tanh"
        )

        assert isinstance(model.layers[1], nn.Tanh)

    def test_forward_pass(self):
        """Test forward pass."""
        model = SimpleNN(input_size=10, hidden_sizes=[64, 32], output_size=1)

        x = torch.randn(5, 10)  # batch_size=5, input_size=10
        output = model(x)

        assert output.shape == (5, 1)
        assert not torch.isnan(output).any()

    def test_forward_pass_no_hidden_layers(self):
        """Test forward pass with no hidden layers."""
        model = SimpleNN(input_size=10, hidden_sizes=[], output_size=3)

        x = torch.randn(2, 10)
        output = model(x)

        assert output.shape == (2, 3)

    def test_different_output_sizes(self):
        """Test different output sizes."""
        # Binary classification
        model_binary = SimpleNN(5, [10], 1)
        x = torch.randn(3, 5)
        out_binary = model_binary(x)
        assert out_binary.shape == (3, 1)

        # Multi-class classification
        model_multi = SimpleNN(5, [10], 5)
        out_multi = model_multi(x)
        assert out_multi.shape == (3, 5)

    def test_dropout_in_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        model = SimpleNN(10, [64], 1, dropout_rate=0.5)
        x = torch.randn(100, 10)

        # In training mode
        model.train()
        out_train1 = model(x)
        out_train2 = model(x)
        # Should be different due to dropout
        assert not torch.allclose(out_train1, out_train2)

        # In eval mode
        model.eval()
        out_eval1 = model(x)
        out_eval2 = model(x)
        # Should be identical (no dropout)
        assert torch.allclose(out_eval1, out_eval2)


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return SimpleNN(input_size=2, hidden_sizes=[10], output_size=1)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for training."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)  # Simple linear relationship
        return TabularDataset(X, y)

    @pytest.fixture
    def data_loaders(self, sample_dataset):
        """Create train and validation data loaders."""
        train_size = int(0.8 * len(sample_dataset))
        val_size = len(sample_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            sample_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        return train_loader, val_loader

    def test_initialization(self, simple_model):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(simple_model)

        assert trainer.model is not None
        assert trainer.device in ["cpu", "cuda"]
        assert "train_loss" in trainer.history
        assert "val_loss" in trainer.history

    def test_device_selection(self, simple_model):
        """Test device selection."""
        # Force CPU
        trainer_cpu = ModelTrainer(simple_model, device="cpu")
        assert trainer_cpu.device == "cpu"

        # Auto selection
        trainer_auto = ModelTrainer(simple_model, device=None)
        assert trainer_auto.device in ["cpu", "cuda"]

    def test_basic_training(self, simple_model, data_loaders):
        """Test basic training functionality."""
        train_loader, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        history = trainer.train(
            train_loader=train_loader, val_loader=val_loader, epochs=5, verbose=False
        )

        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        assert all(isinstance(loss, float) for loss in history["train_loss"])
        assert all(isinstance(loss, float) for loss in history["val_loss"])

    def test_training_without_validation(self, simple_model, data_loaders):
        """Test training without validation loader."""
        train_loader, _ = data_loaders
        trainer = ModelTrainer(simple_model)

        history = trainer.train(train_loader=train_loader, epochs=3, verbose=False)

        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 0

    def test_custom_criterion_and_optimizer(self, simple_model, data_loaders):
        """Test custom criterion and optimizer."""
        train_loader, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer_class=optim.SGD,
            learning_rate=0.01,
            verbose=False,
        )

        assert len(history["train_loss"]) == 2

    def test_early_stopping(self, simple_model, data_loaders):
        """Test early stopping functionality."""
        train_loader, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,  # Large number
            early_stopping_patience=3,
            verbose=False,
        )

        # Should stop early (much less than 100 epochs)
        assert len(history["train_loss"]) < 100

    def test_prediction(self, simple_model, data_loaders):
        """Test prediction functionality."""
        train_loader, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        # Train briefly
        trainer.train(train_loader=train_loader, epochs=2, verbose=False)

        # Make predictions
        predictions = trainer.predict(val_loader)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
        assert not np.isnan(predictions).any()

    def test_loss_decreases_during_training(self, simple_model, data_loaders):
        """Test that loss generally decreases during training."""
        train_loader, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,
            learning_rate=0.01,
            verbose=False,
        )

        # Loss should generally decrease
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        assert final_loss < initial_loss

    def test_history_tracking(self, simple_model, data_loaders):
        """Test that training history is properly tracked."""
        train_loader, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        # First training run
        history1 = trainer.train(train_loader=train_loader, epochs=3, verbose=False)

        # Record state after first training
        first_train_length = len(trainer.history["train_loss"])

        # Second training run (should append to history)
        history2 = trainer.train(train_loader=train_loader, epochs=2, verbose=False)

        # History should accumulate
        assert len(trainer.history["train_loss"]) == first_train_length + 2
        # The returned history contains the complete history, not just the current run
        assert len(history1["train_loss"]) >= 3  # At least the epochs we ran
        assert len(history2["train_loss"]) >= first_train_length + 2

    def test_model_state_changes(self, simple_model, data_loaders):
        """Test that model parameters change during training."""
        train_loader, _ = data_loaders
        trainer = ModelTrainer(simple_model)

        # Get initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Train for a few epochs
        trainer.train(train_loader=train_loader, epochs=5, verbose=False)

        # Check that parameters have changed
        final_params = list(trainer.model.parameters())

        for initial, final in zip(initial_params, final_params, strict=False):
            assert not torch.allclose(
                initial, final
            ), "Parameters should change during training"

    def test_prediction_without_training(self, simple_model, data_loaders):
        """Test prediction on untrained model."""
        _, val_loader = data_loaders
        trainer = ModelTrainer(simple_model)

        # Should be able to predict even without training
        predictions = trainer.predict(val_loader)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
