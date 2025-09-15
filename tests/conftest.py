"""
Shared fixtures and configuration for tests.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def simple_regression_data():
    """Create simple regression dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_coef = np.array([1.5, -2.0, 0.5])
    y = X @ true_coef + 0.1 * np.random.randn(100)
    return X, y, true_coef


@pytest.fixture
def simple_classification_data():
    """Create simple classification dataset for testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def large_regression_data():
    """Create larger regression dataset for performance testing."""
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def pandas_mixed_data():
    """Create pandas DataFrame with mixed data types."""
    np.random.seed(42)
    n_samples = 150

    data = pd.DataFrame(
        {
            "numeric1": np.random.randn(n_samples),
            "numeric2": np.random.randn(n_samples) * 2 + 5,
            "categorical1": np.random.choice(["A", "B", "C"], n_samples),
            "categorical2": np.random.choice(["X", "Y", "Z"], n_samples),
            "binary": np.random.choice([0, 1], n_samples),
            "target": np.random.randn(n_samples),
        }
    )

    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=10, replace=False)
    data.loc[missing_indices, "numeric1"] = np.nan
    data.loc[missing_indices[:5], "categorical2"] = np.nan

    return data


@pytest.fixture
def pytorch_device():
    """Get appropriate PyTorch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def categorical_data():
    """Create dataset with categorical variables for encoding tests."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "category_high_card": np.random.choice(
                [f"cat_{i}" for i in range(20)], n_samples
            ),
            "category_low_card": np.random.choice(["A", "B", "C"], n_samples),
            "category_binary": np.random.choice(["Yes", "No"], n_samples),
            "numeric": np.random.randn(n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )

    return data


@pytest.fixture
def small_sample_data():
    """Create very small dataset for edge case testing."""
    return pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})


@pytest.fixture
def perfect_separation_data():
    """Create dataset with perfect linear separation for testing edge cases."""
    np.random.seed(42)
    X = np.array(
        [[1, 1], [1, 2], [2, 1], [2, 2], [-1, -1], [-1, -2], [-2, -1], [-2, -2]]
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return X, y


@pytest.fixture
def multicollinear_data():
    """Create dataset with multicollinearity issues."""
    np.random.seed(42)
    n_samples = 100

    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = 2 * x1 + 3 * x2 + 0.01 * np.random.randn(n_samples)  # Nearly collinear

    X = np.column_stack([x1, x2, x3])
    y = x1 + x2 + np.random.randn(n_samples) * 0.1

    return X, y


@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory):
    """Create temporary directory for test data files."""
    return tmp_path_factory.mktemp("test_data")


# Configure pytest to handle warnings
@pytest.fixture(autouse=True)
def configure_warnings():
    """Configure warning handling for tests."""
    import warnings

    # Filter out specific warnings that are expected in tests
    warnings.filterwarnings("ignore", message="torch.qr is deprecated")
    warnings.filterwarnings("ignore", message="torch.triangular_solve is deprecated")
    warnings.filterwarnings(
        "ignore", message="Converting a tensor with requires_grad=True"
    )
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered"
    )


# Set random seeds for reproducibility
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
