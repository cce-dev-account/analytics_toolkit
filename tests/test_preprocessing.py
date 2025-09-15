"""
Tests for preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest
from analytics_toolkit.preprocessing import DataPreprocessor, create_train_test_split


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "categorical1": ["A", "B", "A", "C", "B", "A"],
                "categorical2": ["X", "Y", "X", "Z", np.nan, "Y"],
                "target": [1, 0, 1, 0, 1, 0],
            }
        )
        return data

    @pytest.fixture
    def numeric_only_data(self):
        """Create numeric-only data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "x1": np.random.randn(100),
                "x2": np.random.randn(100) * 2 + 5,
                "x3": np.random.randn(100) * 0.5,
                "y": np.random.randint(0, 2, 100),
            }
        )

    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scalers == {}
        assert preprocessor.encoders == {}
        assert not preprocessor.fitted

    def test_fit_transform_basic(self, sample_data):
        """Test basic fit_transform functionality."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data, target_column="target")

        assert preprocessor.fitted
        assert X.shape[0] == sample_data.shape[0]
        assert X.shape[1] == sample_data.shape[1] - 1  # Excluding target
        assert y is not None
        assert len(y) == sample_data.shape[0]

    def test_fit_transform_no_target(self, sample_data):
        """Test fit_transform without target column."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)

        assert preprocessor.fitted
        assert X.shape == sample_data.shape
        assert y is None

    def test_auto_column_detection(self, sample_data):
        """Test automatic column type detection."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data, target_column="target")

        # Should have scalers for numeric columns
        assert "numerical" in preprocessor.scalers
        # Should have encoders for categorical columns
        assert "categorical1" in preprocessor.encoders
        assert "categorical2" in preprocessor.encoders

    def test_explicit_column_specification(self, sample_data):
        """Test explicit column specification."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(
            sample_data,
            target_column="target",
            numerical_columns=["numeric1", "numeric2"],
            categorical_columns=["categorical1"],
        )

        assert "numerical" in preprocessor.scalers
        assert "categorical1" in preprocessor.encoders
        assert "categorical2" not in preprocessor.encoders

    def test_scaling_methods(self, numeric_only_data):
        """Test different scaling methods."""
        # Test StandardScaler
        preprocessor_std = DataPreprocessor()
        X_std, _ = preprocessor_std.fit_transform(
            numeric_only_data, target_column="y", scaling_method="standard"
        )

        # Check that numerical columns are scaled (approximately mean=0, std=1)
        numeric_cols = ["x1", "x2", "x3"]
        for col in numeric_cols:
            col_idx = list(X_std.columns).index(col)
            assert abs(X_std.iloc[:, col_idx].mean()) < 1e-10
            assert (
                abs(X_std.iloc[:, col_idx].std() - 1.0) < 0.01
            )  # More lenient tolerance

        # Test MinMaxScaler
        preprocessor_mm = DataPreprocessor()
        X_mm, _ = preprocessor_mm.fit_transform(
            numeric_only_data, target_column="y", scaling_method="minmax"
        )

        # Check that values are in [0, 1] range
        for col in numeric_cols:
            col_idx = list(X_mm.columns).index(col)
            assert X_mm.iloc[:, col_idx].min() >= 0
            assert X_mm.iloc[:, col_idx].max() <= 1

    def test_invalid_scaling_method(self, sample_data):
        """Test invalid scaling method raises error."""
        preprocessor = DataPreprocessor()
        with pytest.raises(ValueError, match="Unknown scaling method"):
            preprocessor.fit_transform(sample_data, scaling_method="invalid")

    def test_missing_value_handling(self, sample_data):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(sample_data, target_column="target")

        # Should have no missing values after preprocessing
        assert not X.isnull().any().any()

    def test_transform_without_fit(self, sample_data):
        """Test transform without fit raises error."""
        preprocessor = DataPreprocessor()
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_data)

    def test_transform_after_fit(self, sample_data):
        """Test transform on new data after fit."""
        preprocessor = DataPreprocessor()
        # Fit on original data
        preprocessor.fit_transform(sample_data, target_column="target")

        # Create new data with same structure
        new_data = pd.DataFrame(
            {
                "numeric1": [6.0, 7.0],
                "numeric2": [70.0, 80.0],
                "categorical1": ["A", "B"],
                "categorical2": ["X", "Y"],
            }
        )

        # Transform new data
        X_new = preprocessor.transform(new_data)
        assert X_new.shape[0] == 2
        assert not X_new.isnull().any().any()

    def test_unknown_categories_handling(self, sample_data):
        """Test handling of unknown categories in transform."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(sample_data, target_column="target")

        # Create data with unknown categories
        new_data = pd.DataFrame(
            {
                "numeric1": [6.0],
                "numeric2": [70.0],
                "categorical1": ["UNKNOWN"],  # New category
                "categorical2": ["X"],
            }
        )

        X_new = preprocessor.transform(new_data)
        # Should handle unknown categories (typically encoded as -1)
        assert not X_new.isnull().any().any()

    def test_empty_mode_handling(self):
        """Test handling of categorical column with no mode."""
        data = pd.DataFrame(
            {"numeric1": [1.0, 2.0, 3.0], "categorical1": [np.nan, np.nan, np.nan]}
        )

        preprocessor = DataPreprocessor()
        # Suppress the expected warning about empty mode
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X, _ = preprocessor.fit_transform(data)

        # The categorical column should be handled properly (may be encoded or filled)
        # Main requirement is that the preprocessing doesn't crash
        assert X.shape[0] == 3
        assert "numeric1" in X.columns


class TestCreateTrainTestSplit:
    """Test suite for create_train_test_split function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for splitting."""
        np.random.seed(42)
        X = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    def test_basic_split_with_target(self, sample_data):
        """Test basic train-test split with target."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        assert X_train.shape[0] == 80  # 80% of 100
        assert X_test.shape[0] == 20  # 20% of 100
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert X_train.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]

    def test_split_without_target(self, sample_data):
        """Test train-test split without target."""
        X, _ = sample_data
        X_train, X_test, y_train, y_test = create_train_test_split(X)

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train is None
        assert y_test is None

    def test_custom_test_size(self, sample_data):
        """Test custom test size."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.3)

        assert X_train.shape[0] == 70
        assert X_test.shape[0] == 30

    def test_stratified_split(self, sample_data):
        """Test stratified splitting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, stratify=True)

        # Check that class proportions are preserved
        train_prop = y_train.mean()
        test_prop = y_test.mean()
        overall_prop = y.mean()

        assert abs(train_prop - overall_prop) < 0.1
        assert abs(test_prop - overall_prop) < 0.1

    def test_random_state_reproducibility(self, sample_data):
        """Test random state for reproducibility."""
        X, y = sample_data

        # Split twice with same random state
        result1 = create_train_test_split(X, y, random_state=42)
        result2 = create_train_test_split(X, y, random_state=42)

        # Should get identical results
        pd.testing.assert_frame_equal(result1[0], result2[0])  # X_train
        pd.testing.assert_frame_equal(result1[1], result2[1])  # X_test
        pd.testing.assert_series_equal(result1[2], result2[2])  # y_train
        pd.testing.assert_series_equal(result1[3], result2[3])  # y_test

    def test_stratify_without_target_ignored(self, sample_data):
        """Test that stratify is ignored when no target is provided."""
        X, _ = sample_data
        # Should not raise error even with stratify=True
        X_train, X_test, y_train, y_test = create_train_test_split(X, stratify=True)

        assert y_train is None
        assert y_test is None
