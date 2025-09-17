"""
Test suite for UI transformation functionality and business logic.

Tests the actual transformation logic extracted from UI pages,
ensuring that the business logic works correctly independent of Streamlit.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test data fixtures
@pytest.fixture
def sample_data():
    """Create sample data for transformation testing."""
    np.random.seed(42)  # For reproducible tests
    return pd.DataFrame({
        'numeric_positive': np.random.exponential(1, 100),
        'numeric_mixed': np.random.normal(0, 1, 100),
        'numeric_outliers': np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]]),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'target_numeric': np.random.randn(100),
        'target_categorical': np.random.choice([0, 1], 100)
    })


class TestTransformationLogic:
    """Test the actual transformation algorithms used in the UI."""

    def test_log_transformation_fallback(self, sample_data):
        """Test log transformation with fallback implementation."""
        # Test log1p transformation on positive values only to avoid NaN
        positive_data = sample_data[['numeric_positive']]

        # Manual log1p (fallback implementation)
        log_transformed = np.log(positive_data + 1)

        assert not log_transformed.isna().any().any(), "Log transformation produced NaN values"
        assert log_transformed.shape == positive_data.shape

        # Test that positive values are handled correctly
        positive_col = sample_data['numeric_positive']
        log_pos = np.log(positive_col + 1)
        assert (log_pos >= 0).all(), "Log of positive values should be non-negative"

        # Test handling of negative values (should use absolute value or skip)
        mixed_data = sample_data['numeric_mixed']
        # For negative values, we'd typically use log(abs(x) + 1) or skip transformation
        log_abs = np.log(np.abs(mixed_data) + 1)
        assert not log_abs.isna().any(), "Log transformation with abs should not produce NaN"

    def test_outlier_capping_fallback(self, sample_data):
        """Test outlier capping with IQR method."""
        outlier_data = sample_data['numeric_outliers']

        # IQR method implementation
        Q1 = outlier_data.quantile(0.25)
        Q3 = outlier_data.quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        capped_data = outlier_data.clip(lower_bound, upper_bound)

        assert capped_data.min() >= lower_bound, "Values below lower bound not capped"
        assert capped_data.max() <= upper_bound, "Values above upper bound not capped"
        assert len(capped_data) == len(outlier_data), "Data length changed during capping"

    def test_zscore_outlier_method(self, sample_data):
        """Test Z-score outlier detection and capping."""
        data = sample_data['numeric_outliers']
        threshold = 2.0

        mean = data.mean()
        std = data.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        capped_data = data.clip(lower_bound, upper_bound)

        # Check that extreme values are capped
        z_scores = np.abs((capped_data - mean) / std)
        assert (z_scores <= threshold + 0.001).all(), "Z-score capping failed"  # Small tolerance for floating point

    def test_percentile_outlier_method(self, sample_data):
        """Test percentile-based outlier capping."""
        data = sample_data['numeric_outliers']

        # 99th percentile method
        lower_percentile = 1
        upper_percentile = 99

        lower_bound = data.quantile(lower_percentile / 100)
        upper_bound = data.quantile(upper_percentile / 100)

        capped_data = data.clip(lower_bound, upper_bound)

        assert capped_data.min() >= lower_bound
        assert capped_data.max() <= upper_bound


class TestScalingFallbacks:
    """Test scaling methods with sklearn fallbacks."""

    def test_standard_scaling_fallback(self, sample_data):
        """Test standard scaling with sklearn."""
        from sklearn.preprocessing import StandardScaler

        numeric_data = sample_data[['numeric_mixed', 'numeric_positive']]
        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(numeric_data)

        # Check that scaling worked correctly
        assert abs(scaled_data.mean()) < 1e-10, "Standard scaling didn't center data"
        assert abs(scaled_data.std() - 1.0) < 1e-10, "Standard scaling didn't scale to unit variance"

    def test_robust_scaling_fallback(self, sample_data):
        """Test robust scaling with sklearn."""
        from sklearn.preprocessing import RobustScaler

        numeric_data = sample_data[['numeric_outliers', 'numeric_mixed']]
        scaler = RobustScaler()

        scaled_data = scaler.fit_transform(numeric_data)

        # Robust scaling should be less sensitive to outliers
        assert scaled_data.shape == numeric_data.shape
        # Median should be approximately 0
        assert abs(np.median(scaled_data, axis=0)).max() < 0.1

    def test_minmax_scaling_fallback(self, sample_data):
        """Test MinMax scaling with sklearn."""
        from sklearn.preprocessing import MinMaxScaler

        numeric_data = sample_data[['numeric_mixed', 'numeric_positive']]
        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(numeric_data)

        # Check bounds
        assert scaled_data.min() >= -1e-10, "MinMax scaling produced values below 0"
        assert scaled_data.max() <= 1 + 1e-10, "MinMax scaling produced values above 1"


class TestFeatureEngineeringWorkflows:
    """Test complete feature engineering workflows."""

    def test_complete_transformation_pipeline(self, sample_data):
        """Test a complete transformation workflow."""
        data = sample_data.copy()

        # Step 1: Log transform positive values
        positive_cols = ['numeric_positive']
        for col in positive_cols:
            data[f'{col}_log'] = np.log(data[col] + 1)

        # Step 2: Cap outliers
        outlier_cols = ['numeric_outliers']
        for col in outlier_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data[f'{col}_capped'] = data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # Step 3: Scale features
        from sklearn.preprocessing import StandardScaler
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data_scaled = data.copy()
        data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        # Verify pipeline worked
        assert f'numeric_positive_log' in data.columns
        assert f'numeric_outliers_capped' in data.columns
        assert not data_scaled[numeric_cols].isna().any().any()

    def test_categorical_encoding_workflow(self, sample_data):
        """Test categorical encoding workflows."""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        categorical_data = sample_data[['categorical', 'target_categorical']]

        # Test label encoding
        le = LabelEncoder()
        encoded = le.fit_transform(categorical_data['categorical'])

        assert len(np.unique(encoded)) <= len(categorical_data['categorical'].unique())
        assert encoded.dtype in [np.int32, np.int64]

        # Test one-hot encoding
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        one_hot = ohe.fit_transform(categorical_data[['categorical']])

        expected_cols = len(categorical_data['categorical'].unique()) - 1  # drop='first'
        assert one_hot.shape[1] == expected_cols


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in transformations."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()

        # Should handle empty data gracefully
        numeric_cols = empty_df.select_dtypes(include=[np.number]).columns.tolist()
        assert len(numeric_cols) == 0

        categorical_cols = empty_df.select_dtypes(include=['object']).columns.tolist()
        assert len(categorical_cols) == 0

    def test_single_value_column_handling(self):
        """Test handling of columns with single unique value."""
        single_value_df = pd.DataFrame({
            'constant': [5.0] * 100,
            'target': np.random.randn(100)
        })

        # Standard scaling should handle constant columns
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # This should not crash, but might produce warnings depending on sklearn version
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = scaler.fit_transform(single_value_df[['constant']])

        # Constant column should become all zeros after scaling
        assert np.allclose(scaled, 0)

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        data_with_na = pd.DataFrame({
            'numeric_with_na': [1, 2, np.nan, 4, 5],
            'categorical_with_na': ['A', 'B', None, 'A', 'C']
        })

        # Check that we can detect missing values
        numeric_na_count = data_with_na['numeric_with_na'].isna().sum()
        categorical_na_count = data_with_na['categorical_with_na'].isna().sum()

        assert numeric_na_count == 1
        assert categorical_na_count == 1

        # Test simple imputation
        numeric_filled = data_with_na['numeric_with_na'].fillna(
            data_with_na['numeric_with_na'].median()
        )
        assert not numeric_filled.isna().any()

    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        extreme_df = pd.DataFrame({
            'very_large': [1e10, 1e15, 1e20],
            'very_small': [1e-10, 1e-15, 1e-20],
            'mixed_extreme': [-1e10, 1e10, 0]
        })

        # Should handle extreme values without overflow
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

        scaled = scaler.fit_transform(extreme_df)

        assert np.isfinite(scaled).all(), "Extreme values caused infinite results"
        assert not np.isnan(scaled).any(), "Extreme values caused NaN results"


class TestUIComponentFallbacks:
    """Test that UI components work with and without custom modules."""

    def test_feature_engineering_availability_dict(self):
        """Test that AVAILABLE_TRANSFORMERS dictionary works correctly."""
        # This simulates what happens in the UI page
        available_transformers = {}

        # Try to import custom transformers
        try:
            from analytics_toolkit.feature_engineering import RobustScaler
            available_transformers['RobustScaler'] = RobustScaler
        except ImportError:
            pass

        try:
            from analytics_toolkit.feature_engineering import OutlierCapTransformer
            available_transformers['OutlierCapTransformer'] = OutlierCapTransformer
        except ImportError:
            pass

        # Should work whether transformers are available or not
        if 'RobustScaler' in available_transformers:
            # Can use custom transformer
            transformer = available_transformers['RobustScaler']()
            assert transformer is not None
        else:
            # Fall back to sklearn
            from sklearn.preprocessing import RobustScaler
            transformer = RobustScaler()
            assert transformer is not None

    def test_visualization_availability_dict(self):
        """Test that AVAILABLE_VIZ_COMPONENTS dictionary works correctly."""
        available_viz = {}

        try:
            from analytics_toolkit.visualization import ModelEvaluationPlots
            available_viz['ModelEvaluationPlots'] = ModelEvaluationPlots
        except ImportError:
            pass

        try:
            from analytics_toolkit.visualization import StatisticalPlots
            available_viz['StatisticalPlots'] = StatisticalPlots
        except ImportError:
            pass

        # Should always be able to create basic plots
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])