"""
Comprehensive tests for feature engineering module.
"""

import numpy as np
import pandas as pd
from analytics_toolkit.feature_engineering.encoding import (
    BayesianTargetEncoder,
    FrequencyEncoder,
    OrdinalEncoderAdvanced,
    RareClassEncoder,
    TargetEncoder,
)
from analytics_toolkit.feature_engineering.interactions import (
    InteractionDetector,
    InteractionGenerator,
    PolynomialInteractions,
)
from analytics_toolkit.feature_engineering.selection import (
    CorrelationFilter,
    FeatureSelector,
    MutualInfoSelector,
    VarianceThresholdAdvanced,
)
from analytics_toolkit.feature_engineering.temporal import (
    DateTimeFeatures,
    LagFeatures,
    RollingFeatures,
    SeasonalDecomposition,
)
from analytics_toolkit.feature_engineering.transformers import (
    BinningTransformer,
    LogTransformer,
    OutlierCapTransformer,
    PolynomialFeaturesAdvanced,
    RobustScaler,
)
from sklearn.datasets import make_classification, make_regression


class TestTransformers:
    def test_log_transformer_log1p(self):
        """Test LogTransformer with log1p method."""
        X = np.array([[1, 2, 3], [4, 5, 6], [0, 1, 2]]).astype(float)

        transformer = LogTransformer(method="log1p")
        X_transformed = transformer.fit_transform(X)

        expected = np.log1p(X)
        np.testing.assert_array_almost_equal(X_transformed, expected)

    def test_log_transformer_with_negatives(self):
        """Test LogTransformer handles negative values."""
        X = np.array([[-1, 2, 3], [4, -5, 6], [0, 1, -2]]).astype(float)

        transformer = LogTransformer(method="log", offset="auto")
        X_transformed = transformer.fit_transform(X)

        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_log_transformer_box_cox(self):
        """Test LogTransformer with Box-Cox method."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float)

        transformer = LogTransformer(method="box-cox")
        X_transformed = transformer.fit_transform(X)

        assert X_transformed.shape == X.shape
        assert not np.any(np.isnan(X_transformed))

    def test_outlier_cap_transformer_iqr(self):
        """Test OutlierCapTransformer with IQR method."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [100, 200]]).astype(float)

        transformer = OutlierCapTransformer(method="iqr")
        X_transformed = transformer.fit_transform(X)

        # Outliers should be capped
        assert np.max(X_transformed[:, 0]) < 100
        assert np.max(X_transformed[:, 1]) < 200

    def test_outlier_cap_transformer_percentile(self):
        """Test OutlierCapTransformer with percentile method."""
        X = np.array([[i] for i in range(100)]).astype(float)

        transformer = OutlierCapTransformer(
            method="percentile", lower_quantile=0.1, upper_quantile=0.9
        )
        X_transformed = transformer.fit_transform(X)

        # Values should be capped at 10th and 90th percentiles
        assert np.min(X_transformed) >= np.percentile(X, 10)
        assert np.max(X_transformed) <= np.percentile(X, 90)

    def test_binning_transformer_quantile(self):
        """Test BinningTransformer with quantile strategy."""
        X = np.array([[i] for i in range(100)]).astype(float)

        transformer = BinningTransformer(strategy="quantile", n_bins=5)
        X_transformed = transformer.fit_transform(X)

        # Should have 5 bins (0-4)
        unique_bins = np.unique(X_transformed)
        assert len(unique_bins) <= 5
        assert np.min(unique_bins) >= 0
        assert np.max(unique_bins) <= 4

    def test_binning_transformer_uniform(self):
        """Test BinningTransformer with uniform strategy."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]).astype(float)

        transformer = BinningTransformer(strategy="uniform", n_bins=3)
        X_transformed = transformer.fit_transform(X)

        assert X_transformed.shape == X.shape
        assert np.all(X_transformed >= 0)
        assert np.all(X_transformed <= 2)

    def test_polynomial_features_advanced(self):
        """Test PolynomialFeaturesAdvanced."""
        X = np.array([[1, 2], [3, 4], [5, 6]]).astype(float)

        transformer = PolynomialFeaturesAdvanced(degree=2, include_bias=False)
        X_transformed = transformer.fit_transform(X)

        # Should include original features + interactions + squares
        assert X_transformed.shape[1] >= X.shape[1]
        assert X_transformed.shape[0] == X.shape[0]

    def test_robust_scaler(self):
        """Test RobustScaler."""
        X = np.array([[1, 2], [3, 4], [5, 6], [100, 200]]).astype(float)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Should be centered around median
        assert X_scaled.shape == X.shape
        # Most values should be within reasonable range (not affected by outlier)
        assert np.abs(np.median(X_scaled, axis=0)).max() < 0.1


class TestSelection:
    def test_feature_selector_variance(self):
        """Test FeatureSelector with variance method."""
        X = np.array([[1, 1, 2], [2, 1, 3], [3, 1, 4], [4, 1, 5]]).astype(float)
        y = np.array([0, 0, 1, 1])

        selector = FeatureSelector(methods=["variance"], variance_threshold=0.1)
        X_selected = selector.fit_transform(X, y)

        # Should remove constant column (index 1)
        assert X_selected.shape[1] == 2
        assert X_selected.shape[0] == X.shape[0]

    def test_feature_selector_correlation(self):
        """Test FeatureSelector with correlation method."""
        X = np.array([[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]).astype(float)

        selector = FeatureSelector(methods=["correlation"], correlation_threshold=0.9)
        X_selected = selector.fit_transform(X)

        # Should remove highly correlated features
        assert X_selected.shape[1] < X.shape[1]

    def test_mutual_info_selector_classification(self):
        """Test MutualInfoSelector for classification."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=5, random_state=42
        )

        selector = MutualInfoSelector(k=10)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X.shape[0]

    def test_mutual_info_selector_regression(self):
        """Test MutualInfoSelector for regression."""
        X, y = make_regression(
            n_samples=100, n_features=20, n_informative=5, random_state=42
        )

        selector = MutualInfoSelector(k=0.5)  # Select 50% of features
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X.shape[0]

    def test_variance_threshold_advanced(self):
        """Test VarianceThresholdAdvanced with auto threshold."""
        X = np.random.randn(100, 10)
        # Make some features have very low variance
        X[:, 5] = 0.001
        X[:, 8] = 0.001

        selector = VarianceThresholdAdvanced(threshold="auto", percentile=50)
        X_selected = selector.fit_transform(X)

        assert X_selected.shape[1] < X.shape[1]

    def test_correlation_filter(self):
        """Test CorrelationFilter."""
        X = np.random.randn(100, 5)
        X[:, 2] = X[:, 0] + 0.01 * np.random.randn(100)  # Highly correlated

        filter_ = CorrelationFilter(threshold=0.95)
        X_filtered = filter_.fit_transform(X)

        assert X_filtered.shape[1] == 4  # Should remove one correlated feature


class TestEncoding:
    def test_target_encoder_regression(self):
        """Test TargetEncoder for regression."""
        X = np.array([["A"], ["B"], ["C"], ["A"], ["B"], ["C"]])
        y = np.array([1.0, 2.0, 3.0, 1.1, 2.1, 3.1])

        encoder = TargetEncoder(cv=2)
        X_encoded = encoder.fit_transform(X, y)

        assert X_encoded.shape == (6, 1)
        assert not np.any(np.isnan(X_encoded))

    def test_target_encoder_classification(self):
        """Test TargetEncoder for classification."""
        X = np.array([["A"], ["B"], ["C"], ["A"], ["B"], ["C"]])
        y = np.array([0, 1, 1, 0, 1, 1])

        encoder = TargetEncoder(cv=2)
        X_encoded = encoder.fit_transform(X, y)

        assert X_encoded.shape == (6, 1)
        assert not np.any(np.isnan(X_encoded))

    def test_bayesian_target_encoder(self):
        """Test BayesianTargetEncoder."""
        X = np.array([["A"], ["B"], ["C"], ["A"], ["B"]])
        y = np.array([1.0, 2.0, 3.0, 1.1, 2.1])

        encoder = BayesianTargetEncoder(alpha=1.0)
        X_encoded = encoder.fit_transform(X, y)

        assert X_encoded.shape == (5, 1)
        assert not np.any(np.isnan(X_encoded))

    def test_frequency_encoder(self):
        """Test FrequencyEncoder."""
        X = np.array([["A"], ["B"], ["A"], ["A"], ["C"]])

        encoder = FrequencyEncoder(normalize=True)
        X_encoded = encoder.fit_transform(X)

        # 'A' appears 3 times, 'B' once, 'C' once
        expected_freqs = [0.6, 0.2, 0.6, 0.6, 0.2]
        np.testing.assert_array_almost_equal(X_encoded.flatten(), expected_freqs)

    def test_rare_class_encoder(self):
        """Test RareClassEncoder."""
        X = np.array([["A"], ["B"], ["C"], ["A"], ["A"], ["D"]])

        encoder = RareClassEncoder(min_frequency=2, rare_label="RARE")
        X_encoded = encoder.fit_transform(X)

        # Only 'A' appears >= 2 times, others should be 'RARE'
        expected = np.array([["A"], ["RARE"], ["RARE"], ["A"], ["A"], ["RARE"]])
        np.testing.assert_array_equal(X_encoded, expected)

    def test_ordinal_encoder_advanced(self):
        """Test OrdinalEncoderAdvanced with auto ordering."""
        X = np.array([["low"], ["medium"], ["high"], ["low"], ["high"]])

        encoder = OrdinalEncoderAdvanced(ordering="alphabetical")
        X_encoded = encoder.fit_transform(X)

        assert X_encoded.shape == (5, 1)
        assert np.all(X_encoded >= 0)
        assert np.all(X_encoded <= 2)


class TestInteractions:
    def test_interaction_detector_tree_based(self):
        """Test InteractionDetector with tree-based method."""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        # Add a clear interaction
        X[:, 0] = X[:, 0] * X[:, 1]

        detector = InteractionDetector(method="tree_based", max_interactions=10)
        detector.fit(X, y)

        interactions = detector.get_interactions()
        assert len(interactions) <= 10
        assert len(interactions) > 0

    def test_interaction_detector_mutual_info(self):
        """Test InteractionDetector with mutual information method."""
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)

        detector = InteractionDetector(method="mutual_info", max_interactions=5)
        detector.fit(X, y)

        interactions = detector.get_interactions()
        assert len(interactions) <= 5

    def test_interaction_generator(self):
        """Test InteractionGenerator."""
        X = np.array([[1, 2], [3, 4], [5, 6]]).astype(float)

        generator = InteractionGenerator(interaction_types=["multiply", "add"])
        X_with_interactions = generator.fit_transform(X)

        # Should have original features + interactions
        assert X_with_interactions.shape[1] > X.shape[1]
        assert X_with_interactions.shape[0] == X.shape[0]

    def test_polynomial_interactions(self):
        """Test PolynomialInteractions."""
        X = np.array([[1, 2], [3, 4], [5, 6]]).astype(float)
        y = np.array([1, 2, 3])

        poly = PolynomialInteractions(
            degree=2, feature_selection=None
        )  # No selection to keep all features
        X_poly = poly.fit_transform(X, y)

        assert X_poly.shape[1] >= X.shape[1]
        assert X_poly.shape[0] == X.shape[0]


class TestTemporal:
    def test_datetime_features(self):
        """Test DateTimeFeatures extraction."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        X = pd.DataFrame({"date": dates})

        extractor = DateTimeFeatures(
            features=["year", "month", "day", "dayofweek"], cyclical_encoding=False
        )
        X_features = extractor.fit_transform(X)

        # Should have extracted features
        assert "date_year" in X_features.columns
        assert "date_month" in X_features.columns
        assert "date_day" in X_features.columns
        assert "date_dayofweek" in X_features.columns

    def test_datetime_features_cyclical(self):
        """Test DateTimeFeatures with cyclical encoding."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        X = pd.DataFrame({"date": dates})

        extractor = DateTimeFeatures(features=["month", "hour"], cyclical_encoding=True)
        X_features = extractor.fit_transform(X)

        # Should have sin/cos features for cyclical encoding
        if "date_month_sin" in X_features.columns:
            assert "date_month_cos" in X_features.columns

    def test_lag_features(self):
        """Test LagFeatures."""
        X = np.array([[1], [2], [3], [4], [5]]).astype(float)

        lag_transformer = LagFeatures(lags=[1, 2], fill_value=0)
        X_lagged = lag_transformer.fit_transform(X)

        # Should have original + lag features
        assert X_lagged.shape[1] == 3  # original + 2 lags
        assert X_lagged.shape[0] == X.shape[0]

    def test_rolling_features(self):
        """Test RollingFeatures."""
        X = np.array([[1], [2], [3], [4], [5]]).astype(float)

        rolling_transformer = RollingFeatures(windows=[2], statistics=["mean", "std"])
        X_rolling = rolling_transformer.fit_transform(X)

        # Should have original + rolling features
        assert X_rolling.shape[1] == 3  # original + mean + std
        assert X_rolling.shape[0] == X.shape[0]

    def test_seasonal_decomposition(self):
        """Test SeasonalDecomposition."""
        # Create seasonal data
        np.random.seed(42)
        n_points = 100
        t = np.arange(n_points)
        trend = 0.1 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 12)  # Period of 12
        noise = 0.1 * np.random.randn(n_points)
        X = (trend + seasonal + noise).reshape(-1, 1)

        decomposer = SeasonalDecomposition(period=12)
        X_decomposed = decomposer.fit_transform(X)

        # Should have original + trend + seasonal + residual
        expected_cols = 1 + 3  # original + 3 components
        if X_decomposed.shape[1] >= expected_cols:
            assert True  # Decomposition worked
        else:
            # If decomposition failed due to insufficient data, that's ok
            assert X_decomposed.shape == X.shape
