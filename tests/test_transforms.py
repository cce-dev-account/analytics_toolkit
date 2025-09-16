"""
Tests for non-linear transformation classes.
"""

import numpy as np
import pytest
from analytics_toolkit.pytorch_regression.transforms import (
    BSplineTransformer,
    FourierTransformer,
    RadialBasisTransformer,
)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class TestBSplineTransformer:
    """Test cases for BSplineTransformer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        return X, y

    def test_basic_functionality(self, sample_data):
        """Test basic fit and transform functionality."""
        X, y = sample_data

        transformer = BSplineTransformer(n_knots=5, degree=3)
        X_transformed = transformer.fit_transform(X)

        assert transformer.n_features_in_ == 3
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > X.shape[1]  # More features after transformation
        assert not np.any(np.isnan(X_transformed))

    def test_different_knot_strategies(self, sample_data):
        """Test different knot placement strategies."""
        X, y = sample_data

        # Quantile strategy
        transformer_quantile = BSplineTransformer(n_knots=5, knot_strategy="quantile")
        X_quantile = transformer_quantile.fit_transform(X)

        # Uniform strategy
        transformer_uniform = BSplineTransformer(n_knots=5, knot_strategy="uniform")
        X_uniform = transformer_uniform.fit_transform(X)

        # Both should work and have same output shape
        assert X_quantile.shape == X_uniform.shape
        assert not np.any(np.isnan(X_quantile))
        assert not np.any(np.isnan(X_uniform))

    def test_different_degrees(self, sample_data):
        """Test different spline degrees."""
        X, y = sample_data

        degrees = [1, 2, 3, 4]
        for degree in degrees:
            transformer = BSplineTransformer(n_knots=5, degree=degree)
            X_transformed = transformer.fit_transform(X)

            assert X_transformed.shape[0] == X.shape[0]
            assert not np.any(np.isnan(X_transformed))

    def test_bias_inclusion(self, sample_data):
        """Test bias term inclusion."""
        X, y = sample_data

        # With bias
        transformer_with_bias = BSplineTransformer(include_bias=True)
        X_with_bias = transformer_with_bias.fit_transform(X)

        # Without bias
        transformer_no_bias = BSplineTransformer(include_bias=False)
        X_no_bias = transformer_no_bias.fit_transform(X)

        # With bias should have one more column
        assert X_with_bias.shape[1] == X_no_bias.shape[1] + 1

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Single feature
        X_single = np.random.randn(50, 1)
        transformer = BSplineTransformer(n_knots=3)
        X_transformed = transformer.fit_transform(X_single)
        assert X_transformed.shape[0] == 50

        # Small dataset
        X_small = np.random.randn(10, 2)
        transformer = BSplineTransformer(n_knots=3)
        X_transformed = transformer.fit_transform(X_small)
        assert X_transformed.shape[0] == 10

    def test_invalid_knot_strategy(self, sample_data):
        """Test error handling for invalid knot strategy."""
        X, y = sample_data

        with pytest.raises(ValueError):
            transformer = BSplineTransformer(knot_strategy="invalid")
            transformer.fit(X)

    def test_transform_before_fit(self, sample_data):
        """Test error when transform is called before fit."""
        X, y = sample_data

        transformer = BSplineTransformer()
        with pytest.raises(ValueError):
            transformer.transform(X)

    def test_consistent_output(self, sample_data):
        """Test that multiple transforms of same data give consistent results."""
        X, y = sample_data

        transformer = BSplineTransformer(n_knots=5, degree=3)
        transformer.fit(X)

        X_transformed1 = transformer.transform(X)
        X_transformed2 = transformer.transform(X)

        np.testing.assert_array_equal(X_transformed1, X_transformed2)


class TestRadialBasisTransformer:
    """Test cases for RadialBasisTransformer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        return X, y

    def test_basic_functionality(self, sample_data):
        """Test basic fit and transform functionality."""
        X, y = sample_data

        transformer = RadialBasisTransformer(n_centers=10, random_state=42)
        X_transformed = transformer.fit_transform(X)

        assert transformer.n_features_in_ == 3
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 10  # n_centers
        assert not np.any(np.isnan(X_transformed))

    def test_different_kernels(self, sample_data):
        """Test different RBF kernels."""
        X, y = sample_data

        kernels = ["gaussian", "multiquadric", "inverse_multiquadric"]
        for kernel in kernels:
            transformer = RadialBasisTransformer(
                n_centers=8, kernel=kernel, random_state=42
            )
            X_transformed = transformer.fit_transform(X)

            assert X_transformed.shape == (100, 8)
            assert not np.any(np.isnan(X_transformed))

    def test_different_center_strategies(self, sample_data):
        """Test different center selection strategies."""
        X, y = sample_data

        strategies = ["random", "quantile", "kmeans"]
        for strategy in strategies:
            transformer = RadialBasisTransformer(
                n_centers=8, center_strategy=strategy, random_state=42
            )
            X_transformed = transformer.fit_transform(X)

            assert X_transformed.shape == (100, 8)
            assert not np.any(np.isnan(X_transformed))

    def test_include_original_features(self, sample_data):
        """Test including original features in output."""
        X, y = sample_data

        # Without original features
        transformer_no_orig = RadialBasisTransformer(
            n_centers=5, include_original=False
        )
        X_no_orig = transformer_no_orig.fit_transform(X)

        # With original features
        transformer_with_orig = RadialBasisTransformer(
            n_centers=5, include_original=True
        )
        X_with_orig = transformer_with_orig.fit_transform(X)

        assert X_no_orig.shape[1] == 5
        assert X_with_orig.shape[1] == 5 + 3  # RBF features + original features

    def test_custom_gamma(self, sample_data):
        """Test custom gamma parameter."""
        X, y = sample_data

        # Auto gamma
        transformer_auto = RadialBasisTransformer(
            n_centers=5, gamma=None, random_state=42
        )
        X_auto = transformer_auto.fit_transform(X)

        # Custom gamma
        transformer_custom = RadialBasisTransformer(
            n_centers=5, gamma=1.0, random_state=42
        )
        X_custom = transformer_custom.fit_transform(X)

        # Should have same shape but potentially different values
        assert X_auto.shape == X_custom.shape
        assert transformer_custom.gamma_ == 1.0

    def test_single_feature_data(self):
        """Test with single feature data."""
        X = np.random.randn(50, 1)

        transformer = RadialBasisTransformer(n_centers=5, random_state=42)
        X_transformed = transformer.fit_transform(X)

        assert X_transformed.shape == (50, 5)
        assert not np.any(np.isnan(X_transformed))

    def test_invalid_kernel(self, sample_data):
        """Test error handling for invalid kernel."""
        X, y = sample_data

        transformer = RadialBasisTransformer(kernel="invalid")
        transformer.fit(X)

        with pytest.raises(ValueError):
            transformer.transform(X)

    def test_invalid_center_strategy(self, sample_data):
        """Test error handling for invalid center strategy."""
        X, y = sample_data

        with pytest.raises(ValueError):
            transformer = RadialBasisTransformer(center_strategy="invalid")
            transformer.fit(X)

    def test_reproducibility(self, sample_data):
        """Test reproducibility with random_state."""
        X, y = sample_data

        transformer1 = RadialBasisTransformer(n_centers=5, random_state=42)
        X_transformed1 = transformer1.fit_transform(X)

        transformer2 = RadialBasisTransformer(n_centers=5, random_state=42)
        X_transformed2 = transformer2.fit_transform(X)

        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)


class TestFourierTransformer:
    """Test cases for FourierTransformer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        return X, y

    @pytest.fixture
    def periodic_data(self):
        """Generate periodic data for testing."""
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 100).reshape(-1, 1)
        y = np.sin(t.ravel()) + 0.5 * np.sin(3 * t.ravel())
        return t, y

    def test_basic_functionality(self, sample_data):
        """Test basic fit and transform functionality."""
        X, y = sample_data

        transformer = FourierTransformer(n_frequencies=5)
        X_transformed = transformer.fit_transform(X)

        assert transformer.n_features_in_ == 2
        assert X_transformed.shape[0] == X.shape[0]
        # 2 features * 5 frequencies * 2 (sin/cos) + 1 bias = 21
        assert X_transformed.shape[1] == 2 * 5 * 2 + 1
        assert not np.any(np.isnan(X_transformed))

    def test_no_bias(self, sample_data):
        """Test without bias term."""
        X, y = sample_data

        transformer = FourierTransformer(n_frequencies=3, include_bias=False)
        X_transformed = transformer.fit_transform(X)

        # 2 features * 3 frequencies * 2 (sin/cos) = 12
        assert X_transformed.shape[1] == 2 * 3 * 2

    def test_custom_frequency_range(self, sample_data):
        """Test custom frequency range."""
        X, y = sample_data

        transformer = FourierTransformer(
            n_frequencies=4, frequency_range=(0.5, 2.0), include_bias=False
        )
        X_transformed = transformer.fit_transform(X)

        assert X_transformed.shape[1] == 2 * 4 * 2
        np.testing.assert_array_equal(
            transformer.frequencies_, np.linspace(0.5, 2.0, 4)
        )

    def test_normalization(self, sample_data):
        """Test feature normalization."""
        X, y = sample_data

        # With normalization
        transformer_norm = FourierTransformer(normalize_features=True)
        X_norm = transformer_norm.fit_transform(X)

        # Without normalization
        transformer_no_norm = FourierTransformer(normalize_features=False)
        X_no_norm = transformer_no_norm.fit_transform(X)

        # Both should work but give different results
        assert X_norm.shape == X_no_norm.shape
        assert not np.array_equal(X_norm, X_no_norm)

    def test_periodic_data_modeling(self, periodic_data):
        """Test Fourier features on periodic data."""
        X, y = periodic_data

        transformer = FourierTransformer(n_frequencies=5, normalize_features=True)
        X_transformed = transformer.fit_transform(X)

        # Should capture periodic patterns well
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] == 1 * 5 * 2 + 1  # 11 features total
        assert not np.any(np.isnan(X_transformed))

    def test_single_feature(self):
        """Test with single feature data."""
        X = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)

        transformer = FourierTransformer(n_frequencies=3)
        X_transformed = transformer.fit_transform(X)

        # 1 feature * 3 frequencies * 2 (sin/cos) + 1 bias = 7
        assert X_transformed.shape == (50, 7)
        assert not np.any(np.isnan(X_transformed))

    def test_consistent_output(self, sample_data):
        """Test that multiple transforms give consistent results."""
        X, y = sample_data

        transformer = FourierTransformer(n_frequencies=4)
        transformer.fit(X)

        X_transformed1 = transformer.transform(X)
        X_transformed2 = transformer.transform(X)

        np.testing.assert_array_equal(X_transformed1, X_transformed2)

    def test_transform_before_fit(self, sample_data):
        """Test error when transform is called before fit."""
        X, y = sample_data

        transformer = FourierTransformer()
        with pytest.raises(ValueError):
            transformer.transform(X)

    def test_zero_frequencies(self):
        """Test edge case with zero frequencies."""
        X = np.random.randn(20, 2)

        transformer = FourierTransformer(n_frequencies=0, include_bias=True)
        X_transformed = transformer.fit_transform(X)

        # Should only have bias term
        assert X_transformed.shape == (20, 1)
        np.testing.assert_array_equal(X_transformed, np.ones((20, 1)))


class TestTransformersIntegration:
    """Integration tests for transformer combinations."""

    @pytest.fixture
    def complex_data(self):
        """Generate complex data for integration testing."""
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_pipeline_compatibility(self, complex_data):
        """Test compatibility with sklearn pipelines."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = complex_data

        # Test each transformer in a pipeline
        transformers = [
            BSplineTransformer(n_knots=5, degree=2),
            RadialBasisTransformer(n_centers=10, random_state=42),
            FourierTransformer(n_frequencies=3),
        ]

        for transformer in transformers:
            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("transformer", transformer)]
            )

            X_transformed = pipeline.fit_transform(X_train)
            X_test_transformed = pipeline.transform(X_test)

            assert X_transformed.shape[0] == len(X_train)
            assert X_test_transformed.shape[0] == len(X_test)
            assert X_transformed.shape[1] == X_test_transformed.shape[1]
            assert not np.any(np.isnan(X_transformed))
            assert not np.any(np.isnan(X_test_transformed))

    def test_all_transformers_different_outputs(self, complex_data):
        """Test that different transformers produce different outputs."""
        X_train, X_test, y_train, y_test = complex_data

        # Use same input size for fair comparison
        n_output_features = 15

        bspline = BSplineTransformer(n_knots=3, degree=2, include_bias=False)
        rbf = RadialBasisTransformer(n_centers=n_output_features, random_state=42)
        fourier = FourierTransformer(
            n_frequencies=3, include_bias=False
        )  # 5*3*2=30, close enough

        X_bspline = bspline.fit_transform(X_train)
        X_rbf = rbf.fit_transform(X_train)
        X_fourier = fourier.fit_transform(X_train)

        # All should be different
        assert not np.array_equal(X_bspline, X_rbf)
        assert not np.array_equal(X_rbf, X_fourier)
        assert not np.array_equal(X_bspline, X_fourier)

    def test_memory_efficiency(self):
        """Test that transformers handle reasonably large datasets."""
        # Generate larger dataset
        X = np.random.randn(1000, 10)

        transformers = [
            BSplineTransformer(n_knots=5),
            RadialBasisTransformer(n_centers=20, random_state=42),
            FourierTransformer(n_frequencies=5),
        ]

        for transformer in transformers:
            X_transformed = transformer.fit_transform(X)
            assert X_transformed.shape[0] == 1000
            assert not np.any(np.isnan(X_transformed))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values
        X_large = np.random.randn(50, 3) * 1000

        # Test with very small values
        X_small = np.random.randn(50, 3) * 0.001

        transformers = [
            BSplineTransformer(n_knots=5),
            RadialBasisTransformer(n_centers=10, random_state=42),
            FourierTransformer(n_frequencies=3),
        ]

        for transformer in transformers:
            # Should handle large values
            X_large_transformed = transformer.fit_transform(X_large)
            assert not np.any(np.isnan(X_large_transformed))
            assert not np.any(np.isinf(X_large_transformed))

            # Should handle small values
            X_small_transformed = transformer.fit_transform(X_small)
            assert not np.any(np.isnan(X_small_transformed))
            assert not np.any(np.isinf(X_small_transformed))


if __name__ == "__main__":
    pytest.main([__file__])
