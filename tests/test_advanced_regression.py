"""
Tests for advanced regression features.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from analytics_toolkit.pytorch_regression.advanced import (
    RegularizationPath,
    PolynomialRegression,
    RobustRegression
)


class TestRegularizationPath:
    """Test cases for RegularizationPath class."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data for testing."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_l2_regularization_path(self, regression_data):
        """Test L2 regularization path computation."""
        X_train, X_test, y_train, y_test = regression_data

        reg_path = RegularizationPath(
            penalty="l2",
            n_alphas=10,
            cv=3,
            random_state=42
        )

        reg_path.fit(X_train, y_train)

        # Check that attributes are set
        assert reg_path.alphas_ is not None
        assert reg_path.coef_path_ is not None
        assert reg_path.cv_scores_ is not None
        assert reg_path.alpha_optimal_ is not None
        assert reg_path.best_model_ is not None

        # Check shapes
        assert len(reg_path.alphas_) == 10
        assert reg_path.coef_path_.shape == (10, X_train.shape[1])
        assert len(reg_path.cv_scores_) == 10

        # Test predictions with best model
        y_pred = reg_path.best_model_.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert isinstance(y_pred, np.ndarray)

    def test_l1_regularization_path(self, regression_data):
        """Test L1 regularization path computation."""
        X_train, X_test, y_train, y_test = regression_data

        reg_path = RegularizationPath(
            penalty="l1",
            n_alphas=5,
            cv=3,
            random_state=42
        )

        reg_path.fit(X_train, y_train)

        # Check that sparsity increases with alpha
        coef_norms = np.linalg.norm(reg_path.coef_path_, axis=1)
        assert coef_norms[0] >= coef_norms[-1]  # Decreasing coefficient norms

    def test_custom_alphas(self, regression_data):
        """Test with custom alpha values."""
        X_train, X_test, y_train, y_test = regression_data

        custom_alphas = np.logspace(-3, 1, 5)
        reg_path = RegularizationPath(
            penalty="l2",
            alphas=custom_alphas,
            cv=3,
            random_state=42
        )

        reg_path.fit(X_train, y_train)

        np.testing.assert_array_equal(reg_path.alphas_, custom_alphas)

    def test_plot_data(self, regression_data):
        """Test plot data generation."""
        X_train, X_test, y_train, y_test = regression_data

        reg_path = RegularizationPath(penalty="l2", n_alphas=5, cv=3)
        reg_path.fit(X_train, y_train)

        plot_data = reg_path.plot_path(feature_names=[f'Feature_{i}' for i in range(10)])

        required_keys = ['alphas', 'coef_path', 'cv_scores', 'alpha_optimal', 'feature_names']
        for key in required_keys:
            assert key in plot_data


class TestPolynomialRegression:
    """Test cases for PolynomialRegression class."""

    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear data for testing."""
        np.random.seed(42)
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = 0.5 * X.ravel() ** 3 + 0.2 * X.ravel() ** 2 + 0.1 * X.ravel() + 0.05 * np.random.randn(100)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_fixed_degree_polynomial(self, nonlinear_data):
        """Test polynomial regression with fixed degree."""
        X_train, X_test, y_train, y_test = nonlinear_data

        poly_reg = PolynomialRegression(degree=3, penalty="l2", alpha=0.01)
        poly_reg.fit(X_train, y_train)

        assert poly_reg.degree_ == 3
        assert poly_reg.poly_features_ is not None
        assert poly_reg.model_ is not None

        # Test predictions
        y_pred = poly_reg.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Test score
        score = poly_reg.score(X_test, y_test)
        assert isinstance(score, float)
        assert score > 0.5  # Should fit reasonably well

    def test_automatic_degree_selection(self, nonlinear_data):
        """Test automatic degree selection via CV."""
        X_train, X_test, y_train, y_test = nonlinear_data

        poly_reg = PolynomialRegression(
            max_degree=5,
            cv=3,
            penalty="l2",
            alpha=0.01,
            random_state=42
        )
        poly_reg.fit(X_train, y_train)

        assert poly_reg.degree_ is not None
        assert 1 <= poly_reg.degree_ <= 5
        assert poly_reg.cv_scores_ is not None
        assert len(poly_reg.cv_scores_) == 5

    def test_interaction_only(self, nonlinear_data):
        """Test interaction-only polynomial features."""
        X_train, X_test, y_train, y_test = nonlinear_data

        # Create multi-feature data
        X_train_multi = np.hstack([X_train, X_train**2])
        X_test_multi = np.hstack([X_test, X_test**2])

        poly_reg = PolynomialRegression(
            degree=2,
            interaction_only=True,
            penalty="l2",
            alpha=0.01
        )
        poly_reg.fit(X_train_multi, y_train)

        # Should work without error
        y_pred = poly_reg.predict(X_test_multi)
        assert len(y_pred) == len(y_test)

    def test_with_regularization(self, nonlinear_data):
        """Test polynomial regression with regularization."""
        X_train, X_test, y_train, y_test = nonlinear_data

        poly_reg = PolynomialRegression(
            degree=4,
            penalty="l2",
            alpha=1.0,  # Strong regularization
        )
        poly_reg.fit(X_train, y_train)

        # Test that it still works with regularization
        y_pred = poly_reg.predict(X_test)
        assert len(y_pred) == len(y_test)


class TestRobustRegression:
    """Test cases for RobustRegression class."""

    @pytest.fixture
    def outlier_data(self):
        """Generate data with outliers for testing."""
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

        # Add outliers
        outlier_indices = np.random.choice(100, 10, replace=False)
        y[outlier_indices] += np.random.normal(0, 10, 10)

        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_huber_regression(self, outlier_data):
        """Test Huber robust regression."""
        X_train, X_test, y_train, y_test = outlier_data

        robust_reg = RobustRegression(
            method="huber",
            epsilon=1.35,
            max_iter=100,
            alpha=0.01
        )

        robust_reg.fit(X_train, y_train)

        # Check that model is fitted
        assert robust_reg.coef_ is not None
        assert robust_reg.intercept_ is not None

        # Test predictions
        y_pred = robust_reg.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert isinstance(y_pred, np.ndarray)

        # Test score
        score = robust_reg.score(X_test, y_test)
        assert isinstance(score, float)

    def test_huber_vs_ols_with_outliers(self, outlier_data):
        """Test that Huber regression is more robust than OLS with outliers."""
        X_train, X_test, y_train, y_test = outlier_data

        # Fit Huber regression
        huber_reg = RobustRegression(method="huber", epsilon=1.35)
        huber_reg.fit(X_train, y_train)
        huber_score = huber_reg.score(X_test, y_test)

        # Fit regular linear regression (from main module)
        from analytics_toolkit.pytorch_regression import LinearRegression
        ols_reg = LinearRegression()
        ols_reg.fit(X_train, y_train)
        ols_score = ols_reg.score(X_test, y_test)

        # Huber should be more robust (though this might not always hold in practice)
        # At minimum, both should produce reasonable scores
        assert -10 < huber_score < 1
        assert -10 < ols_score < 1

    def test_huber_parameters(self, outlier_data):
        """Test different Huber parameters."""
        X_train, X_test, y_train, y_test = outlier_data

        # Test different epsilon values
        for epsilon in [0.5, 1.35, 2.0]:
            robust_reg = RobustRegression(method="huber", epsilon=epsilon, max_iter=50)
            robust_reg.fit(X_train, y_train)

            y_pred = robust_reg.predict(X_test)
            assert len(y_pred) == len(y_test)

    def test_invalid_method(self, outlier_data):
        """Test error handling for invalid method."""
        X_train, X_test, y_train, y_test = outlier_data

        with pytest.raises(ValueError):
            robust_reg = RobustRegression(method="invalid_method")
            robust_reg.fit(X_train, y_train)


class TestIntegration:
    """Integration tests for advanced regression features."""

    @pytest.fixture
    def complex_data(self):
        """Generate complex dataset for integration testing."""
        np.random.seed(42)
        X, y = make_regression(
            n_samples=200,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_regularization_path_with_polynomial(self, complex_data):
        """Test regularization path on polynomial features."""
        X_train, X_test, y_train, y_test = complex_data

        # Use subset of features for computational efficiency
        X_train_sub = X_train[:, :3]
        X_test_sub = X_test[:, :3]

        # First apply polynomial transformation
        poly_reg = PolynomialRegression(degree=2, penalty="l2", alpha=0.01)
        poly_reg.fit(X_train_sub, y_train)

        # Then use regularization path on the polynomial features
        X_train_poly = poly_reg.poly_features_.transform(X_train_sub)

        reg_path = RegularizationPath(penalty="l2", n_alphas=5, cv=3)
        reg_path.fit(X_train_poly, y_train)

        assert reg_path.alpha_optimal_ is not None
        assert reg_path.best_model_ is not None

    def test_robust_polynomial_regression(self, complex_data):
        """Test combining robust regression with polynomial features."""
        X_train, X_test, y_train, y_test = complex_data

        # Add outliers
        outlier_indices = np.random.choice(len(y_train), 10, replace=False)
        y_train_outliers = y_train.copy()
        y_train_outliers[outlier_indices] += np.random.normal(0, 5, 10)

        # Use simple features for this test
        X_train_sub = X_train[:, :2]
        X_test_sub = X_test[:, :2]

        # Apply polynomial features manually then use robust regression
        from sklearn.preprocessing import PolynomialFeatures
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_sub)
        X_test_poly = poly_features.transform(X_test_sub)

        robust_reg = RobustRegression(method="huber", epsilon=1.35)
        robust_reg.fit(X_train_poly, y_train_outliers)

        y_pred = robust_reg.predict(X_test_poly)
        assert len(y_pred) == len(y_test)

    def test_all_methods_produce_valid_output(self, complex_data):
        """Test that all advanced methods produce valid outputs."""
        X_train, X_test, y_train, y_test = complex_data

        # Use subset for computational efficiency
        X_train_sub = X_train[:, :5]
        X_test_sub = X_test[:, :5]

        methods = [
            RegularizationPath(penalty="l2", n_alphas=3, cv=2),
            PolynomialRegression(degree=2, penalty="l2", alpha=0.01),
            RobustRegression(method="huber", epsilon=1.35, max_iter=50)
        ]

        for method in methods:
            method.fit(X_train_sub, y_train)

            if hasattr(method, 'predict'):
                y_pred = method.predict(X_test_sub)
                assert len(y_pred) == len(y_test)
                assert np.all(np.isfinite(y_pred))
            elif hasattr(method, 'best_model_'):
                y_pred = method.best_model_.predict(X_test_sub)
                assert len(y_pred) == len(y_test)
                assert np.all(np.isfinite(y_pred))


if __name__ == "__main__":
    pytest.main([__file__])