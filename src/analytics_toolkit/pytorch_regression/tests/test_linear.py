"""
Tests for LinearRegression class.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import warnings

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from ..linear import LinearRegression


class TestLinearRegression:
    """Test suite for LinearRegression class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_coef = np.array([1.5, -2.0, 0.5])
        y = X @ true_coef + 0.1 * np.random.randn(100)
        return X, y, true_coef

    @pytest.fixture
    def regression_data(self):
        """Create larger regression dataset."""
        X, y = make_regression(
            n_samples=1000, n_features=10, noise=0.1, random_state=42
        )
        return X, y

    @pytest.fixture
    def categorical_data(self):
        """Create data with categorical variables."""
        np.random.seed(42)
        n_samples = 200

        # Continuous features
        X_cont = np.random.randn(n_samples, 2)

        # Categorical features
        categories = ['A', 'B', 'C']
        cat_feature = np.random.choice(categories, n_samples)

        # Create DataFrame
        df = pd.DataFrame({
            'x1': X_cont[:, 0],
            'x2': X_cont[:, 1],
            'category': cat_feature
        })

        # True coefficients (including dummy variables)
        y = (1.0 * df['x1'] +
             -0.5 * df['x2'] +
             2.0 * (df['category'] == 'B') +
             -1.0 * (df['category'] == 'C') +
             0.1 * np.random.randn(n_samples))

        return df, y

    def test_initialization(self):
        """Test model initialization."""
        # Default parameters
        model = LinearRegression()
        assert model.fit_intercept is True
        assert model.penalty == 'none'
        assert model.alpha == 0.01

        # Custom parameters
        model = LinearRegression(
            fit_intercept=False,
            penalty='l2',
            alpha=0.1,
            max_iter=500
        )
        assert model.fit_intercept is False
        assert model.penalty == 'l2'
        assert model.alpha == 0.1
        assert model.max_iter == 500

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            LinearRegression(penalty='invalid')

    def test_basic_fit_predict(self, simple_data):
        """Test basic fit and predict functionality."""
        X, y, _ = simple_data

        model = LinearRegression()
        model.fit(X, y)

        assert model.is_fitted_ is True
        assert model.coef_ is not None
        assert len(model.coef_) == 4  # 3 features + intercept

        # Test predictions
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert not np.any(np.isnan(y_pred))

        # Test score (R²)
        score = model.score(X, y)
        assert 0 <= score <= 1

    def test_no_intercept(self, simple_data):
        """Test fitting without intercept."""
        X, y, _ = simple_data

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        assert len(model.coef_) == 3  # Only features, no intercept

    def test_sample_weights(self, simple_data):
        """Test sample weight functionality."""
        X, y, _ = simple_data

        # Create sample weights
        sample_weight = np.random.rand(len(y))

        model = LinearRegression()
        model.fit(X, y, sample_weight=sample_weight)

        assert model.is_fitted_ is True

        # Predictions should work
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_regularization(self, regression_data):
        """Test L1 and L2 regularization."""
        X, y = regression_data

        # L2 regularization
        model_l2 = LinearRegression(penalty='l2', alpha=0.1)
        model_l2.fit(X, y)
        assert model_l2.is_fitted_ is True

        # L1 regularization
        model_l1 = LinearRegression(penalty='l1', alpha=0.1)
        model_l1.fit(X, y)
        assert model_l1.is_fitted_ is True

        # L2 coefficients should be smaller in magnitude
        coef_no_reg = LinearRegression(penalty='none').fit(X, y).coef_
        coef_l2 = model_l2.coef_

        # Check that regularization affects coefficients
        assert not np.allclose(coef_no_reg, coef_l2)

    def test_categorical_handling(self, categorical_data):
        """Test automatic categorical variable handling."""
        df, y = categorical_data

        model = LinearRegression()
        model.fit(df, y)

        assert model.is_fitted_ is True

        # Should have coefficients for continuous + dummy variables
        # x1, x2 + category_B, category_C + intercept = 5 coefficients
        assert len(model.coef_) == 5

        # Test predictions on new data
        y_pred = model.predict(df)
        assert len(y_pred) == len(y)

    def test_statistical_inference(self, simple_data):
        """Test statistical inference capabilities."""
        X, y, _ = simple_data

        model = LinearRegression()
        model.fit(X, y)

        # Test standard errors
        assert model.standard_errors_ is not None
        assert len(model.standard_errors_) == len(model.coef_)

        # Test confidence intervals
        conf_int = model.conf_int()
        assert isinstance(conf_int, pd.DataFrame)
        assert len(conf_int) == len(model.coef_)
        assert 'lower' in conf_int.columns
        assert 'upper' in conf_int.columns

        # Test summary
        summary = model.summary()
        assert isinstance(summary, str)
        assert 'coef' in summary
        assert 'std err' in summary
        assert 'R-squared' in summary

    def test_prediction_intervals(self, simple_data):
        """Test prediction interval computation."""
        X, y, _ = simple_data

        model = LinearRegression()
        model.fit(X, y)

        # Test prediction intervals
        predictions, lower, upper = model.predict_interval(X)

        assert len(predictions) == len(y)
        assert len(lower) == len(y)
        assert len(upper) == len(y)

        # Lower bounds should be less than upper bounds
        assert np.all(lower < upper)

        # Point predictions should be between bounds (approximately)
        assert np.all(lower <= predictions)
        assert np.all(predictions <= upper)

    def test_model_statistics(self, simple_data):
        """Test model fit statistics."""
        X, y, _ = simple_data

        model = LinearRegression()
        model.fit(X, y)

        # Check R²
        assert hasattr(model, 'r_squared_')
        assert 0 <= model.r_squared_ <= 1

        # Check adjusted R²
        assert hasattr(model, 'adj_r_squared_')

        # Check information criteria
        assert hasattr(model, 'aic_')
        assert hasattr(model, 'bic_')
        assert model.aic_ is not None
        assert model.bic_ is not None

    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not available")
    def test_comparison_with_statsmodels(self, simple_data):
        """Compare results with statsmodels."""
        X, y, _ = simple_data

        # Fit our model
        our_model = LinearRegression()
        our_model.fit(X, y)

        # Fit statsmodels
        X_sm = sm.add_constant(X)
        sm_model = sm.OLS(y, X_sm).fit()

        # Compare coefficients (should be close)
        np.testing.assert_allclose(
            our_model.coef_.detach().cpu().numpy(),
            sm_model.params.values,
            rtol=1e-3
        )

        # Compare standard errors
        if our_model.standard_errors_ is not None:
            np.testing.assert_allclose(
                our_model.standard_errors_.detach().cpu().numpy(),
                sm_model.bse.values,
                rtol=1e-2
            )

        # Compare R²
        np.testing.assert_allclose(our_model.r_squared_, sm_model.rsquared, rtol=1e-3)

    def test_comparison_with_sklearn(self, regression_data):
        """Compare predictions with scikit-learn."""
        X, y = regression_data

        # Fit our model
        our_model = LinearRegression()
        our_model.fit(X, y)

        # Fit sklearn model
        sklearn_model = SklearnLinearRegression()
        sklearn_model.fit(X, y)

        # Compare predictions
        our_pred = our_model.predict(X)
        sklearn_pred = sklearn_model.predict(X)

        np.testing.assert_allclose(our_pred, sklearn_pred, rtol=1e-3)

        # Compare R² scores
        our_score = our_model.score(X, y)
        sklearn_score = sklearn_model.score(X, y)

        np.testing.assert_allclose(our_score, sklearn_score, rtol=1e-3)

    def test_different_input_types(self, simple_data):
        """Test different input data types."""
        X, y, _ = simple_data

        model = LinearRegression()

        # Test numpy arrays
        model.fit(X, y)
        pred_numpy = model.predict(X)

        # Test pandas DataFrame
        df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
        model_df = LinearRegression()
        model_df.fit(df, y)
        pred_df = model_df.predict(df)

        # Results should be similar
        np.testing.assert_allclose(pred_numpy, pred_df, rtol=1e-3)

        # Test torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        model_tensor = LinearRegression()
        model_tensor.fit(X_tensor, y_tensor)
        pred_tensor = model_tensor.predict(X_tensor)

        np.testing.assert_allclose(pred_numpy, pred_tensor, rtol=1e-3)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        model = LinearRegression()

        # Test fitting before prediction
        with pytest.raises(ValueError):
            model.predict(np.random.randn(10, 5))

        # Test singular matrix handling
        X_singular = np.ones((10, 3))  # All columns identical
        y = np.random.randn(10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_singular, y)
            # Should not crash, but may have warnings

    def test_residuals(self, simple_data):
        """Test residual computation."""
        X, y, _ = simple_data

        model = LinearRegression()
        model.fit(X, y)

        # Test raw residuals
        residuals = model.get_residuals(X, y, residual_type='raw')
        assert len(residuals) == len(y)

        # Test standardized residuals
        std_residuals = model.get_residuals(X, y, residual_type='standardized')
        assert len(std_residuals) == len(y)

        # Standardized residuals should have approximately unit variance
        assert abs(np.std(std_residuals) - 1.0) < 0.2

    def test_get_set_params(self):
        """Test parameter getting and setting."""
        model = LinearRegression(alpha=0.1, penalty='l2')

        # Test get_params
        params = model.get_params()
        assert params['alpha'] == 0.1
        assert params['penalty'] == 'l2'

        # Test set_params
        model.set_params(alpha=0.2, max_iter=500)
        assert model.alpha == 0.2
        assert model.max_iter == 500

        # Test invalid parameter
        with pytest.raises(ValueError):
            model.set_params(invalid_param=123)

    def test_device_handling(self):
        """Test GPU/CPU device handling."""
        model = LinearRegression(device='cpu')
        assert model.device.type == 'cpu'

        # Test auto device selection
        model_auto = LinearRegression(device='auto')
        assert model_auto.device is not None

    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        # Create larger dataset
        n_samples, n_features = 5000, 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        model = LinearRegression()

        # Should fit without memory issues
        model.fit(X, y)
        assert model.is_fitted_ is True

        # Predictions should work
        y_pred = model.predict(X)
        assert len(y_pred) == n_samples