"""
Tests for statistical computation functions.
"""

import numpy as np
import pytest
import torch
from scipy import stats

from ..stats import (
    compute_confidence_intervals,
    compute_information_criteria,
    compute_model_statistics,
    compute_p_values,
    compute_residuals,
    compute_standard_errors,
    compute_test_statistics,
    format_summary_table,
)


class TestStats:
    """Test suite for statistical functions."""

    @pytest.fixture
    def sample_covariance_matrix(self):
        """Create a sample covariance matrix."""
        # Create a positive definite matrix
        A = np.random.randn(4, 4)
        cov_matrix = A @ A.T + 0.1 * np.eye(4)
        return torch.tensor(cov_matrix, dtype=torch.float32)

    @pytest.fixture
    def sample_coefficients(self):
        """Create sample coefficients."""
        return torch.tensor([2.0, -1.5, 0.8, 3.2], dtype=torch.float32)

    def test_compute_standard_errors(self, sample_covariance_matrix):
        """Test standard error computation."""
        # Test with torch computation
        std_errors_torch = compute_standard_errors(
            sample_covariance_matrix, use_torch=True
        )
        assert isinstance(std_errors_torch, torch.Tensor)
        assert len(std_errors_torch) == 4
        assert torch.all(std_errors_torch > 0)

        # Test with numpy computation (default)
        std_errors_numpy = compute_standard_errors(
            sample_covariance_matrix, use_torch=False
        )
        assert isinstance(std_errors_numpy, torch.Tensor)
        assert len(std_errors_numpy) == 4
        assert torch.all(std_errors_numpy > 0)

        # Results should be similar
        torch.testing.assert_close(std_errors_torch, std_errors_numpy, rtol=1e-5, atol=1e-8)

        # Should be square root of diagonal
        expected = torch.sqrt(torch.diag(sample_covariance_matrix))
        torch.testing.assert_close(std_errors_numpy, expected, rtol=1e-5, atol=1e-8)

    def test_compute_test_statistics(self, sample_coefficients):
        """Test test statistic computation."""
        std_errors = torch.tensor([0.5, 0.3, 0.2, 0.8], dtype=torch.float32)

        # Test t-statistics
        t_stats = compute_test_statistics(sample_coefficients, std_errors, "t")
        expected_t = sample_coefficients / std_errors
        torch.testing.assert_close(t_stats, expected_t, rtol=1e-5, atol=1e-8)

        # Test z-statistics (should be same calculation)
        z_stats = compute_test_statistics(sample_coefficients, std_errors, "z")
        torch.testing.assert_close(z_stats, t_stats, rtol=1e-5, atol=1e-8)

        # Test with zero standard errors (should not crash)
        std_errors_zero = torch.tensor([0.5, 0.0, 0.2, 0.8], dtype=torch.float32)
        t_stats_safe = compute_test_statistics(
            sample_coefficients, std_errors_zero, "t"
        )
        assert torch.isfinite(t_stats_safe).all()

    def test_compute_p_values(self, sample_coefficients):
        """Test p-value computation."""
        std_errors = torch.tensor([0.5, 0.3, 0.2, 0.8], dtype=torch.float32)
        t_stats = compute_test_statistics(sample_coefficients, std_errors)

        # Test t-distribution p-values
        dof = 95  # degrees of freedom
        p_values_t = compute_p_values(t_stats, dof, "t")

        assert isinstance(p_values_t, np.ndarray)
        assert len(p_values_t) == 4
        assert np.all(p_values_t >= 0)
        assert np.all(p_values_t <= 1)

        # Test z-distribution p-values
        p_values_z = compute_p_values(t_stats, dof, "z")
        assert isinstance(p_values_z, np.ndarray)
        assert len(p_values_z) == 4

        # Manually verify one calculation
        t_stat_manual = sample_coefficients[0] / std_errors[0]
        p_value_manual = 2 * (1 - stats.t.cdf(abs(t_stat_manual.item()), df=dof))
        np.testing.assert_allclose(p_values_t[0], p_value_manual, rtol=1e-6)

    def test_compute_confidence_intervals(self, sample_coefficients):
        """Test confidence interval computation."""
        std_errors = torch.tensor([0.5, 0.3, 0.2, 0.8], dtype=torch.float32)

        # Test t-distribution intervals
        lower_t, upper_t = compute_confidence_intervals(
            sample_coefficients, std_errors, alpha=0.05, dof=95, distribution="t"
        )

        assert isinstance(lower_t, np.ndarray)
        assert isinstance(upper_t, np.ndarray)
        assert len(lower_t) == len(upper_t) == 4
        assert np.all(lower_t < upper_t)

        # Coefficients should be between bounds
        coef_np = sample_coefficients.numpy()
        assert np.all(lower_t <= coef_np)
        assert np.all(coef_np <= upper_t)

        # Test z-distribution intervals
        lower_z, upper_z = compute_confidence_intervals(
            sample_coefficients, std_errors, alpha=0.05, distribution="z"
        )

        # Z intervals should be narrower than t intervals (for reasonable dof)
        interval_width_t = upper_t - lower_t
        interval_width_z = upper_z - lower_z
        assert np.all(interval_width_z <= interval_width_t)

        # Test different confidence levels
        lower_99, upper_99 = compute_confidence_intervals(
            sample_coefficients, std_errors, alpha=0.01, dof=95
        )

        # 99% intervals should be wider than 95% intervals
        interval_width_95 = upper_t - lower_t
        interval_width_99 = upper_99 - lower_99
        assert np.all(interval_width_99 > interval_width_95)

    def test_compute_model_statistics_linear(self):
        """Test model statistics for linear regression."""
        # Create synthetic data
        n_obs = 100
        y_true = torch.randn(n_obs)
        y_pred = y_true + 0.1 * torch.randn(n_obs)  # Add some noise

        log_likelihood = -50.0  # Example value
        n_params = 5

        stats_dict = compute_model_statistics(
            y_true, y_pred, log_likelihood, n_params, "linear"
        )

        # Check required keys
        required_keys = [
            "log_likelihood",
            "aic",
            "bic",
            "n_obs",
            "n_params",
            "r_squared",
            "adj_r_squared",
            "mse",
            "rmse",
        ]
        for key in required_keys:
            assert key in stats_dict

        # Check values are reasonable
        assert stats_dict["n_obs"] == n_obs
        assert stats_dict["n_params"] == n_params
        assert stats_dict["log_likelihood"] == log_likelihood
        assert 0 <= stats_dict["r_squared"] <= 1
        assert stats_dict["mse"] > 0
        assert stats_dict["rmse"] > 0
        assert stats_dict["rmse"] == np.sqrt(stats_dict["mse"])

        # AIC and BIC formulas
        expected_aic = 2 * n_params - 2 * log_likelihood
        expected_bic = n_params * np.log(n_obs) - 2 * log_likelihood
        assert abs(stats_dict["aic"] - expected_aic) < 1e-6
        assert abs(stats_dict["bic"] - expected_bic) < 1e-6

    def test_compute_model_statistics_logistic(self):
        """Test model statistics for logistic regression."""
        # Create synthetic binary data
        n_obs = 100
        y_true = torch.randint(0, 2, (n_obs,)).float()
        y_pred = torch.rand(n_obs)  # Probabilities

        log_likelihood = -40.0  # Example value
        n_params = 4

        stats_dict = compute_model_statistics(
            y_true, y_pred, log_likelihood, n_params, "logistic"
        )

        # Check required keys
        required_keys = [
            "log_likelihood",
            "aic",
            "bic",
            "n_obs",
            "n_params",
            "accuracy",
        ]
        for key in required_keys:
            assert key in stats_dict

        # Check values are reasonable
        assert stats_dict["n_obs"] == n_obs
        assert stats_dict["n_params"] == n_params
        assert 0 <= stats_dict["accuracy"] <= 1

    def test_format_summary_table(self, sample_coefficients):
        """Test summary table formatting."""
        std_errors = torch.tensor([0.5, 0.3, 0.2, 0.8], dtype=torch.float32)
        feature_names = ["const", "x1", "x2", "x3"]
        model_stats = {
            "log_likelihood": -50.2,
            "aic": 108.4,
            "bic": 119.1,
            "n_obs": 100,
            "n_params": 4,
            "r_squared": 0.75,
            "adj_r_squared": 0.73,
        }
        dof = 96

        summary = format_summary_table(
            sample_coefficients, std_errors, feature_names, model_stats, dof
        )

        assert isinstance(summary, str)

        # Check that important elements are in the summary
        assert "Statistical Summary" in summary
        assert "coef" in summary
        assert "std err" in summary
        assert "R-squared" in summary
        assert "Log-Likelihood" in summary
        assert "AIC" in summary
        assert "BIC" in summary

        # Check that all feature names appear
        for name in feature_names:
            assert name in summary

        # Check that coefficients appear (approximately)
        for coef in sample_coefficients:
            assert f"{coef.item():.3f}" in summary

    def test_compute_residuals(self):
        """Test residual computation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.1])

        # Test raw residuals
        raw_residuals = compute_residuals(y_true, y_pred, "raw")
        expected_raw = y_true - y_pred
        torch.testing.assert_close(raw_residuals, expected_raw, rtol=1e-5, atol=1e-8)

        # Test standardized residuals
        std_residuals = compute_residuals(y_true, y_pred, "standardized")
        mse = torch.mean((y_true - y_pred) ** 2)
        expected_std = (y_true - y_pred) / torch.sqrt(mse)
        torch.testing.assert_close(std_residuals, expected_std, rtol=1e-5, atol=1e-8)

        # Test studentized residuals (simplified version)
        student_residuals = compute_residuals(y_true, y_pred, "studentized")
        # Should be same as standardized in this simplified implementation
        torch.testing.assert_close(student_residuals, std_residuals, rtol=1e-5, atol=1e-8)

        # Test invalid residual type
        with pytest.raises(ValueError):
            compute_residuals(y_true, y_pred, "invalid")

    def test_compute_information_criteria(self):
        """Test information criteria computation."""
        log_likelihood = -123.45
        n_params = 5
        n_obs = 100

        criteria = compute_information_criteria(log_likelihood, n_params, n_obs)

        # Check that all criteria are computed
        assert "aic" in criteria
        assert "bic" in criteria
        assert "aicc" in criteria

        # Check formulas
        expected_aic = 2 * n_params - 2 * log_likelihood
        expected_bic = n_params * np.log(n_obs) - 2 * log_likelihood

        assert abs(criteria["aic"] - expected_aic) < 1e-6
        assert abs(criteria["bic"] - expected_bic) < 1e-6

        # For large samples, AICc should be close to AIC
        assert abs(criteria["aicc"] - criteria["aic"]) < 1.0

        # Test small sample correction
        criteria_small = compute_information_criteria(log_likelihood, n_params, 15)
        # AICc should be noticeably different from AIC for small samples
        assert abs(criteria_small["aicc"] - criteria_small["aic"]) > 1.0

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with very small covariance matrix
        small_cov = torch.eye(2) * 1e-10
        std_errors = compute_standard_errors(small_cov)
        assert torch.all(torch.isfinite(std_errors))

        # Test with zero coefficients
        zero_coef = torch.zeros(3)
        std_err = torch.ones(3)
        t_stats = compute_test_statistics(zero_coef, std_err)
        expected_zeros = torch.zeros(3)
        torch.testing.assert_close(t_stats, expected_zeros, rtol=1e-5, atol=1e-8)

        # Test p-values with zero test statistics
        p_vals = compute_p_values(t_stats, 95, "t")
        # P-values for zero t-statistics should be 1.0
        np.testing.assert_allclose(p_vals, 1.0, rtol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large coefficients
        large_coef = torch.tensor([1e6, -1e6, 1e-6, -1e-6])
        large_std = torch.tensor([1e3, 1e3, 1e-9, 1e-9])

        t_stats = compute_test_statistics(large_coef, large_std)
        assert torch.all(torch.isfinite(t_stats))

        p_vals = compute_p_values(t_stats, 100, "t")
        assert np.all(np.isfinite(p_vals))
        assert np.all(p_vals >= 0)
        assert np.all(p_vals <= 1)

    def test_distribution_consistency(self):
        """Test consistency between t and z distributions for large dof."""
        coef = torch.tensor([2.0, -1.5])
        std_err = torch.tensor([0.5, 0.3])

        # For large degrees of freedom, t and z should be similar
        large_dof = 1000

        # Confidence intervals
        lower_t, upper_t = compute_confidence_intervals(
            coef, std_err, dof=large_dof, distribution="t"
        )
        lower_z, upper_z = compute_confidence_intervals(coef, std_err, distribution="z")

        # Should be very close for large dof
        np.testing.assert_allclose(lower_t, lower_z, rtol=1e-2)
        np.testing.assert_allclose(upper_t, upper_z, rtol=1e-2)

        # P-values
        t_stats = compute_test_statistics(coef, std_err)
        p_vals_t = compute_p_values(t_stats, large_dof, "t")
        p_vals_z = compute_p_values(t_stats, large_dof, "z")

        np.testing.assert_allclose(p_vals_t, p_vals_z, rtol=1e-2)

    def test_summary_table_content(self):
        """Test that summary table contains expected statistical content."""
        coef = torch.tensor([1.5, -0.8, 2.3])
        std_err = torch.tensor([0.2, 0.3, 0.4])
        feature_names = ["const", "feature1", "feature2"]
        model_stats = {
            "log_likelihood": -89.5,
            "aic": 185.0,
            "bic": 191.2,
            "n_obs": 150,
            "n_params": 3,
            "r_squared": 0.823,
            "adj_r_squared": 0.818,
        }
        dof = 147

        summary = format_summary_table(
            coef, std_err, feature_names, model_stats, dof, "t"
        )

        # Check for specific statistical values
        assert "1.500" in summary  # Coefficient value
        assert "0.200" in summary  # Standard error
        assert "0.823" in summary  # R-squared
        assert "185.0" in summary  # AIC
        assert "147" in summary  # Degrees of freedom

        # Check statistical significance indicators
        lines = summary.split("\n")
        coef_lines = [
            line for line in lines if any(name in line for name in feature_names)
        ]
        assert len(coef_lines) >= 3  # Should have lines for all coefficients
