"""
Tests for scorecard integration module.
"""

import numpy as np
import pandas as pd
import pytest
from analytics_toolkit.scorecard_integration import ScorecardIntegrator


@pytest.fixture
def simple_scorecard_data():
    """Create simple scorecard data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Create three different scorecards with different predictive power
    scorecard1 = np.random.randn(n_samples)
    scorecard2 = np.random.randn(n_samples) + 0.5
    scorecard3 = np.random.randn(n_samples) - 0.2

    # Create target that's related to scorecards
    target = (
        0.4 * scorecard1
        + 0.3 * scorecard2
        + 0.3 * scorecard3
        + np.random.randn(n_samples) * 0.1
        > 0
    ).astype(int)

    scores_df = pd.DataFrame(
        {
            "scorecard_1": scorecard1,
            "scorecard_2": scorecard2,
            "scorecard_3": scorecard3,
        }
    )

    return scores_df, target


@pytest.fixture
def regression_scorecard_data():
    """Create scorecard data for regression target."""
    np.random.seed(42)
    n_samples = 150

    scorecard1 = np.random.randn(n_samples)
    scorecard2 = np.random.randn(n_samples) * 2

    # Continuous target
    target = 0.6 * scorecard1 + 0.4 * scorecard2 + np.random.randn(n_samples) * 0.2

    scores_df = pd.DataFrame({"score_a": scorecard1, "score_b": scorecard2})

    return scores_df, target


@pytest.fixture
def scorecard_data_with_missing():
    """Create scorecard data with missing values."""
    np.random.seed(42)
    n_samples = 100

    scores_df = pd.DataFrame(
        {
            "scorecard_1": np.random.randn(n_samples),
            "scorecard_2": np.random.randn(n_samples),
            "scorecard_3": np.random.randn(n_samples),
        }
    )

    # Add missing values
    scores_df.loc[10:15, "scorecard_1"] = np.nan
    scores_df.loc[20:22, "scorecard_2"] = np.nan

    target = np.random.randint(0, 2, n_samples)

    return scores_df, target


class TestScorecardIntegratorInit:
    """Test ScorecardIntegrator initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        integrator = ScorecardIntegrator(["score1", "score2"])
        assert integrator.scorecard_columns == ["score1", "score2"]
        assert integrator.objective == "log_loss"
        assert integrator.weight_bounds == {}
        assert integrator.fixed_weights == {}
        assert not integrator.is_fitted_

    def test_initialization_with_params(self):
        """Test initialization with all parameters."""
        weight_bounds = {"score1": (0.2, 0.8), "score2": (0.1, 0.9)}
        fixed_weights = {"score3": 0.5}

        integrator = ScorecardIntegrator(
            scorecard_columns=["score1", "score2", "score3"],
            objective="auc",
            weight_bounds=weight_bounds,
            fixed_weights=fixed_weights,
        )

        assert integrator.objective == "auc"
        assert integrator.weight_bounds == weight_bounds
        assert integrator.fixed_weights == fixed_weights

    def test_invalid_objective(self):
        """Test initialization with invalid objective."""
        with pytest.raises(ValueError, match="objective must be one of"):
            ScorecardIntegrator(["score1"], objective="invalid")

    def test_invalid_fixed_weights(self):
        """Test initialization with fixed weights for unknown scorecards."""
        with pytest.raises(
            ValueError, match="Fixed weights specified for unknown scorecards"
        ):
            ScorecardIntegrator(
                ["score1", "score2"], fixed_weights={"unknown_score": 0.5}
            )

    def test_invalid_weight_bounds(self):
        """Test initialization with bounds for unknown scorecards."""
        with pytest.raises(
            ValueError, match="Weight bounds specified for unknown scorecards"
        ):
            ScorecardIntegrator(
                ["score1", "score2"], weight_bounds={"unknown_score": (0.1, 0.9)}
            )


class TestScorecardIntegratorFit:
    """Test ScorecardIntegrator fitting functionality."""

    def test_basic_fit(self, simple_scorecard_data):
        """Test basic fitting."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])
        integrator.fit(scores_df, target)

        assert integrator.is_fitted_
        assert integrator.optimal_weights_ is not None
        assert len(integrator.optimal_weights_) == 3
        assert np.allclose(np.sum(integrator.optimal_weights_), 1.0)
        assert integrator.optimization_result_ is not None

    def test_fit_with_auc_objective(self, simple_scorecard_data):
        """Test fitting with AUC objective."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(
            ["scorecard_1", "scorecard_2", "scorecard_3"], objective="auc"
        )
        integrator.fit(scores_df, target)

        assert integrator.is_fitted_
        assert np.allclose(np.sum(integrator.optimal_weights_), 1.0)

    def test_fit_with_fixed_weights(self, simple_scorecard_data):
        """Test fitting with fixed weights."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(
            ["scorecard_1", "scorecard_2", "scorecard_3"],
            fixed_weights={"scorecard_1": 0.5},
        )
        integrator.fit(scores_df, target)

        weights = integrator.get_weights()
        assert weights["scorecard_1"] == pytest.approx(0.5, abs=1e-6)
        assert np.allclose(np.sum(integrator.optimal_weights_), 1.0)

    def test_fit_with_weight_bounds(self, simple_scorecard_data):
        """Test fitting with weight bounds."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(
            ["scorecard_1", "scorecard_2", "scorecard_3"],
            weight_bounds={"scorecard_1": (0.4, 0.6), "scorecard_2": (0.1, 0.3)},
        )
        integrator.fit(scores_df, target)

        weights = integrator.get_weights()
        assert (
            0.4 <= weights["scorecard_1"] <= 0.6 + 1e-10
        )  # Allow for floating point precision
        assert (
            0.1 <= weights["scorecard_2"] <= 0.3 + 1e-10
        )  # Allow for floating point precision

    def test_fit_with_missing_values(self, scorecard_data_with_missing):
        """Test fitting with missing values."""
        scores_df, target = scorecard_data_with_missing

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])

        with pytest.warns(UserWarning, match="Missing values detected"):
            integrator.fit(scores_df, target)

        assert integrator.is_fitted_

    def test_fit_missing_columns(self, simple_scorecard_data):
        """Test fitting with missing scorecard columns."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(
            ["scorecard_1", "scorecard_2", "missing_scorecard"]
        )

        with pytest.raises(ValueError, match="Missing scorecard columns"):
            integrator.fit(scores_df, target)

    def test_fit_mismatched_lengths(self, simple_scorecard_data):
        """Test fitting with mismatched lengths."""
        scores_df, target = simple_scorecard_data
        short_target = target[:100]

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])

        with pytest.raises(ValueError, match="must have the same length"):
            integrator.fit(scores_df, short_target)


class TestScorecardIntegratorPredict:
    """Test ScorecardIntegrator prediction functionality."""

    def test_predict_basic(self, simple_scorecard_data):
        """Test basic prediction."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])
        integrator.fit(scores_df, target)

        predictions = integrator.predict(scores_df)

        assert len(predictions) == len(scores_df)
        assert isinstance(predictions, np.ndarray)

    def test_predict_not_fitted(self, simple_scorecard_data):
        """Test prediction without fitting."""
        scores_df, _ = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])

        with pytest.raises(ValueError, match="must be fitted before calling predict"):
            integrator.predict(scores_df)

    def test_predict_missing_columns(self, simple_scorecard_data):
        """Test prediction with missing columns."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])
        integrator.fit(scores_df, target)

        # Remove a column
        incomplete_df = scores_df.drop("scorecard_2", axis=1)

        with pytest.raises(ValueError, match="Missing scorecard columns"):
            integrator.predict(incomplete_df)

    def test_predict_with_missing_values(self, simple_scorecard_data):
        """Test prediction with missing values."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])
        integrator.fit(scores_df, target)

        # Add missing values to prediction data
        test_df = scores_df.copy()
        test_df.loc[0, "scorecard_1"] = np.nan

        with pytest.warns(UserWarning, match="Missing values detected"):
            predictions = integrator.predict(test_df)

        assert np.isnan(predictions[0])


class TestScorecardIntegratorUtilityMethods:
    """Test utility methods of ScorecardIntegrator."""

    def test_get_weights(self, simple_scorecard_data):
        """Test getting weights as dictionary."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])
        integrator.fit(scores_df, target)

        weights = integrator.get_weights()

        assert isinstance(weights, dict)
        assert set(weights.keys()) == {"scorecard_1", "scorecard_2", "scorecard_3"}
        assert np.allclose(sum(weights.values()), 1.0)

    def test_get_weights_not_fitted(self):
        """Test getting weights without fitting."""
        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2"])

        with pytest.raises(ValueError, match="must be fitted before getting weights"):
            integrator.get_weights()

    def test_get_combined_scorecard(self, simple_scorecard_data):
        """Test getting combined scorecard function."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2", "scorecard_3"])
        integrator.fit(scores_df, target)

        combined_func = integrator.get_combined_scorecard()

        assert callable(combined_func)

        # Test the function
        predictions1 = combined_func(scores_df)
        predictions2 = integrator.predict(scores_df)

        np.testing.assert_array_equal(predictions1, predictions2)

    def test_get_combined_scorecard_not_fitted(self):
        """Test getting combined scorecard without fitting."""
        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2"])

        with pytest.raises(
            ValueError, match="must be fitted before getting combined scorecard"
        ):
            integrator.get_combined_scorecard()

    def test_summary(self, simple_scorecard_data):
        """Test summary method."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(
            ["scorecard_1", "scorecard_2", "scorecard_3"],
            weight_bounds={"scorecard_1": (0.2, 0.8)},
            fixed_weights={"scorecard_2": 0.3},
        )
        integrator.fit(scores_df, target)

        summary = integrator.summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3
        assert set(summary.columns) == {
            "scorecard",
            "weight",
            "is_fixed",
            "lower_bound",
            "upper_bound",
        }
        assert summary[summary["scorecard"] == "scorecard_2"]["is_fixed"].iloc[0]
        assert not summary[summary["scorecard"] == "scorecard_1"]["is_fixed"].iloc[0]

    def test_summary_not_fitted(self):
        """Test summary without fitting."""
        integrator = ScorecardIntegrator(["scorecard_1", "scorecard_2"])

        with pytest.raises(ValueError, match="must be fitted before getting summary"):
            integrator.summary()


class TestScorecardIntegratorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_scorecard(self):
        """Test with single scorecard."""
        np.random.seed(42)
        scores_df = pd.DataFrame({"single_score": np.random.randn(100)})
        target = np.random.randint(0, 2, 100)

        integrator = ScorecardIntegrator(["single_score"])
        integrator.fit(scores_df, target)

        weights = integrator.get_weights()
        assert weights["single_score"] == pytest.approx(1.0)

    def test_all_fixed_weights(self, simple_scorecard_data):
        """Test with all weights fixed."""
        scores_df, target = simple_scorecard_data

        integrator = ScorecardIntegrator(
            ["scorecard_1", "scorecard_2", "scorecard_3"],
            fixed_weights={"scorecard_1": 0.5, "scorecard_2": 0.3, "scorecard_3": 0.2},
        )
        integrator.fit(scores_df, target)

        weights = integrator.get_weights()
        assert weights["scorecard_1"] == pytest.approx(0.5, abs=1e-6)
        assert weights["scorecard_2"] == pytest.approx(0.3, abs=1e-6)
        assert weights["scorecard_3"] == pytest.approx(0.2, abs=1e-6)

    def test_perfect_separation(self):
        """Test with perfectly separable data."""
        scores_df = pd.DataFrame(
            {
                "perfect_score": [1, 1, 1, -1, -1, -1],
            }
        )
        target = np.array([1, 1, 1, 0, 0, 0])

        integrator = ScorecardIntegrator(["perfect_score"])
        integrator.fit(scores_df, target)

        # Should still work even with perfect separation
        assert integrator.is_fitted_
        assert integrator.get_weights()["perfect_score"] == pytest.approx(1.0)

    def test_constant_scores(self):
        """Test with constant scorecard values."""
        scores_df = pd.DataFrame(
            {"constant_score": [1.0] * 100, "variable_score": np.random.randn(100)}
        )
        target = np.random.randint(0, 2, 100)

        integrator = ScorecardIntegrator(["constant_score", "variable_score"])
        integrator.fit(scores_df, target)

        # Should handle constant scores gracefully
        assert integrator.is_fitted_
        weights = integrator.get_weights()
        # Weights should be valid and sum to 1
        assert all(w >= 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)


class TestScorecardIntegratorNormalization:
    """Test weight normalization functionality."""

    def test_normalize_weights_normal(self):
        """Test weight normalization with normal weights."""
        integrator = ScorecardIntegrator(["score1", "score2"])
        weights = np.array([0.3, 0.7])
        normalized = integrator._normalize_weights(weights)

        np.testing.assert_array_almost_equal(normalized, [0.3, 0.7])
        assert np.allclose(np.sum(normalized), 1.0)

    def test_normalize_weights_zero_sum(self):
        """Test weight normalization with zero sum."""
        integrator = ScorecardIntegrator(["score1", "score2"])
        weights = np.array([0.0, 0.0])
        normalized = integrator._normalize_weights(weights)

        np.testing.assert_array_almost_equal(normalized, [0.5, 0.5])
        assert np.allclose(np.sum(normalized), 1.0)

    def test_normalize_weights_unnormalized(self):
        """Test weight normalization with unnormalized weights."""
        integrator = ScorecardIntegrator(["score1", "score2"])
        weights = np.array([2.0, 4.0])
        normalized = integrator._normalize_weights(weights)

        expected = np.array([2.0, 4.0]) / 6.0
        np.testing.assert_array_almost_equal(normalized, expected)
        assert np.allclose(np.sum(normalized), 1.0)
