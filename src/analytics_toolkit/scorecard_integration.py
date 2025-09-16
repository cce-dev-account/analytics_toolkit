"""
Scorecard Integration Module

This module provides functionality to combine multiple scorecards using weighted sums,
with optimization capabilities to find optimal weights based on target variables.
"""

import warnings
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score


class ScorecardIntegrator:
    """
    A class to integrate multiple scorecards using weighted combinations.

    This class allows for:
    - Weighted combination of scorecard scores
    - Optimization of weights based on target variables
    - Constraint handling (bounds, fixed weights)
    - Multiple optimization objectives (log likelihood, AUC)
    """

    def __init__(
        self,
        scorecard_columns: list[str],
        objective: str = "log_loss",
        weight_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the ScorecardIntegrator.

        Parameters
        ----------
        scorecard_columns : List[str]
            List of column names in the DataFrame that contain scorecard scores
        objective : str, default 'log_loss'
            Optimization objective. Options: 'log_loss', 'auc'
        weight_bounds : Dict[str, Tuple[float, float]], optional
            Dictionary mapping scorecard names to (min_weight, max_weight) bounds
        fixed_weights : Dict[str, float], optional
            Dictionary mapping scorecard names to fixed weight values
        """
        self.scorecard_columns = scorecard_columns
        self.objective = objective
        self.weight_bounds = weight_bounds or {}
        self.fixed_weights = fixed_weights or {}

        # Validate inputs
        self._validate_inputs()

        # Initialize optimization results
        self.optimal_weights_ = None
        self.optimization_result_ = None
        self.is_fitted_ = False

    def _validate_inputs(self):
        """Validate input parameters."""
        valid_objectives = ["log_loss", "auc"]
        if self.objective not in valid_objectives:
            raise ValueError(
                f"objective must be one of {valid_objectives}, got {self.objective}"
            )

        # Check that fixed weights are for valid scorecards
        invalid_fixed = set(self.fixed_weights.keys()) - set(self.scorecard_columns)
        if invalid_fixed:
            raise ValueError(
                f"Fixed weights specified for unknown scorecards: {invalid_fixed}"
            )

        # Check that weight bounds are for valid scorecards
        invalid_bounds = set(self.weight_bounds.keys()) - set(self.scorecard_columns)
        if invalid_bounds:
            raise ValueError(
                f"Weight bounds specified for unknown scorecards: {invalid_bounds}"
            )

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1."""
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.ones_like(weights) / len(weights)
        return weights / weight_sum

    def _objective_function(
        self, weights: np.ndarray, scores_df: pd.DataFrame, target: np.ndarray
    ) -> float:
        """
        Compute the objective function value for given weights.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector for scorecards
        scores_df : pd.DataFrame
            DataFrame containing scorecard scores
        target : np.ndarray
            Target variable

        Returns
        -------
        float
            Objective function value
        """
        # Normalize weights
        normalized_weights = self._normalize_weights(weights)

        # Compute weighted sum
        weighted_scores = np.dot(
            scores_df[self.scorecard_columns].values, normalized_weights
        )

        try:
            if self.objective == "log_loss":
                # For log loss, we need probabilities between 0 and 1
                # Apply sigmoid transformation to ensure this
                probabilities = 1 / (1 + np.exp(-weighted_scores))
                # Clip to avoid numerical issues
                probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
                return log_loss(target, probabilities)

            elif self.objective == "auc":
                # For AUC, we want to maximize, so return negative AUC
                try:
                    auc_score = roc_auc_score(target, weighted_scores)
                    return -auc_score  # Negative because we're minimizing
                except ValueError:
                    # If AUC can't be computed (e.g., only one class), return large penalty
                    return 1.0

        except Exception as e:
            warnings.warn(f"Error in objective function: {e}")
            return 1e6  # Large penalty for invalid combinations

    def _create_constraints(self) -> list[dict[str, Any]]:
        """Create optimization constraints."""
        constraints = []

        # Constraint: weights sum to 1
        def weight_sum_constraint(weights):
            return np.sum(weights) - 1.0

        constraints.append({"type": "eq", "fun": weight_sum_constraint})

        return constraints

    def _create_bounds(self) -> list[tuple[float, float]]:
        """Create weight bounds for optimization."""
        bounds = []

        for scorecard in self.scorecard_columns:
            if scorecard in self.fixed_weights:
                # Fixed weight - set both bounds to the fixed value
                fixed_val = self.fixed_weights[scorecard]
                bounds.append((fixed_val, fixed_val))
            elif scorecard in self.weight_bounds:
                # User-specified bounds
                bounds.append(self.weight_bounds[scorecard])
            else:
                # Default bounds: between 0 and 1
                bounds.append((0.0, 1.0))

        return bounds

    def fit(
        self, scores_df: pd.DataFrame, target: np.ndarray | pd.Series
    ) -> "ScorecardIntegrator":
        """
        Fit the scorecard integrator by finding optimal weights.

        Parameters
        ----------
        scores_df : pd.DataFrame
            DataFrame containing scorecard scores for each observation
        target : array-like
            Target variable for optimization

        Returns
        -------
        self : ScorecardIntegrator
            Returns self for method chaining
        """
        # Validate inputs
        missing_cols = set(self.scorecard_columns) - set(scores_df.columns)
        if missing_cols:
            raise ValueError(f"Missing scorecard columns in DataFrame: {missing_cols}")

        if len(scores_df) != len(target):
            raise ValueError("scores_df and target must have the same length")

        # Handle missing values
        if scores_df[self.scorecard_columns].isnull().any().any():
            warnings.warn(
                "Missing values detected in scorecard scores. Rows with missing values will be dropped."
            )
            valid_idx = scores_df[self.scorecard_columns].notna().all(axis=1)
            scores_df = scores_df[valid_idx].copy()
            target = np.array(target)[valid_idx]

        # Convert target to numpy array
        target = np.array(target)

        # Initialize weights - start with equal weights, but respect fixed weights
        initial_weights = np.ones(len(self.scorecard_columns))
        for i, scorecard in enumerate(self.scorecard_columns):
            if scorecard in self.fixed_weights:
                initial_weights[i] = self.fixed_weights[scorecard]

        # Normalize initial weights
        initial_weights = self._normalize_weights(initial_weights)

        # Set up optimization
        bounds = self._create_bounds()
        constraints = self._create_constraints()

        # Optimize
        result = minimize(
            fun=self._objective_function,
            x0=initial_weights,
            args=(scores_df, target),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "disp": False},
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Store results
        self.optimal_weights_ = self._normalize_weights(result.x)
        self.optimization_result_ = result
        self.is_fitted_ = True

        return self

    def predict(self, scores_df: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted weighted combination of scorecards.

        Parameters
        ----------
        scores_df : pd.DataFrame
            DataFrame containing scorecard scores

        Returns
        -------
        np.ndarray
            Combined scores
        """
        if not self.is_fitted_:
            raise ValueError(
                "ScorecardIntegrator must be fitted before calling predict"
            )

        missing_cols = set(self.scorecard_columns) - set(scores_df.columns)
        if missing_cols:
            raise ValueError(f"Missing scorecard columns in DataFrame: {missing_cols}")

        # Handle missing values
        if scores_df[self.scorecard_columns].isnull().any().any():
            warnings.warn(
                "Missing values detected in scorecard scores. These will result in NaN predictions."
            )

        return np.dot(scores_df[self.scorecard_columns].values, self.optimal_weights_)

    def get_weights(self) -> dict[str, float]:
        """
        Get the optimal weights as a dictionary.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping scorecard names to optimal weights
        """
        if not self.is_fitted_:
            raise ValueError(
                "ScorecardIntegrator must be fitted before getting weights"
            )

        return dict(zip(self.scorecard_columns, self.optimal_weights_, strict=False))

    def get_combined_scorecard(self) -> Callable[[pd.DataFrame], np.ndarray]:
        """
        Get a callable function that applies the optimal weighted combination.

        Returns
        -------
        Callable
            Function that takes a DataFrame and returns combined scores
        """
        if not self.is_fitted_:
            raise ValueError(
                "ScorecardIntegrator must be fitted before getting combined scorecard"
            )

        def combined_scorecard(scores_df: pd.DataFrame) -> np.ndarray:
            return self.predict(scores_df)

        return combined_scorecard

    def summary(self) -> pd.DataFrame:
        """
        Get a summary of the optimization results.

        Returns
        -------
        pd.DataFrame
            Summary table with scorecard names, weights, bounds, and fixed status
        """
        if not self.is_fitted_:
            raise ValueError(
                "ScorecardIntegrator must be fitted before getting summary"
            )

        summary_data = []
        for i, scorecard in enumerate(self.scorecard_columns):
            row = {
                "scorecard": scorecard,
                "weight": self.optimal_weights_[i],
                "is_fixed": scorecard in self.fixed_weights,
                "lower_bound": self.weight_bounds.get(scorecard, (0.0, 1.0))[0],
                "upper_bound": self.weight_bounds.get(scorecard, (0.0, 1.0))[1],
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)

        return summary_df
