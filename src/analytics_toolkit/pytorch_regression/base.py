"""
Base regression class providing common functionality for all regression models.
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Any
from .utils import detect_categorical_columns, create_dummy_variables, to_tensor
from .stats import compute_standard_errors, compute_confidence_intervals, format_summary_table


class BaseRegression(ABC, nn.Module):
    """
    Base class for all regression models.

    Provides common functionality including data preprocessing, device management,
    statistical inference, and result formatting.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        penalty: str = 'none',
        alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: str = 'auto',
        categorical_encoding: str = 'dummy'
    ):
        """
        Initialize base regression model.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether to fit an intercept term.
        penalty : str, default='none'
            Regularization penalty ('none', 'l1', 'l2', 'elastic_net').
        alpha : float, default=0.01
            Regularization strength.
        max_iter : int, default=1000
            Maximum number of iterations for optimization.
        tol : float, default=1e-4
            Tolerance for convergence.
        device : str, default='auto'
            Device for computation ('auto', 'cpu', 'cuda').
        categorical_encoding : str, default='dummy'
            Method for encoding categorical variables ('dummy', 'target').
        """
        super().__init__()

        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.categorical_encoding = categorical_encoding

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model state
        self.is_fitted_ = False
        self.feature_names_ = None
        self.n_features_in_ = None
        self.coef_ = None
        self.intercept_ = None
        self.standard_errors_ = None
        self.covariance_matrix_ = None
        self.log_likelihood_ = None
        self.aic_ = None
        self.bic_ = None
        self.encoding_mappings_ = {}

        # Validation
        valid_penalties = ['none', 'l1', 'l2', 'elastic_net']
        if self.penalty not in valid_penalties:
            raise ValueError(f"penalty must be one of {valid_penalties}")

        valid_encodings = ['dummy', 'target']
        if self.categorical_encoding not in valid_encodings:
            raise ValueError(f"categorical_encoding must be one of {valid_encodings}")

    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Validate and preprocess input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,), optional
            Target vector.

        Returns
        -------
        X_tensor : torch.Tensor
            Preprocessed feature matrix.
        y_tensor : torch.Tensor or None
            Preprocessed target vector.
        feature_names : list or None
            Feature names if available.
        """
        # Extract feature names if available
        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()

            # Handle categorical variables
            categorical_cols = detect_categorical_columns(X)
            if categorical_cols:
                if self.categorical_encoding == 'dummy':
                    X, feature_names, mappings = create_dummy_variables(X, categorical_cols)
                    if not self.is_fitted_:
                        self.encoding_mappings_.update(mappings)
                else:
                    raise NotImplementedError("Target encoding not implemented yet")

        # Convert to tensors
        X_tensor = to_tensor(X, device=self.device, dtype=torch.float32)
        y_tensor = to_tensor(y, device=self.device, dtype=torch.float32) if y is not None else None

        # Validate shapes
        if X_tensor.dim() != 2:
            raise ValueError("X must be 2-dimensional")

        if y_tensor is not None:
            if y_tensor.dim() != 1:
                raise ValueError("y must be 1-dimensional")
            if X_tensor.shape[0] != y_tensor.shape[0]:
                raise ValueError("X and y must have the same number of samples")

        # Check for missing values
        if torch.isnan(X_tensor).any():
            raise ValueError("X contains NaN values")
        if y_tensor is not None and torch.isnan(y_tensor).any():
            raise ValueError("y contains NaN values")

        return X_tensor, y_tensor, feature_names

    def _add_intercept(self, X: torch.Tensor) -> torch.Tensor:
        """Add intercept column to feature matrix."""
        if self.fit_intercept:
            ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
            return torch.cat([ones, X], dim=1)
        return X

    def _get_regularization_penalty(self, coef: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization penalty.

        Parameters
        ----------
        coef : torch.Tensor
            Model coefficients (excluding intercept if fit_intercept=True).

        Returns
        -------
        penalty : torch.Tensor
            Regularization penalty value.
        """
        if self.penalty == 'none':
            return torch.tensor(0.0, device=self.device)
        elif self.penalty == 'l1':
            return self.alpha * torch.sum(torch.abs(coef))
        elif self.penalty == 'l2':
            return self.alpha * torch.sum(coef ** 2)
        elif self.penalty == 'elastic_net':
            # Default l1_ratio=0.5 for now (will be added as parameter later)
            l1_ratio = 0.5
            l1_penalty = l1_ratio * torch.sum(torch.abs(coef))
            l2_penalty = (1 - l1_ratio) * torch.sum(coef ** 2)
            return self.alpha * (l1_penalty + l2_penalty)
        else:
            raise ValueError(f"Unknown penalty: {self.penalty}")

    def _check_condition_number(self, X: torch.Tensor) -> None:
        """Check condition number and warn if poorly conditioned."""
        try:
            # Use double precision for numerical stability
            X_double = X.double()
            if self.fit_intercept:
                X_double = X_double[:, 1:]  # Exclude intercept for condition number

            # Compute condition number using SVD
            _, s, _ = torch.svd(X_double)
            condition_number = s[0] / s[-1]

            if condition_number > 1e8:
                warnings.warn(
                    f"Design matrix is poorly conditioned (condition number: {condition_number:.2e}). "
                    "Consider regularization or removing collinear features.",
                    RuntimeWarning
                )
        except Exception:
            # Silently continue if SVD fails
            pass

    @abstractmethod
    def _compute_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the loss function (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def _fit_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> None:
        """Fit the model (to be implemented by subclasses)."""
        pass

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.Series, torch.Tensor],
        sample_weight: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> 'BaseRegression':
        """
        Fit the regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : BaseRegression
            Fitted model.
        """
        # Validate and preprocess data
        X_tensor, y_tensor, feature_names = self._validate_input(X, y)

        # Store feature information
        self.feature_names_ = feature_names
        self.n_features_in_ = X_tensor.shape[1]

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = to_tensor(sample_weight, device=self.device, dtype=torch.float32)
            if sample_weight.shape[0] != X_tensor.shape[0]:
                raise ValueError("sample_weight must have the same length as X")

        # Check condition number
        X_with_intercept = self._add_intercept(X_tensor)
        self._check_condition_number(X_with_intercept)

        # Fit the model
        self._fit_model(X_tensor, y_tensor, sample_weight)

        # Compute statistical measures
        self._compute_statistics(X_tensor, y_tensor, sample_weight)

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        X_tensor, _, _ = self._validate_input(X)

        # Apply same preprocessing as training
        if isinstance(X, pd.DataFrame) and self.encoding_mappings_:
            categorical_cols = list(self.encoding_mappings_.keys())
            if categorical_cols:
                X_encoded, _, _ = create_dummy_variables(X, categorical_cols, self.encoding_mappings_)
                X_tensor = to_tensor(X_encoded, device=self.device, dtype=torch.float32)

        X_with_intercept = self._add_intercept(X_tensor)

        with torch.no_grad():
            predictions = self._predict_tensor(X_with_intercept)

        return predictions.cpu().numpy()

    @abstractmethod
    def _predict_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using torch tensors (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def _compute_statistics(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> None:
        """Compute statistical measures (to be implemented by subclasses)."""
        pass

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compute confidence intervals for model parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (confidence level = 1 - alpha).

        Returns
        -------
        conf_int : pd.DataFrame
            Confidence intervals with columns [lower, upper].
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing confidence intervals")

        if self.standard_errors_ is None:
            raise ValueError("Standard errors not available")

        lower, upper = compute_confidence_intervals(
            self.coef_, self.standard_errors_, alpha, self._get_dof()
        )

        feature_names = self._get_feature_names()
        return pd.DataFrame({
            'lower': lower,
            'upper': upper
        }, index=feature_names)

    def summary(self) -> str:
        """
        Generate a statistical summary of the model results.

        Returns
        -------
        summary : str
            Formatted summary table.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before generating summary")

        return format_summary_table(
            coef=self.coef_,
            std_err=self.standard_errors_,
            feature_names=self._get_feature_names(),
            model_stats={
                'log_likelihood': self.log_likelihood_,
                'aic': self.aic_,
                'bic': self.bic_,
                'n_obs': self._get_n_obs(),
                'n_params': len(self.coef_)
            },
            dof=self._get_dof()
        )

    def _get_feature_names(self) -> list:
        """Get feature names including intercept."""
        if self.feature_names_ is not None:
            names = self.feature_names_.copy()
        else:
            names = [f'x{i}' for i in range(self.n_features_in_)]

        if self.fit_intercept:
            names = ['const'] + names

        return names

    def _get_n_obs(self) -> int:
        """Get number of observations (to be overridden if needed)."""
        return getattr(self, 'n_obs_', 0)

    def _get_dof(self) -> int:
        """Get degrees of freedom (to be overridden by subclasses)."""
        return max(1, self._get_n_obs() - len(self.coef_))

    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y: Union[np.ndarray, pd.Series, torch.Tensor]
    ) -> float:
        """
        Compute the coefficient of determination R^2 (for linear) or accuracy (for logistic).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            Model score.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing score")

        y_pred = self.predict(X)
        return self._compute_score(y, y_pred)

    @abstractmethod
    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute model score (to be implemented by subclasses)."""
        pass

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'fit_intercept': self.fit_intercept,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'device': str(self.device),
            'categorical_encoding': self.categorical_encoding
        }

    def set_params(self, **params) -> 'BaseRegression':
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self