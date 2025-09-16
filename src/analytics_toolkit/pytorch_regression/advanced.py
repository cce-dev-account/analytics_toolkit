"""
Advanced regression features including regularization paths, polynomial regression, and robust methods.
"""

from typing import Any, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from .base import BaseRegression
from .linear import LinearRegression


class RegularizationPath:
    """
    Compute regularization paths for Ridge and Lasso regression.

    This class provides functionality to compute the full regularization path,
    helping to select the optimal regularization parameter via cross-validation.
    """

    def __init__(
        self,
        penalty: str = "l2",
        alphas: Optional[np.ndarray] = None,
        n_alphas: int = 100,
        alpha_min_ratio: float = 1e-4,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        penalty : str, default='l2'
            Type of penalty ('l1', 'l2', 'elastic_net').
        alphas : array-like, optional
            List of alphas to try. If None, will be auto-generated.
        n_alphas : int, default=100
            Number of alphas to try if alphas is None.
        alpha_min_ratio : float, default=1e-4
            Minimum alpha as a fraction of maximum alpha.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, default='neg_mean_squared_error'
            Scoring metric for cross-validation.
        max_iter : int, default=1000
            Maximum iterations for each alpha.
        tol : float, default=1e-4
            Tolerance for convergence.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.penalty = penalty
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.cv = cv
        self.scoring = scoring
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Results storage
        self.alphas_ = None
        self.coef_path_ = None
        self.cv_scores_ = None
        self.cv_scores_std_ = None
        self.alpha_optimal_ = None
        self.best_model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizationPath":
        """
        Compute the regularization path using cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RegularizationPath
            Fitted regularization path object.
        """
        X = self._validate_input(X)
        y = self._validate_input(y)

        # Generate alpha values if not provided
        if self.alphas is None:
            self.alphas_ = self._generate_alphas(X, y)
        else:
            self.alphas_ = np.asarray(self.alphas)

        # Initialize storage
        n_features = X.shape[1]
        self.coef_path_ = np.zeros((len(self.alphas_), n_features))
        self.cv_scores_ = np.zeros(len(self.alphas_))
        self.cv_scores_std_ = np.zeros(len(self.alphas_))

        # Compute path
        for i, alpha in enumerate(self.alphas_):
            coef, cv_score, cv_std = self._fit_alpha(X, y, alpha)
            self.coef_path_[i] = coef
            self.cv_scores_[i] = cv_score
            self.cv_scores_std_[i] = cv_std

        # Find optimal alpha
        self.alpha_optimal_ = self._find_optimal_alpha()

        # Fit best model
        self.best_model_ = LinearRegression(
            penalty=self.penalty,
            alpha=self.alpha_optimal_,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.best_model_.fit(X, y)

        return self

    def _validate_input(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def _generate_alphas(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate logarithmically spaced alpha values."""
        # Estimate maximum useful alpha
        if self.penalty in ["l1", "elastic_net"]:
            # For L1, max alpha is when all coefficients are zero
            alpha_max = np.abs(X.T @ y).max() / len(y)
        else:
            # For L2, use a reasonable upper bound
            alpha_max = np.var(y) * 10

        alpha_min = alpha_max * self.alpha_min_ratio
        return np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alphas)

    def _fit_alpha(
        self, X: np.ndarray, y: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, float, float]:
        """Fit model for a single alpha using cross-validation."""
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        coefs = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit model
            model = LinearRegression(
                penalty=self.penalty, alpha=alpha, max_iter=self.max_iter, tol=self.tol
            )
            model.fit(X_train, y_train)

            # Evaluate
            if self.scoring == "neg_mean_squared_error":
                y_pred = model.predict(X_val)
                score = -np.mean((y_val - y_pred) ** 2)
            else:
                score = model.score(X_val, y_val)

            scores.append(score)

            # Extract coefficients (only the feature coefficients, not intercept)
            coef = model.coef_.detach().cpu().numpy()
            if coef.ndim > 1:
                coef = coef.flatten()

            # If fit_intercept=True, the first coefficient is the intercept
            # We want only the feature coefficients for the path
            if len(coef) == X_train.shape[1] + 1:
                # Skip the first coefficient (intercept)
                coef = coef[1:]

            coefs.append(coef)

        # Average coefficient (for path visualization)
        mean_coef = np.mean(coefs, axis=0)

        return mean_coef, np.mean(scores), np.std(scores)

    def _find_optimal_alpha(self) -> float:
        """Find the optimal alpha using 1-standard-error rule."""
        # Find alpha with best score
        best_idx = np.argmax(self.cv_scores_)
        best_score = self.cv_scores_[best_idx]
        best_std = self.cv_scores_std_[best_idx]

        # 1-SE rule: find most regularized model within 1 std of best
        threshold = best_score - best_std
        candidates = np.where(self.cv_scores_ >= threshold)[0]

        # Choose the largest alpha (most regularized)
        optimal_idx = candidates[np.argmax(self.alphas_[candidates])]
        return self.alphas_[optimal_idx]

    def plot_path(self, feature_names: list[str] | None = None) -> dict[str, Any]:
        """
        Generate data for plotting the regularization path.

        Returns
        -------
        dict
            Dictionary with plot data including alphas, coefficients, and CV scores.
        """
        if self.coef_path_ is None:
            raise ValueError("Must call fit() before plotting.")

        plot_data = {
            "alphas": self.alphas_,
            "coef_path": self.coef_path_,
            "cv_scores": self.cv_scores_,
            "cv_scores_std": self.cv_scores_std_,
            "alpha_optimal": self.alpha_optimal_,
            "feature_names": feature_names
            or [f"Feature_{i}" for i in range(self.coef_path_.shape[1])],
        }

        return plot_data


class PolynomialRegression(BaseEstimator, TransformerMixin):
    """
    Polynomial regression with automatic degree selection via cross-validation.
    """

    def __init__(
        self,
        degree: Optional[int] = None,
        max_degree: int = 5,
        include_bias: bool = False,
        interaction_only: bool = False,
        penalty: str = "none",
        alpha: float = 0.01,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        degree : int, optional
            Polynomial degree. If None, will be selected via CV.
        max_degree : int, default=5
            Maximum degree to try if degree is None.
        include_bias : bool, default=False
            Include bias column in polynomial features.
        interaction_only : bool, default=False
            Only include interaction features.
        penalty : str, default='none'
            Regularization penalty for the linear model.
        alpha : float, default=0.01
            Regularization strength.
        cv : int, default=5
            Cross-validation folds for degree selection.
        scoring : str, default='neg_mean_squared_error'
            Scoring metric.
        random_state : int, optional
            Random seed.
        """
        self.degree = degree
        self.max_degree = max_degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.penalty = penalty
        self.alpha = alpha
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

        # Fitted attributes
        self.degree_ = None
        self.poly_features_ = None
        self.model_ = None
        self.cv_scores_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        """
        Fit polynomial regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PolynomialRegression
            Fitted polynomial regression object.
        """
        X = self._validate_input(X)
        y = self._validate_input(y)

        if self.degree is not None:
            # Use specified degree
            self.degree_ = self.degree
        else:
            # Select degree via cross-validation
            self.degree_ = self._select_degree(X, y)

        # Fit final model
        self.poly_features_ = PolynomialFeatures(
            degree=self.degree_,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only,
        )

        X_poly = self.poly_features_.fit_transform(X)

        self.model_ = LinearRegression(
            penalty=self.penalty, alpha=self.alpha, fit_intercept=not self.include_bias
        )
        self.model_.fit(X_poly, y)

        return self

    def _validate_input(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def _select_degree(self, X: np.ndarray, y: np.ndarray) -> int:
        """Select optimal polynomial degree via cross-validation."""
        degrees = range(1, self.max_degree + 1)
        cv_scores = []

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for degree in degrees:
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Create polynomial features
                poly_features = PolynomialFeatures(
                    degree=degree,
                    include_bias=self.include_bias,
                    interaction_only=self.interaction_only,
                )
                X_train_poly = poly_features.fit_transform(X_train)
                X_val_poly = poly_features.transform(X_val)

                # Fit model
                model = LinearRegression(
                    penalty=self.penalty,
                    alpha=self.alpha,
                    fit_intercept=not self.include_bias,
                )
                model.fit(X_train_poly, y_train)

                # Evaluate
                if self.scoring == "neg_mean_squared_error":
                    y_pred = model.predict(X_val_poly)
                    score = -np.mean((y_val - y_pred) ** 2)
                else:
                    score = model.score(X_val_poly, y_val)

                scores.append(score)

            cv_scores.append(np.mean(scores))

        self.cv_scores_ = cv_scores
        return degrees[np.argmax(cv_scores)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted polynomial model."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_input(X)
        X_poly = self.poly_features_.transform(X)
        return self.model_.predict(X_poly)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R² score."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_input(X)
        y = self._validate_input(y)
        X_poly = self.poly_features_.transform(X)
        return self.model_.score(X_poly, y)


class RobustRegression(BaseRegression):
    """
    Robust regression methods for handling outliers.

    This class implements various robust regression techniques including
    Huber regression and RANSAC-style robust fitting.
    """

    def __init__(
        self,
        method: str = "huber",
        epsilon: float = 1.35,
        max_iter: int = 1000,
        tol: float = 1e-4,
        alpha: float = 0.01,
        fit_intercept: bool = True,
        device: str = "auto",
    ):
        """
        Parameters
        ----------
        method : str, default='huber'
            Robust regression method ('huber', 'quantile').
        epsilon : float, default=1.35
            Huber loss epsilon parameter.
        max_iter : int, default=1000
            Maximum iterations.
        tol : float, default=1e-4
            Convergence tolerance.
        alpha : float, default=0.01
            Regularization strength.
        fit_intercept : bool, default=True
            Whether to fit intercept.
        device : str, default='auto'
            Device for computation.
        """
        super().__init__(
            fit_intercept=fit_intercept, device=device, max_iter=max_iter, tol=tol
        )
        self.method = method
        self.epsilon = epsilon
        self.alpha = alpha

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "RobustRegression":
        """
        Fit robust regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self : RobustRegression
            Fitted robust regression object.
        """
        # Validate and preprocess data
        X_tensor, y_tensor, feature_names = self._validate_input(X, y)

        # Store feature information
        self.feature_names_ = feature_names
        self.n_features_in_ = X_tensor.shape[1]

        # Handle sample weights
        if sample_weight is not None:
            from .utils import to_tensor

            sample_weight = to_tensor(
                sample_weight, device=self.device, dtype=torch.float32
            )
            if sample_weight.shape[0] != X_tensor.shape[0]:
                raise ValueError("sample_weight must have the same length as X")

        # Check condition number
        X_with_intercept = self._add_intercept(X_tensor)
        self._check_condition_number(X_with_intercept)

        if self.method == "huber":
            self._fit_huber(X_tensor, y_tensor, sample_weight)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute statistics after fitting
        self._compute_statistics(X_tensor, y_tensor, sample_weight)

        self.is_fitted_ = True
        return self

    def _fit_huber(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """Fit Huber robust regression using IRLS."""
        n_samples, n_features = X.shape

        # Convert to double precision
        X = X.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)

        # Initialize parameters
        if self.fit_intercept:
            self.coef_ = torch.zeros(
                n_features + 1, device=self.device, dtype=torch.float64
            )
            X_design = torch.cat(
                [torch.ones(n_samples, 1, device=self.device, dtype=torch.float64), X],
                dim=1,
            )
        else:
            self.coef_ = torch.zeros(
                n_features, device=self.device, dtype=torch.float64
            )
            X_design = X

        # IRLS iterations
        weights = torch.ones(n_samples, device=self.device, dtype=torch.float64)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.clone()

            # Weighted normal equations
            W = torch.diag(weights)
            if sample_weight is not None:
                W = W * torch.diag(sample_weight)

            # Solve weighted least squares
            XtWX = X_design.T @ W @ X_design
            XtWy = X_design.T @ W @ y

            # Add regularization
            if self.alpha > 0:
                reg_matrix = self.alpha * torch.eye(
                    XtWX.shape[0], device=self.device, dtype=torch.float64
                )
                if self.fit_intercept:
                    reg_matrix[0, 0] = 0  # Don't regularize intercept
                XtWX += reg_matrix

            try:
                self.coef_ = torch.linalg.solve(XtWX, XtWy)
            except torch.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                self.coef_ = torch.linalg.pinv(XtWX) @ XtWy

            # Update weights based on residuals
            residuals = y - X_design @ self.coef_
            abs_residuals = torch.abs(residuals)

            # Huber weights
            weights = torch.where(
                abs_residuals <= self.epsilon,
                torch.ones_like(abs_residuals),
                self.epsilon / abs_residuals,
            )

            # Check convergence
            if torch.norm(self.coef_ - coef_old) < self.tol:
                break

        # Split coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = torch.tensor(0.0, device=self.device, dtype=torch.float64)

    def _compute_loss(
        self, X: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute Huber loss."""
        y_pred = self._predict_tensor(X)
        residuals = y - y_pred
        abs_residuals = torch.abs(residuals)

        # Huber loss
        huber_loss = torch.where(
            abs_residuals <= self.epsilon,
            0.5 * residuals**2,
            self.epsilon * abs_residuals - 0.5 * self.epsilon**2,
        )

        if sample_weight is not None:
            huber_loss = huber_loss * sample_weight

        return huber_loss.mean()

    def _fit_model(
        self, X: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor = None
    ):
        """Fit the robust regression model (already implemented in _fit_huber)."""
        self._fit_huber(X, y, sample_weight)

    def _predict_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using torch tensors."""
        # Check if X already has intercept column (from base class predict method)
        if self.fit_intercept:
            if X.shape[1] == len(self.coef_) + 1:
                # Already has intercept column
                coef_full = torch.cat([self.intercept_.unsqueeze(0), self.coef_])
                return X.to(dtype=torch.float64) @ coef_full
            else:
                # Need to add intercept column
                X_with_intercept = torch.cat(
                    [
                        torch.ones(
                            X.shape[0], 1, device=self.device, dtype=torch.float64
                        ),
                        X.to(dtype=torch.float64),
                    ],
                    dim=1,
                )
                coef_full = torch.cat([self.intercept_.unsqueeze(0), self.coef_])
                return X_with_intercept @ coef_full
        else:
            return X @ self.coef_

    def _compute_statistics(
        self, X: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor = None
    ):
        """Compute model statistics for robust regression."""
        # Basic statistics - more could be added later
        y_pred = self._predict_tensor(X)
        residuals = y - y_pred

        self.fitted_values_ = y_pred
        self.residuals_ = residuals

        # Robust standard errors could be computed here
        # For now, use basic computation
        try:
            from .stats import compute_standard_errors

            self.standard_errors_ = compute_standard_errors(
                X, residuals, self.coef_, self.fit_intercept
            )
        except Exception:
            # Fallback if stats computation fails
            self.standard_errors_ = None

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
