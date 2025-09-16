"""
Linear regression implementation with statistical inference.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseRegression
from .stats import compute_model_statistics, compute_standard_errors


class LinearRegression(BaseRegression):
    """
    Linear regression with comprehensive statistical inference.

    This implementation provides PyTorch-based linear regression with statistical
    measures similar to statsmodels, including standard errors, t-statistics,
    p-values, confidence intervals, and model diagnostics.

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
    solver : str, default='auto'
        Solver to use ('auto', 'normal_equation', 'qr', 'gradient_descent').
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        penalty: str = "none",
        alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: str = "auto",
        solver: str = "auto",
        **kwargs,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            device=device,
            **kwargs,
        )

        self.solver = solver
        self.residual_std_: torch.Tensor | None = None
        self.r_squared_: float | None = None
        self.adj_r_squared_: float | None = None

    def _compute_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the mean squared error loss with optional regularization.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix (with intercept if fit_intercept=True).
        y : torch.Tensor
            Target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value.
        """
        # Compute predictions
        if self.coef_ is None:
            raise ValueError("Model coefficients not available")
        predictions = torch.matmul(X, self.coef_)

        # Compute MSE loss
        if sample_weight is None:
            mse_loss = F.mse_loss(predictions, y)
        else:
            squared_errors = (predictions - y) ** 2
            weighted_loss = torch.sum(sample_weight * squared_errors) / torch.sum(
                sample_weight
            )
            mse_loss = weighted_loss

        # Add regularization penalty (exclude intercept)
        if self.penalty != "none":
            if self.coef_ is None:
                raise ValueError("Model coefficients not available")
            coef_to_regularize = self.coef_[1:] if self.fit_intercept else self.coef_
            regularization = self._get_regularization_penalty(coef_to_regularize)
            return mse_loss + regularization

        return mse_loss

    def _fit_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Fit the linear regression model.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix.
        y : torch.Tensor
            Target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.
        """
        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)

        # Store number of observations
        self.n_obs_ = X.shape[0]

        # Choose solver
        solver = self._choose_solver(X_with_intercept, sample_weight)

        if solver in ["normal_equation", "qr"] and self.penalty == "none":
            self._fit_analytical(X_with_intercept, y, sample_weight, solver)
        else:
            self._fit_iterative(X_with_intercept, y, sample_weight)

    def _choose_solver(
        self, X: torch.Tensor, sample_weight: torch.Tensor | None
    ) -> str:
        """Choose the appropriate solver based on problem characteristics."""
        if self.solver != "auto":
            return self.solver

        n_samples, n_features = X.shape

        # Use analytical solution for unregularized problems
        if self.penalty == "none" and sample_weight is None:
            if n_features < 10000:  # QR is more stable
                return "qr"
            else:
                return "normal_equation"
        else:
            return "gradient_descent"

    def _fit_analytical(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None,
        solver: str,
    ) -> None:
        """
        Fit using analytical solution (normal equations or QR decomposition).

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix with intercept.
        y : torch.Tensor
            Target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.
        solver : str
            Solver type ('normal_equation' or 'qr').
        """
        try:
            if solver == "qr":
                # QR decomposition (more numerically stable)
                if sample_weight is not None:
                    # Apply weights
                    sqrt_weights = torch.sqrt(sample_weight).unsqueeze(1)
                    X_weighted = X * sqrt_weights
                    y_weighted = y * torch.sqrt(sample_weight)
                else:
                    X_weighted = X
                    y_weighted = y

                # Use double precision for numerical stability
                X_double = X_weighted.double()
                y_double = y_weighted.double()

                Q, R = torch.qr(X_double)
                self.coef_ = (
                    torch.triangular_solve(
                        torch.matmul(Q.t(), y_double).unsqueeze(1), R, upper=True
                    )
                    .solution.squeeze()
                    .float()
                )

            else:  # normal_equation
                if sample_weight is not None:
                    # Apply weights
                    W = torch.diag(sample_weight)
                    XtWX = torch.matmul(torch.matmul(X.t(), W), X)
                    XtWy = torch.matmul(torch.matmul(X.t(), W), y)
                else:
                    XtWX = torch.matmul(X.t(), X)
                    XtWy = torch.matmul(X.t(), y)

                # Use double precision for numerical stability
                XtWX_double = XtWX.double()
                XtWy_double = XtWy.double()

                self.coef_ = torch.linalg.solve(XtWX_double, XtWy_double).float()

        except RuntimeError as e:
            warnings.warn(
                f"Analytical solver failed ({str(e)}). Falling back to gradient descent.",
                RuntimeWarning,
            )
            self._fit_iterative(X, y, sample_weight)

    def _fit_iterative(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Fit using iterative optimization (gradient descent).

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix with intercept.
        y : torch.Tensor
            Target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.
        """
        # Initialize coefficients
        n_features = X.shape[1]
        self.coef_ = (
            torch.randn(n_features, device=self.device, dtype=torch.float32) * 0.01
        )

        # Set up optimizer
        self.coef_.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [self.coef_], max_iter=self.max_iter, tolerance_grad=self.tol
        )

        def closure():
            optimizer.zero_grad()
            loss = self._compute_loss(X, y, sample_weight)
            loss.backward()
            return loss

        # Optimize
        prev_loss = float("inf")
        for _ in range(self.max_iter):
            optimizer.step(closure)

            with torch.no_grad():
                current_loss = self._compute_loss(X, y, sample_weight).item()

                if abs(prev_loss - current_loss) < self.tol:
                    break

                prev_loss = current_loss

        # Detach coefficients
        if self.coef_ is not None:
            self.coef_ = self.coef_.detach()

    def _predict_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using torch tensors."""
        if self.coef_ is None:
            raise ValueError("Model coefficients not available")
        return torch.matmul(X, self.coef_)

    def _compute_statistics(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Compute statistical measures for the fitted model.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix.
        y : torch.Tensor
            Target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.
        """
        X_with_intercept = self._add_intercept(X)

        # Compute predictions and residuals
        with torch.no_grad():
            y_pred = self._predict_tensor(X_with_intercept)
            residuals = y - y_pred

        # Compute residual standard error
        if sample_weight is None:
            mse = torch.mean(residuals**2)
        else:
            weighted_sq_residuals = sample_weight * (residuals**2)
            mse = torch.sum(weighted_sq_residuals) / torch.sum(sample_weight)

        self.residual_std_ = torch.sqrt(mse)

        # Compute covariance matrix and standard errors
        self._compute_covariance_matrix(X_with_intercept, mse, sample_weight)

        if self.covariance_matrix_ is not None:
            self.standard_errors_ = compute_standard_errors(self.covariance_matrix_)

        # Compute model statistics

        # R-squared
        ss_res = torch.sum(residuals**2).item()
        ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
        self.r_squared_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Adjusted R-squared
        n_obs = len(y)
        if self.coef_ is None:
            raise ValueError("Model coefficients not available")
        n_params = len(self.coef_)
        self.adj_r_squared_ = 1 - (1 - self.r_squared_) * (n_obs - 1) / (
            n_obs - n_params
        )

        # Log-likelihood (assuming normal errors)
        if self.residual_std_ is None:
            raise ValueError("Residual standard error not available")
        log_likelihood = -0.5 * n_obs * (
            np.log(2 * np.pi) + 2 * np.log(self.residual_std_.item())
        ) - 0.5 * ss_res / (self.residual_std_.item() ** 2)
        self.log_likelihood_ = log_likelihood

        # Information criteria
        model_stats = compute_model_statistics(
            y, y_pred, log_likelihood, n_params, "linear"
        )
        self.aic_ = model_stats["aic"]
        self.bic_ = model_stats["bic"]

    def _compute_covariance_matrix(
        self,
        X: torch.Tensor,
        mse: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Compute parameter covariance matrix.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix with intercept.
        mse : torch.Tensor
            Mean squared error.
        sample_weight : torch.Tensor, optional
            Sample weights.
        """
        try:
            if sample_weight is None:
                XtX = torch.matmul(X.t(), X)
            else:
                W = torch.diag(sample_weight)
                XtX = torch.matmul(torch.matmul(X.t(), W), X)

            # Use double precision for numerical stability
            XtX_double = XtX.double()

            # Add small regularization for numerical stability
            if self.penalty == "none":
                reg_term = 1e-8 * torch.eye(
                    XtX_double.shape[0],
                    device=XtX_double.device,
                    dtype=XtX_double.dtype,
                )
                XtX_double += reg_term

            # Compute covariance matrix: (X'X)^(-1) * σ²
            XtX_inv = torch.linalg.inv(XtX_double)
            self.covariance_matrix_ = (XtX_inv * mse.double()).float()

        except RuntimeError:
            warnings.warn(
                "Could not compute covariance matrix. Standard errors will not be available.",
                RuntimeWarning,
            )
            self.covariance_matrix_ = None

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _get_dof(self) -> int:
        """Get degrees of freedom for t-distribution."""
        if self.coef_ is None:
            return 1
        return max(1, self.n_obs_ - len(self.coef_))

    def predict_interval(
        self, X: np.ndarray | torch.Tensor, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute prediction intervals for new observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        alpha : float, default=0.05
            Significance level (confidence level = 1 - alpha).

        Returns
        -------
        predictions : ndarray
            Point predictions.
        lower : ndarray
            Lower bounds of prediction intervals.
        upper : ndarray
            Upper bounds of prediction intervals.
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before computing prediction intervals"
            )

        if self.covariance_matrix_ is None:
            raise ValueError("Covariance matrix not available")

        # Get predictions
        predictions = self.predict(X)

        # Convert to tensor for computation
        X_tensor, _, _ = self._validate_input(X)
        X_with_intercept = self._add_intercept(X_tensor)

        # Compute prediction variance
        # Var(y_new) = σ² * (1 + x_new^T * (X^T*X)^(-1) * x_new)
        with torch.no_grad():
            # Get covariance matrix in double precision for stability
            if self.covariance_matrix_ is None:
                raise ValueError("Covariance matrix not available")
            if self.residual_std_ is None:
                raise ValueError("Residual standard error not available")
            cov_matrix = self.covariance_matrix_.double()
            residual_var = self.residual_std_.item() ** 2

            pred_variances = []
            for i in range(X_with_intercept.shape[0]):
                x_new = X_with_intercept[i : i + 1].double()  # Shape: (1, n_features)
                leverage = (
                    torch.matmul(torch.matmul(x_new, cov_matrix), x_new.t())
                    / residual_var
                )
                pred_var = residual_var * (1 + leverage.item())
                pred_variances.append(pred_var)

            pred_std = np.sqrt(pred_variances)

        # Compute critical value (t-distribution)
        from scipy import stats

        dof = self._get_dof()
        t_critical = stats.t.ppf(1 - alpha / 2, df=dof)

        # Compute intervals
        margin_of_error = t_critical * pred_std
        lower = predictions - margin_of_error
        upper = predictions + margin_of_error

        return predictions, lower, upper

    def get_residuals(self, X, y, residual_type="raw"):
        """
        Compute model residuals.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        residual_type : str, default='raw'
            Type of residuals ('raw', 'standardized').

        Returns
        -------
        residuals : np.ndarray
            Computed residuals.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing residuals")

        y_pred = self.predict(X)
        residuals = y - y_pred

        if residual_type == "standardized":
            if self.residual_std_ is None:
                raise ValueError("Residual standard error not available")
            residuals = residuals / self.residual_std_.item()

        return residuals
