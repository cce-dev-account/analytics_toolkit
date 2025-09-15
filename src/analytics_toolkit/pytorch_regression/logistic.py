"""
Logistic regression implementation with statistical inference.
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Optional, Tuple
from .base import BaseRegression
from .stats import compute_standard_errors, compute_model_statistics


class LogisticRegression(BaseRegression):
    """
    Logistic regression with comprehensive statistical inference.

    This implementation provides PyTorch-based logistic regression with statistical
    measures including standard errors, z-statistics, p-values, confidence intervals,
    and model diagnostics.

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
    solver : str, default='lbfgs'
        Solver to use ('lbfgs', 'adam', 'sgd').
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        penalty: str = 'none',
        alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: str = 'auto',
        solver: str = 'lbfgs',
        **kwargs
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            device=device,
            **kwargs
        )

        self.solver = solver
        self.classes_ = None
        self.n_iter_ = None

        # Validate solver
        valid_solvers = ['lbfgs', 'adam', 'sgd']
        if self.solver not in valid_solvers:
            raise ValueError(f"solver must be one of {valid_solvers}")

    def _validate_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Validate and preprocess target values for binary classification.

        Parameters
        ----------
        y : torch.Tensor
            Target values.

        Returns
        -------
        y_binary : torch.Tensor
            Binary target values (0 or 1).
        """
        # Get unique classes
        unique_classes = torch.unique(y)

        if len(unique_classes) == 1:
            raise ValueError("Only one class present in target variable")
        elif len(unique_classes) == 2:
            # Binary classification
            self.classes_ = unique_classes.detach().cpu().numpy()

            # Convert to 0/1 encoding
            y_binary = torch.where(y == unique_classes[1], 1.0, 0.0)
            return y_binary
        else:
            raise ValueError("Multiclass classification not supported yet")

    def _compute_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the logistic loss with optional regularization.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix (with intercept if fit_intercept=True).
        y : torch.Tensor
            Binary target vector (0/1).
        sample_weight : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        loss : torch.Tensor
            Computed loss value.
        """
        # Compute logits
        logits = torch.matmul(X, self.coef_)

        # Compute binary cross-entropy loss
        if sample_weight is None:
            bce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='mean')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, y, weight=sample_weight, reduction='mean')

        # Add regularization penalty (exclude intercept)
        if self.penalty != 'none':
            coef_to_regularize = self.coef_[1:] if self.fit_intercept else self.coef_
            regularization = self._get_regularization_penalty(coef_to_regularize)
            return bce_loss + regularization

        return bce_loss

    def _compute_log_likelihood(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute log-likelihood for logistic regression.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix with intercept.
        y : torch.Tensor
            Binary target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.

        Returns
        -------
        log_likelihood : float
            Log-likelihood value.
        """
        with torch.no_grad():
            logits = torch.matmul(X, self.coef_)

            # Use log-sum-exp trick for numerical stability
            log_probs = -F.binary_cross_entropy_with_logits(
                logits, y, reduction='none'
            )

            if sample_weight is None:
                log_likelihood = torch.sum(log_probs).item()
            else:
                log_likelihood = torch.sum(sample_weight * log_probs).item()

        return log_likelihood

    def _fit_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> None:
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix.
        y : torch.Tensor
            Target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.
        """
        # Validate and preprocess targets
        y_binary = self._validate_targets(y)

        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]

        # Store number of observations
        self.n_obs_ = X.shape[0]

        # Initialize coefficients
        self.coef_ = torch.zeros(n_features, device=self.device, dtype=torch.float32)
        self.coef_.requires_grad_(True)

        # Choose and set up optimizer
        if self.solver == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                [self.coef_],
                max_iter=self.max_iter,
                tolerance_grad=self.tol,
                tolerance_change=self.tol * 1e-2,
                line_search_fn='strong_wolfe'
            )
        elif self.solver == 'adam':
            optimizer = torch.optim.Adam([self.coef_], lr=0.01)
        else:  # sgd
            optimizer = torch.optim.SGD([self.coef_], lr=0.01)

        # Define closure for LBFGS
        def closure():
            optimizer.zero_grad()
            loss = self._compute_loss(X_with_intercept, y_binary, sample_weight)
            loss.backward()
            return loss

        # Optimization loop
        prev_loss = float('inf')
        self.n_iter_ = 0

        for iteration in range(self.max_iter):
            if self.solver == 'lbfgs':
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = self._compute_loss(X_with_intercept, y_binary, sample_weight)
                loss.backward()
                optimizer.step()

            # Check convergence
            with torch.no_grad():
                current_loss = self._compute_loss(X_with_intercept, y_binary, sample_weight).item()

                if abs(prev_loss - current_loss) < self.tol:
                    break

                prev_loss = current_loss
                self.n_iter_ = iteration + 1

        # Detach coefficients
        self.coef_ = self.coef_.detach()

        # Check for convergence issues
        if self.n_iter_ == self.max_iter:
            warnings.warn(
                f"Maximum number of iterations ({self.max_iter}) reached. "
                "The optimization may not have converged.",
                RuntimeWarning
            )

        # Check for perfect separation
        self._check_perfect_separation(X_with_intercept, y_binary)

    def _check_perfect_separation(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Check for perfect separation in logistic regression."""
        with torch.no_grad():
            predictions = torch.sigmoid(torch.matmul(X, self.coef_))
            predicted_classes = (predictions > 0.5).float()

            if torch.all(predicted_classes == y):
                warnings.warn(
                    "Perfect separation detected. Standard errors may be unreliable.",
                    RuntimeWarning
                )

    def _predict_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Make probability predictions using torch tensors."""
        logits = torch.matmul(X, self.coef_)
        return torch.sigmoid(logits)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        X_tensor, _, _ = self._validate_input(X)
        X_with_intercept = self._add_intercept(X_tensor)

        with torch.no_grad():
            prob_positive = self._predict_tensor(X_with_intercept)
            prob_negative = 1 - prob_positive

        # Return probabilities for both classes
        probabilities = torch.stack([prob_negative, prob_positive], dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make binary class predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Get probabilities
        probabilities = self.predict_proba(X)

        # Convert to class predictions
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def _compute_statistics(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
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
        y_binary = self._validate_targets(y)

        # Compute log-likelihood
        self.log_likelihood_ = self._compute_log_likelihood(X_with_intercept, y_binary, sample_weight)

        # Compute covariance matrix and standard errors using observed information matrix
        self._compute_covariance_matrix(X_with_intercept, y_binary, sample_weight)

        if self.covariance_matrix_ is not None:
            self.standard_errors_ = compute_standard_errors(self.covariance_matrix_)

        # Compute model statistics
        n_params = len(self.coef_)
        model_stats = compute_model_statistics(
            y_binary, self._predict_tensor(X_with_intercept),
            self.log_likelihood_, n_params, 'logistic'
        )
        self.aic_ = model_stats['aic']
        self.bic_ = model_stats['bic']

    def _compute_covariance_matrix(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> None:
        """
        Compute parameter covariance matrix using observed information matrix.

        For logistic regression, the covariance matrix is the inverse of the
        observed information matrix (negative Hessian of log-likelihood).

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix with intercept.
        y : torch.Tensor
            Binary target vector.
        sample_weight : torch.Tensor, optional
            Sample weights.
        """
        try:
            with torch.no_grad():
                # Compute predicted probabilities
                logits = torch.matmul(X, self.coef_)
                probs = torch.sigmoid(logits)

                # Compute weights for information matrix: p(1-p)
                weights = probs * (1 - probs)

                if sample_weight is not None:
                    weights = weights * sample_weight

                # Compute information matrix: X^T W X
                # where W is diagonal matrix of weights
                weighted_X = X * weights.unsqueeze(1)
                information_matrix = torch.matmul(X.t(), weighted_X)

                # Use double precision for numerical stability
                info_matrix_double = information_matrix.double()

                # Add small regularization for numerical stability
                reg_term = 1e-8 * torch.eye(
                    info_matrix_double.shape[0],
                    device=info_matrix_double.device,
                    dtype=info_matrix_double.dtype
                )
                info_matrix_double += reg_term

                # Covariance matrix is inverse of information matrix
                self.covariance_matrix_ = torch.linalg.inv(info_matrix_double).float()

        except RuntimeError:
            warnings.warn(
                "Could not compute covariance matrix. Standard errors will not be available.",
                RuntimeWarning
            )
            self.covariance_matrix_ = None

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        return np.mean(y_true == y_pred)

    def _get_dof(self) -> int:
        """Get degrees of freedom (use normal distribution for large samples)."""
        # For logistic regression, we typically use z-distribution (normal)
        # But return dof for compatibility
        return max(1, self.n_obs_ - len(self.coef_))

    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute the decision function (log-odds).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        decision : ndarray of shape (n_samples,)
            Decision function values (log-odds).
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing decision function")

        X_tensor, _, _ = self._validate_input(X)
        X_with_intercept = self._add_intercept(X_tensor)

        with torch.no_grad():
            logits = torch.matmul(X_with_intercept, self.coef_)

        return logits.cpu().numpy()

    def score(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        y_pred = self.predict(X)
        return self._compute_score(y, y_pred)