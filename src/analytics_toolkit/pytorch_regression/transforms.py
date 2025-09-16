"""
Non-linear transformations for regression including splines and basis functions.
"""

import warnings
from typing import Optional

import numpy as np
from scipy import interpolate
from sklearn.base import BaseEstimator, TransformerMixin


class BSplineTransformer(BaseEstimator, TransformerMixin):
    """
    B-spline basis functions for non-linear feature transformations.

    This transformer creates B-spline basis functions for each input feature,
    allowing for flexible non-linear modeling while maintaining smoothness.
    """

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        include_bias: bool = True,
        knot_strategy: str = "quantile",
        clip_bounds: bool = True,
        extrapolation: str = "constant",
    ):
        """
        Parameters
        ----------
        n_knots : int, default=10
            Number of knots for the splines.
        degree : int, default=3
            Degree of the splines (3 = cubic).
        include_bias : bool, default=True
            Include bias term in transformation.
        knot_strategy : str, default="quantile"
            Strategy for knot placement ("quantile", "uniform").
        clip_bounds : bool, default=True
            Clip values to training bounds during transform.
        extrapolation : str, default="constant"
            Extrapolation method ("constant", "linear", "extrapolate").
        """
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias
        self.knot_strategy = knot_strategy
        self.clip_bounds = clip_bounds
        self.extrapolation = extrapolation

        # Fitted attributes
        self.knots_ = None
        self.feature_bounds_ = None
        self.n_features_in_ = None
        self.n_output_features_ = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "BSplineTransformer":
        """
        Fit the B-spline transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : BSplineTransformer
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.knots_ = []
        self.feature_bounds_ = []

        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]

            # Store bounds for clipping
            bounds = (np.min(feature_data), np.max(feature_data))
            self.feature_bounds_.append(bounds)

            # Generate knots
            if self.knot_strategy == "quantile":
                # Place knots at quantiles
                quantiles = np.linspace(0, 1, self.n_knots + 2)[1:-1]  # Exclude 0 and 1
                knots = np.quantile(feature_data, quantiles)
            elif self.knot_strategy == "uniform":
                # Place knots uniformly
                knots = np.linspace(bounds[0], bounds[1], self.n_knots + 2)[1:-1]
            else:
                raise ValueError(f"Unknown knot_strategy: {self.knot_strategy}")

            # Add boundary knots for B-splines
            boundary_knots = np.array([bounds[0], bounds[1]])
            all_knots = np.concatenate(
                [
                    np.repeat(boundary_knots[0], self.degree),
                    knots,
                    np.repeat(boundary_knots[1], self.degree),
                ]
            )

            self.knots_.append(all_knots)

        # Calculate output dimensions
        n_basis_per_feature = len(self.knots_[0]) - self.degree - 1
        self.n_output_features_ = n_basis_per_feature * n_features
        if self.include_bias:
            self.n_output_features_ += 1

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using B-spline basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_output_features_)
            Transformed data.
        """
        if self.knots_ is None:
            raise ValueError("This BSplineTransformer instance is not fitted yet.")

        X = self._validate_data(X, reset=False)
        n_samples = X.shape[0]

        # Initialize output array
        X_transformed = []

        for feature_idx in range(self.n_features_in_):
            feature_data = X[:, feature_idx].copy()

            # Clip to bounds if requested
            if self.clip_bounds:
                bounds = self.feature_bounds_[feature_idx]
                feature_data = np.clip(feature_data, bounds[0], bounds[1])

            # Compute B-spline basis
            knots = self.knots_[feature_idx]

            try:
                # Create B-spline basis functions
                basis_funcs = self._compute_bspline_basis(
                    feature_data, knots, self.degree
                )
                X_transformed.append(basis_funcs)
            except Exception as e:
                # Fallback to simple polynomial if B-spline fails
                warnings.warn(
                    f"B-spline computation failed for feature {feature_idx}: {e}. "
                    "Falling back to polynomial basis."
                )
                poly_basis = self._compute_polynomial_basis(feature_data, self.degree)
                X_transformed.append(poly_basis)

        # Concatenate all features
        X_out = np.hstack(X_transformed)

        # Add bias term if requested
        if self.include_bias:
            bias = np.ones((n_samples, 1))
            X_out = np.hstack([bias, X_out])

        return X_out

    def _compute_bspline_basis(
        self, x: np.ndarray, knots: np.ndarray, degree: int
    ) -> np.ndarray:
        """Compute B-spline basis functions using scipy."""
        n_basis = len(knots) - degree - 1
        basis_matrix = np.zeros((len(x), n_basis))

        for i in range(n_basis):
            # Create a B-spline with a single coefficient set to 1
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0

            try:
                spline = interpolate.BSpline(knots, coeffs, degree, extrapolate=False)
                basis_matrix[:, i] = spline(x)
            except:
                # If BSpline fails, use a simple approximation
                basis_matrix[:, i] = self._simple_basis_function(x, i, n_basis)

        # Handle NaNs by setting them to 0
        basis_matrix = np.nan_to_num(basis_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return basis_matrix

    def _simple_basis_function(
        self, x: np.ndarray, index: int, total: int
    ) -> np.ndarray:
        """Simple polynomial basis function as fallback."""
        normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        return normalized_x ** (index + 1)

    def _compute_polynomial_basis(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Compute polynomial basis functions as fallback."""
        basis_matrix = np.zeros((len(x), degree + 1))
        normalized_x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        for i in range(degree + 1):
            basis_matrix[:, i] = normalized_x**i

        return basis_matrix

    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if not reset and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but transformer was fitted with {self.n_features_in_}"
            )

        return X


class RadialBasisTransformer(BaseEstimator, TransformerMixin):
    """
    Radial Basis Function (RBF) transformer for non-linear feature mapping.

    This transformer creates RBF features centered at specified locations,
    useful for creating flexible non-linear models.
    """

    def __init__(
        self,
        n_centers: int = 10,
        kernel: str = "gaussian",
        gamma: Optional[float] = None,
        center_strategy: str = "kmeans",
        include_original: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_centers : int, default=10
            Number of RBF centers.
        kernel : str, default="gaussian"
            Type of RBF kernel ("gaussian", "multiquadric", "inverse_multiquadric").
        gamma : float, optional
            Kernel bandwidth parameter. If None, uses heuristic.
        center_strategy : str, default="kmeans"
            Strategy for selecting centers ("kmeans", "random", "quantile").
        include_original : bool, default=False
            Include original features in output.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.n_centers = n_centers
        self.kernel = kernel
        self.gamma = gamma
        self.center_strategy = center_strategy
        self.include_original = include_original
        self.random_state = random_state

        # Fitted attributes
        self.centers_ = None
        self.gamma_ = None
        self.n_features_in_ = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "RadialBasisTransformer":
        """
        Fit the RBF transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : RadialBasisTransformer
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Select centers
        if self.center_strategy == "random":
            # Random selection from training data
            indices = np.random.choice(n_samples, self.n_centers, replace=False)
            self.centers_ = X[indices].copy()
        elif self.center_strategy == "quantile":
            # Centers at quantiles for each feature
            self.centers_ = self._get_quantile_centers(X)
        elif self.center_strategy == "kmeans":
            # K-means clustering (simplified version)
            self.centers_ = self._kmeans_centers(X)
        else:
            raise ValueError(f"Unknown center_strategy: {self.center_strategy}")

        # Set gamma if not provided
        if self.gamma is None:
            # Use median distance heuristic
            distances = []
            for i in range(min(100, len(self.centers_))):  # Sample for efficiency
                for j in range(i + 1, min(100, len(self.centers_))):
                    dist = np.linalg.norm(self.centers_[i] - self.centers_[j])
                    distances.append(dist)

            if distances:
                self.gamma_ = 1.0 / (2 * np.median(distances) ** 2)
            else:
                self.gamma_ = 1.0
        else:
            self.gamma_ = self.gamma

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using RBF basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray
            Transformed data.
        """
        if self.centers_ is None:
            raise ValueError("This RadialBasisTransformer instance is not fitted yet.")

        X = self._validate_data(X, reset=False)

        # Compute RBF features
        rbf_features = self._compute_rbf_features(X)

        # Include original features if requested
        if self.include_original:
            return np.hstack([X, rbf_features])
        else:
            return rbf_features

    def _compute_rbf_features(self, X: np.ndarray) -> np.ndarray:
        """Compute RBF features for input data."""
        n_samples = X.shape[0]
        rbf_features = np.zeros((n_samples, len(self.centers_)))

        for i, center in enumerate(self.centers_):
            distances_sq = np.sum((X - center) ** 2, axis=1)

            if self.kernel == "gaussian":
                rbf_features[:, i] = np.exp(-self.gamma_ * distances_sq)
            elif self.kernel == "multiquadric":
                rbf_features[:, i] = np.sqrt(1 + self.gamma_ * distances_sq)
            elif self.kernel == "inverse_multiquadric":
                rbf_features[:, i] = 1.0 / np.sqrt(1 + self.gamma_ * distances_sq)
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")

        return rbf_features

    def _get_quantile_centers(self, X: np.ndarray) -> np.ndarray:
        """Get centers based on quantiles of each feature."""
        n_features = X.shape[1]
        n_per_dim = max(1, int(np.power(self.n_centers, 1.0 / n_features)))

        # Create grid of quantile-based centers
        centers = []
        quantiles = np.linspace(0.1, 0.9, n_per_dim)

        if n_features == 1:
            for q in quantiles:
                center = np.quantile(X, q, axis=0)
                centers.append(center)
        else:
            # For multi-dimensional case, sample from marginal quantiles
            for _ in range(self.n_centers):
                center = []
                for feat_idx in range(n_features):
                    q = np.random.choice(quantiles)
                    center.append(np.quantile(X[:, feat_idx], q))
                centers.append(center)

        return np.array(centers)

    def _kmeans_centers(self, X: np.ndarray) -> np.ndarray:
        """Simple K-means clustering to find centers."""
        # Simple K-means implementation
        n_samples, n_features = X.shape

        # Initialize centers randomly
        centers = X[np.random.choice(n_samples, self.n_centers, replace=False)].copy()

        for iteration in range(10):  # Max 10 iterations
            # Assign points to clusters
            distances = np.zeros((n_samples, self.n_centers))
            for i, center in enumerate(centers):
                distances[:, i] = np.linalg.norm(X - center, axis=1)

            labels = np.argmin(distances, axis=1)

            # Update centers
            new_centers = centers.copy()
            for i in range(self.n_centers):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centers[i] = np.mean(X[mask], axis=0)

            # Check convergence
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        return centers

    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if not reset and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but transformer was fitted with {self.n_features_in_}"
            )

        return X


class FourierTransformer(BaseEstimator, TransformerMixin):
    """
    Fourier basis function transformer for periodic patterns.

    Creates sine and cosine basis functions at different frequencies,
    useful for modeling periodic or cyclical patterns in data.
    """

    def __init__(
        self,
        n_frequencies: int = 5,
        frequency_range: Optional[tuple[float, float]] = None,
        include_bias: bool = True,
        normalize_features: bool = True,
    ):
        """
        Parameters
        ----------
        n_frequencies : int, default=5
            Number of frequency components to include.
        frequency_range : tuple, optional
            Range of frequencies (min_freq, max_freq). If None, auto-determined.
        include_bias : bool, default=True
            Include bias term in transformation.
        normalize_features : bool, default=True
            Normalize input features to [0, 2π] range.
        """
        self.n_frequencies = n_frequencies
        self.frequency_range = frequency_range
        self.include_bias = include_bias
        self.normalize_features = normalize_features

        # Fitted attributes
        self.frequencies_ = None
        self.feature_ranges_ = None
        self.n_features_in_ = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "FourierTransformer":
        """
        Fit the Fourier transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : FourierTransformer
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Store feature ranges for normalization
        if self.normalize_features:
            self.feature_ranges_ = [
                (np.min(X[:, i]), np.max(X[:, i])) for i in range(n_features)
            ]

        # Determine frequencies
        if self.frequency_range is None:
            # Default: frequencies from 1 to n_frequencies
            self.frequencies_ = np.arange(1, self.n_frequencies + 1)
        else:
            min_freq, max_freq = self.frequency_range
            self.frequencies_ = np.linspace(min_freq, max_freq, self.n_frequencies)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using Fourier basis functions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : ndarray
            Transformed data with Fourier features.
        """
        if self.frequencies_ is None:
            raise ValueError("This FourierTransformer instance is not fitted yet.")

        X = self._validate_data(X, reset=False)
        n_samples = X.shape[0]

        # Normalize features to [0, 2π] if requested
        if self.normalize_features:
            X_normalized = np.zeros_like(X)
            for i in range(self.n_features_in_):
                min_val, max_val = self.feature_ranges_[i]
                if max_val > min_val:
                    X_normalized[:, i] = (
                        2 * np.pi * (X[:, i] - min_val) / (max_val - min_val)
                    )
                else:
                    X_normalized[:, i] = X[:, i]
        else:
            X_normalized = X.copy()

        # Create Fourier features
        fourier_features = []

        for feature_idx in range(self.n_features_in_):
            feature_data = X_normalized[:, feature_idx]

            for freq in self.frequencies_:
                # Add sine and cosine components
                sine_feature = np.sin(freq * feature_data)
                cosine_feature = np.cos(freq * feature_data)

                fourier_features.append(sine_feature.reshape(-1, 1))
                fourier_features.append(cosine_feature.reshape(-1, 1))

        # Concatenate all features
        if fourier_features:
            X_transformed = np.hstack(fourier_features)
        else:
            # Handle case with no frequencies
            X_transformed = np.zeros((n_samples, 0))

        # Add bias term if requested
        if self.include_bias:
            bias = np.ones((n_samples, 1))
            if X_transformed.shape[1] > 0:
                X_transformed = np.hstack([bias, X_transformed])
            else:
                X_transformed = bias

        return X_transformed

    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if not reset and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but transformer was fitted with {self.n_features_in_}"
            )

        return X
