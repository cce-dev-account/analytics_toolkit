"""
Advanced sklearn-compatible transformers for feature engineering.
"""

import warnings

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply logarithmic transformation with automatic handling of negative values.

    Parameters
    ----------
    method : str, default='log1p'
        Method to use: 'log', 'log1p', 'log10', 'box-cox', 'yeo-johnson'
    offset : float, default='auto'
        Offset to add before transformation. 'auto' finds minimum positive offset.
    handle_zeros : bool, default=True
        Whether to handle zero values by adding small epsilon.
    """

    def __init__(self, method="log1p", offset="auto", handle_zeros=True):
        self.method = method
        self.offset = offset
        self.handle_zeros = handle_zeros

    def fit(self, X, y=None):
        """Fit the transformer."""
        X = check_array(X, ensure_all_finite="allow-nan")

        if self.offset == "auto":
            # Find minimum offset to make all values positive
            min_val = np.nanmin(X)
            if min_val <= 0:
                self.offset_ = abs(min_val) + 1e-6
            else:
                self.offset_ = 0
        else:
            self.offset_ = self.offset

        # For Box-Cox, fit lambda parameter
        if self.method == "box-cox":
            # Box-Cox requires positive values
            X_positive = X + self.offset_
            if np.any(X_positive <= 0):
                raise ValueError("Box-Cox transformation requires positive values")

            # Fit Box-Cox parameter on first column or flattened array
            if X.ndim > 1:
                sample_data = X_positive[:, 0]
            else:
                sample_data = X_positive.flatten()

            # Remove NaN values for fitting
            sample_data = sample_data[~np.isnan(sample_data)]
            if len(sample_data) > 0:
                _, self.lambda_ = stats.boxcox(sample_data)
            else:
                self.lambda_ = 0

        return self

    def transform(self, X):
        """Transform the data."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        # Apply offset
        X_transformed = X + self.offset_

        # Handle zeros if requested
        if self.handle_zeros:
            epsilon = 1e-10
            X_transformed = np.where(X_transformed == 0, epsilon, X_transformed)

        # Apply transformation
        if self.method == "log":
            return np.log(X_transformed)
        elif self.method == "log1p":
            return np.log1p(X_transformed - self.offset_)  # log1p handles the +1
        elif self.method == "log10":
            return np.log10(X_transformed)
        elif self.method == "box-cox":
            if hasattr(self, "lambda_"):
                return stats.boxcox(X_transformed, self.lambda_)
            else:
                return stats.boxcox(X_transformed)
        elif self.method == "yeo-johnson":
            return stats.yeojohnson(X_transformed)[0]
        else:
            raise ValueError(f"Unknown method: {self.method}")


class OutlierCapTransformer(BaseEstimator, TransformerMixin):
    """
    Cap outliers using various methods (IQR, percentile, z-score, modified z-score).

    Parameters
    ----------
    method : str, default='iqr'
        Method to use: 'iqr', 'percentile', 'zscore', 'modified_zscore'
    lower_quantile : float, default=0.01
        Lower quantile for capping (used with 'percentile' method)
    upper_quantile : float, default=0.99
        Upper quantile for capping (used with 'percentile' method)
    iqr_multiplier : float, default=1.5
        IQR multiplier for outlier detection
    zscore_threshold : float, default=3.0
        Z-score threshold for outlier detection
    """

    def __init__(
        self,
        method="iqr",
        lower_quantile=0.01,
        upper_quantile=0.99,
        iqr_multiplier=1.5,
        zscore_threshold=3.0,
    ):
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold

    def fit(self, X, y=None):
        """Fit the transformer."""
        X = check_array(X, ensure_all_finite="allow-nan")

        if self.method == "iqr":
            self.q1_ = np.nanpercentile(X, 25, axis=0)
            self.q3_ = np.nanpercentile(X, 75, axis=0)
            iqr = self.q3_ - self.q1_
            self.lower_bound_ = self.q1_ - self.iqr_multiplier * iqr
            self.upper_bound_ = self.q3_ + self.iqr_multiplier * iqr

        elif self.method == "percentile":
            self.lower_bound_ = np.nanpercentile(X, self.lower_quantile * 100, axis=0)
            self.upper_bound_ = np.nanpercentile(X, self.upper_quantile * 100, axis=0)

        elif self.method in ["zscore", "modified_zscore"]:
            self.mean_ = np.nanmean(X, axis=0)
            if self.method == "zscore":
                self.std_ = np.nanstd(X, axis=0)
                self.lower_bound_ = self.mean_ - self.zscore_threshold * self.std_
                self.upper_bound_ = self.mean_ + self.zscore_threshold * self.std_
            else:  # modified z-score
                self.median_ = np.nanmedian(X, axis=0)
                self.mad_ = np.nanmedian(np.abs(X - self.median_), axis=0)
                threshold = (
                    self.zscore_threshold * 0.6745 * self.mad_
                )  # 0.6745 is scaling factor
                self.lower_bound_ = self.median_ - threshold
                self.upper_bound_ = self.median_ + threshold

        return self

    def transform(self, X):
        """Transform the data by capping outliers."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        return np.clip(X, self.lower_bound_, self.upper_bound_)


class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    Bin continuous features into discrete bins using various strategies.

    Parameters
    ----------
    strategy : str, default='quantile'
        Binning strategy: 'uniform', 'quantile', 'kmeans'
    n_bins : int, default=5
        Number of bins to create
    encode : str, default='ordinal'
        Encoding method: 'ordinal', 'onehot', 'binary'
    """

    def __init__(self, strategy="quantile", n_bins=5, encode="ordinal"):
        self.strategy = strategy
        self.n_bins = n_bins
        self.encode = encode

    def fit(self, X, y=None):
        """Fit the binning transformer."""
        X = check_array(X, ensure_all_finite="allow-nan")

        self.bin_edges_ = []

        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            col_data = col_data[~np.isnan(col_data)]  # Remove NaN values

            if self.strategy == "uniform":
                edges = np.linspace(col_data.min(), col_data.max(), self.n_bins + 1)
            elif self.strategy == "quantile":
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col_data, quantiles)
                # Ensure unique edges
                edges = np.unique(edges)
                if len(edges) < self.n_bins + 1:
                    warnings.warn(
                        f"Column {col_idx}: Not enough unique values for {self.n_bins} bins"
                    )
            elif self.strategy == "kmeans":
                from sklearn.cluster import KMeans

                if len(col_data) >= self.n_bins:
                    kmeans = KMeans(n_clusters=self.n_bins, random_state=42, n_init=10)
                    kmeans.fit(col_data.reshape(-1, 1))
                    centers = sorted(kmeans.cluster_centers_.flatten())
                    edges = [col_data.min()]
                    for i in range(len(centers) - 1):
                        edges.append((centers[i] + centers[i + 1]) / 2)
                    edges.append(col_data.max())
                    edges = np.array(edges)
                else:
                    edges = np.linspace(
                        col_data.min(),
                        col_data.max(),
                        min(self.n_bins, len(col_data)) + 1,
                    )
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.bin_edges_.append(edges)

        return self

    def transform(self, X):
        """Transform data by binning."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        if self.encode == "ordinal":
            result = np.zeros_like(X)
            for col_idx in range(X.shape[1]):
                result[:, col_idx] = (
                    np.digitize(X[:, col_idx], self.bin_edges_[col_idx]) - 1
                )
                # Handle out-of-bounds values
                result[:, col_idx] = np.clip(
                    result[:, col_idx], 0, len(self.bin_edges_[col_idx]) - 2
                )
            return result

        elif self.encode == "onehot":
            from sklearn.preprocessing import OneHotEncoder

            # First bin the data
            binned = self.transform(X)  # This will use ordinal encoding
            # Then apply one-hot encoding
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            return encoder.fit_transform(binned)

        else:
            raise ValueError(f"Unknown encoding: {self.encode}")


class PolynomialFeaturesAdvanced(BaseEstimator, TransformerMixin):
    """
    Advanced polynomial features with automatic interaction detection.

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of polynomial features
    interaction_only : bool, default=False
        Only include interaction terms (no powers)
    include_bias : bool, default=True
        Include bias column
    feature_selection : str, default=None
        Automatic feature selection: None, 'variance', 'correlation'
    max_features : int, default=None
        Maximum number of features to keep after selection
    """

    def __init__(
        self,
        degree=2,
        interaction_only=False,
        include_bias=True,
        feature_selection=None,
        max_features=None,
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.feature_selection = feature_selection
        self.max_features = max_features

    def fit(self, X, y=None):
        """Fit the polynomial transformer."""
        from sklearn.preprocessing import PolynomialFeatures

        X = check_array(X, ensure_all_finite="allow-nan")

        # Create polynomial features
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        X_poly = self.poly_.fit_transform(X)

        # Apply feature selection if requested
        if self.feature_selection:
            self.selected_features_ = self._select_features(X_poly, y)
        else:
            self.selected_features_ = np.arange(X_poly.shape[1])

        return self

    def _select_features(self, X, y):
        """Select features based on specified criteria."""
        n_features = X.shape[1]

        if self.feature_selection == "variance":
            # Remove low-variance features
            variances = np.var(X, axis=0)
            threshold = np.percentile(variances, 25)  # Remove bottom 25%
            selected = variances > threshold

        elif self.feature_selection == "correlation":
            # Remove highly correlated features
            corr_matrix = np.corrcoef(X.T)
            selected = np.ones(n_features, dtype=bool)

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if abs(corr_matrix[i, j]) > 0.95:  # High correlation threshold
                        selected[j] = False

        else:
            selected = np.ones(n_features, dtype=bool)

        # Limit number of features if specified
        if self.max_features and np.sum(selected) > self.max_features:
            selected_indices = np.where(selected)[0]
            # Keep features with highest variance
            if hasattr(self, "variances"):
                variances = self.variances[selected_indices]
            else:
                variances = np.var(X[:, selected_indices], axis=0)
            top_features = selected_indices[np.argsort(variances)[-self.max_features :]]
            selected = np.zeros(n_features, dtype=bool)
            selected[top_features] = True

        return np.where(selected)[0]

    def transform(self, X):
        """Transform data using fitted polynomial features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        X_poly = self.poly_.transform(X)
        return X_poly[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.poly_.n_features_in_)]

        all_names = self.poly_.get_feature_names_out(input_features)
        return all_names[self.selected_features_]


class RobustScaler(BaseEstimator, TransformerMixin):
    """
    Robust scaling using median and IQR instead of mean and standard deviation.

    Parameters
    ----------
    quantile_range : tuple, default=(25.0, 75.0)
        Quantile range used to calculate scale
    with_centering : bool, default=True
        Whether to center data at median
    with_scaling : bool, default=True
        Whether to scale data to unit IQR
    """

    def __init__(
        self, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True
    ):
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def fit(self, X, y=None):
        """Fit the robust scaler."""
        X = check_array(X, ensure_all_finite="allow-nan")

        if self.with_centering:
            self.center_ = np.nanmedian(X, axis=0)
        else:
            self.center_ = 0

        if self.with_scaling:
            q_min, q_max = self.quantile_range
            self.scale_ = np.nanpercentile(X, q_max, axis=0) - np.nanpercentile(
                X, q_min, axis=0
            )
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        else:
            self.scale_ = 1

        return self

    def transform(self, X):
        """Scale features using robust statistics."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        return (X - self.center_) / self.scale_
