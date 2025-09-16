"""
Advanced categorical encoding methods including target encoding and Bayesian approaches.
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with cross-validation to prevent overfitting.

    Parameters
    ----------
    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy
    smooth : float, default=1.0
        Smoothing parameter for regularization
    min_samples_leaf : int, default=1
        Minimum samples required to encode a category
    noise_level : float, default=0.01
        Level of noise to add for regularization
    """

    def __init__(self, cv=5, smooth=1.0, min_samples_leaf=1, noise_level=0.01):
        self.cv = cv
        self.smooth = smooth
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level

    def fit(self, X, y):
        """Fit the target encoder."""
        X = self._validate_input(X)
        y = np.array(y)

        # Determine task type
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            self.task_type_ = "classification"
        else:
            self.task_type_ = "regression"

        # Calculate global mean
        self.global_mean_ = np.mean(y)

        # Store encodings for each column
        self.encodings_ = {}

        # Set up cross-validation
        if isinstance(self.cv, int):
            if self.task_type_ == "classification":
                cv_splitter = StratifiedKFold(
                    n_splits=self.cv, shuffle=True, random_state=42
                )
            else:
                cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_splitter = self.cv

        for col_idx in range(X.shape[1]):
            self.encodings_[col_idx] = self._fit_column(X[:, col_idx], y, cv_splitter)

        return self

    def _fit_column(self, column, y, cv_splitter):
        """Fit target encoding for a single column."""
        unique_values = np.unique(column)
        encoding_map = {}

        # Calculate out-of-fold encodings to prevent overfitting
        oof_encodings = np.full(len(column), self.global_mean_)

        for train_idx, val_idx in cv_splitter.split(column, y):
            train_col = column[train_idx]
            train_y = y[train_idx]

            # Calculate mean target for each category in training fold
            fold_encodings = {}
            for category in unique_values:
                mask = train_col == category
                if np.sum(mask) >= self.min_samples_leaf:
                    category_mean = np.mean(train_y[mask])
                    category_count = np.sum(mask)

                    # Apply smoothing
                    smoothed_mean = (
                        category_count * category_mean + self.smooth * self.global_mean_
                    ) / (category_count + self.smooth)
                    fold_encodings[category] = smoothed_mean
                else:
                    fold_encodings[category] = self.global_mean_

            # Apply encodings to validation fold
            for i, val_i in enumerate(val_idx):
                category = column[val_i]
                oof_encodings[val_i] = fold_encodings.get(category, self.global_mean_)

        # Calculate final encodings using all data
        for category in unique_values:
            mask = column == category
            if np.sum(mask) >= self.min_samples_leaf:
                category_mean = np.mean(y[mask])
                category_count = np.sum(mask)

                # Apply smoothing
                smoothed_mean = (
                    category_count * category_mean + self.smooth * self.global_mean_
                ) / (category_count + self.smooth)
                encoding_map[category] = smoothed_mean
            else:
                encoding_map[category] = self.global_mean_

        return {"encoding_map": encoding_map, "oof_encodings": oof_encodings}

    def transform(self, X):
        """Transform categorical features using target encoding."""
        check_is_fitted(self)
        X = self._validate_input(X)

        result = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            encoding_info = self.encodings_[col_idx]
            encoding_map = encoding_info["encoding_map"]

            for i, category in enumerate(X[:, col_idx]):
                encoded_value = encoding_map.get(category, self.global_mean_)

                # Add noise for regularization
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level)
                    encoded_value += noise

                result[i, col_idx] = encoded_value

        return result

    def _validate_input(self, X):
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        else:
            return np.array(X)


class BayesianTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Bayesian target encoder with hierarchical smoothing.

    Parameters
    ----------
    alpha : float, default=1.0
        Prior strength parameter
    beta : float, default=1.0
        Prior variance parameter
    hierarchy : list, default=None
        Hierarchical structure for smoothing (e.g., ['category', 'subcategory'])
    """

    def __init__(self, alpha=1.0, beta=1.0, hierarchy=None):
        self.alpha = alpha
        self.beta = beta
        self.hierarchy = hierarchy

    def fit(self, X, y):
        """Fit the Bayesian target encoder."""
        X = self._validate_input(X)
        y = np.array(y)

        # Calculate prior parameters
        self.global_mean_ = np.mean(y)
        self.global_var_ = np.var(y)

        # Store encodings for each column
        self.encodings_ = {}

        for col_idx in range(X.shape[1]):
            self.encodings_[col_idx] = self._fit_bayesian_column(X[:, col_idx], y)

        return self

    def _fit_bayesian_column(self, column, y):
        """Fit Bayesian encoding for a single column."""
        unique_values = np.unique(column)
        encoding_map = {}

        for category in unique_values:
            mask = column == category
            category_y = y[mask]
            n_samples = len(category_y)

            if n_samples > 0:
                sample_mean = np.mean(category_y)
                sample_var = np.var(category_y) if n_samples > 1 else self.global_var_

                # Bayesian update
                posterior_precision = self.alpha + n_samples / sample_var
                posterior_mean = (
                    self.alpha * self.global_mean_
                    + n_samples * sample_mean / sample_var
                ) / posterior_precision

                encoding_map[category] = posterior_mean
            else:
                encoding_map[category] = self.global_mean_

        return encoding_map

    def transform(self, X):
        """Transform using Bayesian target encoding."""
        check_is_fitted(self)
        X = self._validate_input(X)

        result = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            encoding_map = self.encodings_[col_idx]

            for i, category in enumerate(X[:, col_idx]):
                result[i, col_idx] = encoding_map.get(category, self.global_mean_)

        return result

    def _validate_input(self, X):
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        else:
            return np.array(X)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features by their frequency of occurrence.

    Parameters
    ----------
    normalize : bool, default=True
        Whether to normalize frequencies to [0, 1]
    handle_unknown : str, default='value'
        How to handle unknown categories: 'value', 'ignore'
    unknown_value : float, default=0
        Value to assign to unknown categories
    """

    def __init__(self, normalize=True, handle_unknown="value", unknown_value=0):
        self.normalize = normalize
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        """Fit the frequency encoder."""
        X = self._validate_input(X)

        self.frequency_maps_ = {}

        for col_idx in range(X.shape[1]):
            column = X[:, col_idx]

            # Calculate frequencies
            unique_values, counts = np.unique(column, return_counts=True)

            if self.normalize:
                frequencies = counts / len(column)
            else:
                frequencies = counts

            self.frequency_maps_[col_idx] = dict(
                zip(unique_values, frequencies, strict=False)
            )

        return self

    def transform(self, X):
        """Transform using frequency encoding."""
        check_is_fitted(self)
        X = self._validate_input(X)

        result = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            frequency_map = self.frequency_maps_[col_idx]

            for i, category in enumerate(X[:, col_idx]):
                if category in frequency_map:
                    result[i, col_idx] = frequency_map[category]
                else:
                    if self.handle_unknown == "value":
                        result[i, col_idx] = self.unknown_value
                    else:  # ignore
                        result[i, col_idx] = 0

        return result

    def _validate_input(self, X):
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        else:
            return np.array(X)


class RareClassEncoder(BaseEstimator, TransformerMixin):
    """
    Group rare categories into a single 'rare' category.

    Parameters
    ----------
    min_frequency : int or float, default=0.01
        Minimum frequency threshold. If int, minimum count; if float, minimum proportion.
    rare_label : str, default='rare'
        Label for rare categories
    """

    def __init__(self, min_frequency=0.01, rare_label="rare"):
        self.min_frequency = min_frequency
        self.rare_label = rare_label

    def fit(self, X, y=None):
        """Fit the rare class encoder."""
        X = self._validate_input(X)

        self.category_maps_ = {}

        for col_idx in range(X.shape[1]):
            column = X[:, col_idx]
            unique_values, counts = np.unique(column, return_counts=True)

            # Determine threshold
            if isinstance(self.min_frequency, float):
                threshold = self.min_frequency * len(column)
            else:
                threshold = self.min_frequency

            # Create mapping
            category_map = {}
            for value, count in zip(unique_values, counts, strict=False):
                if count >= threshold:
                    category_map[value] = value
                else:
                    category_map[value] = self.rare_label

            self.category_maps_[col_idx] = category_map

        return self

    def transform(self, X):
        """Transform by grouping rare categories."""
        check_is_fitted(self)
        X = self._validate_input(X)

        result = X.copy().astype(object)

        for col_idx in range(X.shape[1]):
            category_map = self.category_maps_[col_idx]

            for i, category in enumerate(X[:, col_idx]):
                result[i, col_idx] = category_map.get(category, self.rare_label)

        return result

    def _validate_input(self, X):
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        else:
            return np.array(X)


class OrdinalEncoderAdvanced(BaseEstimator, TransformerMixin):
    """
    Advanced ordinal encoder with automatic ordering and missing value handling.

    Parameters
    ----------
    ordering : str or dict, default='auto'
        Ordering strategy: 'auto', 'frequency', 'alphabetical', or custom dict
    handle_unknown : str, default='use_encoded_value'
        How to handle unknown categories
    unknown_value : int, default=-1
        Value for unknown categories
    """

    def __init__(
        self, ordering="auto", handle_unknown="use_encoded_value", unknown_value=-1
    ):
        self.ordering = ordering
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        """Fit the advanced ordinal encoder."""
        X = self._validate_input(X)

        self.category_maps_ = {}

        for col_idx in range(X.shape[1]):
            column = X[:, col_idx]
            unique_values = np.unique(column[~pd.isna(column)])

            if self.ordering == "auto":
                # Try to detect natural ordering
                if self._is_numeric_like(unique_values):
                    ordered_categories = sorted(
                        unique_values,
                        key=lambda x: float(x)
                        if x.replace(".", "").replace("-", "").isdigit()
                        else float("inf"),
                    )
                else:
                    ordered_categories = sorted(unique_values)
            elif self.ordering == "frequency":
                # Order by frequency
                unique_values, counts = np.unique(column, return_counts=True)
                ordered_categories = [
                    x
                    for _, x in sorted(
                        zip(counts, unique_values, strict=False), reverse=True
                    )
                ]
            elif self.ordering == "alphabetical":
                ordered_categories = sorted(unique_values)
            elif isinstance(self.ordering, dict) and col_idx in self.ordering:
                ordered_categories = self.ordering[col_idx]
            else:
                ordered_categories = sorted(unique_values)

            # Create encoding map
            encoding_map = {cat: i for i, cat in enumerate(ordered_categories)}
            self.category_maps_[col_idx] = encoding_map

        return self

    def _is_numeric_like(self, values):
        """Check if values look like they should be ordered numerically."""
        try:
            # Try to convert to float
            numeric_values = []
            for val in values:
                if (
                    isinstance(val, str)
                    and val.replace(".", "").replace("-", "").isdigit()
                ):
                    numeric_values.append(float(val))
                elif isinstance(val, (int, float)):
                    numeric_values.append(float(val))
                else:
                    return False
            return len(numeric_values) == len(values)
        except:
            return False

    def transform(self, X):
        """Transform using ordinal encoding."""
        check_is_fitted(self)
        X = self._validate_input(X)

        result = np.zeros_like(X, dtype=int)

        for col_idx in range(X.shape[1]):
            encoding_map = self.category_maps_[col_idx]

            for i, category in enumerate(X[:, col_idx]):
                if pd.isna(category):
                    result[i, col_idx] = self.unknown_value
                elif category in encoding_map:
                    result[i, col_idx] = encoding_map[category]
                else:
                    if self.handle_unknown == "use_encoded_value":
                        result[i, col_idx] = self.unknown_value
                    else:
                        raise ValueError(f"Unknown category: {category}")

        return result

    def _validate_input(self, X):
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, list):
            return np.array(X)
        else:
            return np.array(X)
