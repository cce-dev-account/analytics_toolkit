"""
Advanced feature selection methods with automated selection capabilities.
"""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_array, check_is_fitted


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature selector that combines multiple selection methods.

    Parameters
    ----------
    methods : list, default=['variance', 'correlation', 'mutual_info']
        Feature selection methods to apply in sequence
    variance_threshold : float, default=0.01
        Variance threshold for removing low-variance features
    correlation_threshold : float, default=0.95
        Correlation threshold for removing highly correlated features
    mutual_info_k : int, default=50
        Number of top features to keep based on mutual information
    recursive_estimator : estimator, default=None
        Estimator for recursive feature elimination
    task_type : str, default='auto'
        Task type: 'classification', 'regression', or 'auto'
    """

    def __init__(
        self,
        methods=["variance", "correlation", "mutual_info"],
        variance_threshold=0.01,
        correlation_threshold=0.95,
        mutual_info_k=50,
        recursive_estimator=None,
        task_type="auto",
    ):
        self.methods = methods
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mutual_info_k = mutual_info_k
        self.recursive_estimator = recursive_estimator
        self.task_type = task_type

    def fit(self, X, y=None):
        """Fit the feature selector."""
        X = check_array(X, ensure_all_finite="allow-nan")

        if y is not None:
            y = np.array(y)
            if self.task_type == "auto":
                # Auto-detect task type
                unique_values = len(np.unique(y))
                if unique_values <= 20 and np.all(y == y.astype(int)):
                    self.task_type_ = "classification"
                else:
                    self.task_type_ = "regression"
            else:
                self.task_type_ = self.task_type
        else:
            self.task_type_ = None

        # Initialize selection mask
        self.selected_features_ = np.ones(X.shape[1], dtype=bool)
        self.selection_history_ = {}

        # Apply selection methods in sequence
        for method in self.methods:
            if method == "variance":
                self._apply_variance_threshold(X)
            elif method == "correlation":
                self._apply_correlation_filter(X)
            elif method == "mutual_info" and y is not None:
                self._apply_mutual_info_selection(X, y)
            elif (
                method == "recursive"
                and self.recursive_estimator is not None
                and y is not None
            ):
                self._apply_recursive_elimination(X, y)
            elif method == "stability":
                self._apply_stability_selection(X, y)

        return self

    def _apply_variance_threshold(self, X):
        """Apply variance-based feature selection."""
        current_features = np.where(self.selected_features_)[0]
        X_selected = X[:, current_features]

        variances = np.var(X_selected, axis=0)
        mask = variances > self.variance_threshold

        # Update global selection
        new_selection = np.zeros_like(self.selected_features_)
        new_selection[current_features[mask]] = True
        self.selected_features_ = new_selection

        removed = np.sum(~mask)
        self.selection_history_["variance"] = {
            "removed": removed,
            "remaining": np.sum(self.selected_features_),
        }

    def _apply_correlation_filter(self, X):
        """Remove highly correlated features."""
        current_features = np.where(self.selected_features_)[0]
        X_selected = X[:, current_features]

        if X_selected.shape[1] <= 1:
            return

        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_selected.T)
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix)

        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(current_features)):
            for j in range(i + 1, len(current_features)):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove feature with lower variance
                    var_i = np.var(X_selected[:, i])
                    var_j = np.var(X_selected[:, j])
                    if var_i < var_j:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        # Update selection
        mask = np.ones(len(current_features), dtype=bool)
        mask[list(to_remove)] = False

        new_selection = np.zeros_like(self.selected_features_)
        new_selection[current_features[mask]] = True
        self.selected_features_ = new_selection

        self.selection_history_["correlation"] = {
            "removed": len(to_remove),
            "remaining": np.sum(self.selected_features_),
        }

    def _apply_mutual_info_selection(self, X, y):
        """Select features based on mutual information."""
        current_features = np.where(self.selected_features_)[0]
        X_selected = X[:, current_features]

        if self.task_type_ == "classification":
            mi_scores = mutual_info_classif(X_selected, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_selected, y, random_state=42)

        # Select top k features
        k = min(self.mutual_info_k, len(current_features))
        top_indices = np.argsort(mi_scores)[-k:]

        # Update selection
        new_selection = np.zeros_like(self.selected_features_)
        new_selection[current_features[top_indices]] = True
        self.selected_features_ = new_selection

        self.selection_history_["mutual_info"] = {
            "removed": len(current_features) - k,
            "remaining": k,
            "scores": mi_scores[top_indices],
        }

    def _apply_recursive_elimination(self, X, y):
        """Apply recursive feature elimination."""
        from sklearn.feature_selection import RFE

        current_features = np.where(self.selected_features_)[0]
        X_selected = X[:, current_features]

        # Use RFE to select features
        rfe = RFE(estimator=self.recursive_estimator, n_features_to_select=0.5)
        rfe.fit(X_selected, y)

        # Update selection
        new_selection = np.zeros_like(self.selected_features_)
        new_selection[current_features[rfe.support_]] = True
        self.selected_features_ = new_selection

        self.selection_history_["recursive"] = {
            "removed": np.sum(~rfe.support_),
            "remaining": np.sum(rfe.support_),
            "ranking": rfe.ranking_,
        }

    def _apply_stability_selection(self, X, y):
        """Apply stability selection using bootstrap sampling."""
        if y is None:
            return

        current_features = np.where(self.selected_features_)[0]
        X_selected = X[:, current_features]

        n_bootstrap = 50
        selection_probs = np.zeros(X_selected.shape[1])

        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = X_selected.shape[0]
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_selected[bootstrap_idx]
            y_boot = y[bootstrap_idx]

            # Fit sparse model
            if self.task_type_ == "classification":
                model = LogisticRegressionCV(
                    penalty="l1", solver="liblinear", random_state=42
                )
            else:
                model = LassoCV(random_state=42)

            try:
                model.fit(X_boot, y_boot)
                # Count non-zero coefficients
                if hasattr(model, "coef_"):
                    coefs = model.coef_
                    if coefs.ndim > 1:
                        coefs = coefs[0]
                    selection_probs += np.abs(coefs) > 1e-5
            except:
                continue

        # Select features that appear in at least 60% of bootstrap samples
        selection_probs /= n_bootstrap
        stable_features = selection_probs > 0.6

        # Update selection
        new_selection = np.zeros_like(self.selected_features_)
        new_selection[current_features[stable_features]] = True
        self.selected_features_ = new_selection

        self.selection_history_["stability"] = {
            "removed": np.sum(~stable_features),
            "remaining": np.sum(stable_features),
            "selection_probs": selection_probs[stable_features],
        }

    def transform(self, X):
        """Transform data by selecting features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get mask or indices of selected features."""
        check_is_fitted(self)
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)

        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.selected_features_))]

        return np.array(input_features)[self.selected_features_]


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on mutual information scores.

    Parameters
    ----------
    k : int or float, default=10
        Number of features to select. If float, interpreted as fraction.
    score_func : callable, default=None
        Custom scoring function. If None, uses appropriate mutual info function.
    """

    def __init__(self, k=10, score_func=None):
        self.k = k
        self.score_func = score_func

    def fit(self, X, y):
        """Fit the mutual information selector."""
        X = check_array(X, ensure_all_finite="allow-nan")
        y = np.array(y)

        # Determine task type
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            self.task_type_ = "classification"
            if self.score_func is None:
                self.score_func = mutual_info_classif
        else:
            self.task_type_ = "regression"
            if self.score_func is None:
                self.score_func = mutual_info_regression

        # Calculate mutual information scores
        self.scores_ = self.score_func(X, y, random_state=42)

        # Determine number of features to select
        if isinstance(self.k, float):
            k = int(self.k * X.shape[1])
        else:
            k = min(self.k, X.shape[1])

        # Select top k features
        self.selected_indices_ = np.argsort(self.scores_)[-k:]
        self.selected_features_ = np.zeros(X.shape[1], dtype=bool)
        self.selected_features_[self.selected_indices_] = True

        return self

    def transform(self, X):
        """Transform data by selecting features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get mask or indices of selected features."""
        check_is_fitted(self)
        if indices:
            return self.selected_indices_
        return self.selected_features_


class VarianceThresholdAdvanced(BaseEstimator, TransformerMixin):
    """
    Advanced variance threshold with automatic threshold selection.

    Parameters
    ----------
    threshold : float or str, default='auto'
        Variance threshold. If 'auto', automatically determines threshold.
    percentile : float, default=25
        Percentile to use for automatic threshold (only if threshold='auto')
    """

    def __init__(self, threshold="auto", percentile=25):
        self.threshold = threshold
        self.percentile = percentile

    def fit(self, X, y=None):
        """Fit the variance threshold selector."""
        X = check_array(X, ensure_all_finite="allow-nan")

        self.variances_ = np.var(X, axis=0)

        if self.threshold == "auto":
            self.threshold_ = np.percentile(self.variances_, self.percentile)
        else:
            self.threshold_ = self.threshold

        self.selected_features_ = self.variances_ > self.threshold_

        return self

    def transform(self, X):
        """Transform data by removing low-variance features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get mask or indices of selected features."""
        check_is_fitted(self)
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with high correlation to other features.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold above which features are considered redundant
    method : str, default='variance'
        Method to choose which feature to keep: 'variance', 'random', 'first'
    """

    def __init__(self, threshold=0.95, method="variance"):
        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        """Fit the correlation filter."""
        X = check_array(X, ensure_all_finite="allow-nan")

        # Compute correlation matrix
        self.corr_matrix_ = np.corrcoef(X.T)
        self.corr_matrix_ = np.nan_to_num(self.corr_matrix_)

        # Find features to remove
        to_remove = set()
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                if abs(self.corr_matrix_[i, j]) > self.threshold:
                    if self.method == "variance":
                        # Keep feature with higher variance
                        var_i = np.var(X[:, i])
                        var_j = np.var(X[:, j])
                        if var_i < var_j:
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
                    elif self.method == "random":
                        # Randomly choose which to remove
                        to_remove.add(np.random.choice([i, j]))
                    elif self.method == "first":
                        # Always remove the second feature
                        to_remove.add(j)

        self.selected_features_ = np.ones(X.shape[1], dtype=bool)
        self.selected_features_[list(to_remove)] = False

        return self

    def transform(self, X):
        """Transform data by removing correlated features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get mask or indices of selected features."""
        check_is_fitted(self)
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_


class RecursiveFeatureElimination(BaseEstimator, TransformerMixin):
    """
    Recursive Feature Elimination with cross-validation for optimal number of features.

    Parameters
    ----------
    estimator : estimator
        Base estimator for feature ranking
    min_features : int, default=1
        Minimum number of features to select
    step : int or float, default=1
        Number/fraction of features to remove at each step
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default=None
        Scoring metric for cross-validation
    """

    def __init__(self, estimator, min_features=1, step=1, cv=5, scoring=None):
        self.estimator = estimator
        self.min_features = min_features
        self.step = step
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        """Fit the recursive feature elimination."""
        from sklearn.feature_selection import RFECV

        X = check_array(X, ensure_all_finite="allow-nan")
        y = np.array(y)

        # Use RFECV for automatic selection
        self.selector_ = RFECV(
            estimator=self.estimator,
            min_features_to_select=self.min_features,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )

        self.selector_.fit(X, y)
        self.selected_features_ = self.selector_.support_

        return self

    def transform(self, X):
        """Transform data using selected features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get mask or indices of selected features."""
        check_is_fitted(self)
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_

    @property
    def ranking_(self):
        """Feature ranking."""
        check_is_fitted(self)
        return self.selector_.ranking_
