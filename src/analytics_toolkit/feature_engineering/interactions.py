"""
Automated interaction detection and generation for feature engineering.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_array, check_is_fitted


class InteractionDetector(BaseEstimator, TransformerMixin):
    """
    Detect significant feature interactions using multiple methods.

    Parameters
    ----------
    method : str, default='tree_based'
        Detection method: 'tree_based', 'mutual_info', 'correlation', 'statistical'
    max_interactions : int, default=50
        Maximum number of interactions to detect
    min_interaction_strength : float, default=0.01
        Minimum strength threshold for interactions
    max_depth : int, default=3
        Maximum interaction order (2=pairwise, 3=three-way, etc.)
    """

    def __init__(
        self,
        method="tree_based",
        max_interactions=50,
        min_interaction_strength=0.01,
        max_depth=3,
    ):
        self.method = method
        self.max_interactions = max_interactions
        self.min_interaction_strength = min_interaction_strength
        self.max_depth = max_depth

    def fit(self, X, y):
        """Detect interactions in the data."""
        X = check_array(X, ensure_all_finite="allow-nan")
        y = np.array(y)

        # Determine task type
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            self.task_type_ = "classification"
        else:
            self.task_type_ = "regression"

        if self.method == "tree_based":
            self.interactions_ = self._detect_tree_interactions(X, y)
        elif self.method == "mutual_info":
            self.interactions_ = self._detect_mutual_info_interactions(X, y)
        elif self.method == "correlation":
            self.interactions_ = self._detect_correlation_interactions(X, y)
        elif self.method == "statistical":
            self.interactions_ = self._detect_statistical_interactions(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def _detect_tree_interactions(self, X, y):
        """Detect interactions using tree-based feature importance."""
        if self.task_type_ == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Get baseline feature importance
        model.fit(X, y)

        interactions = []

        # Test pairwise interactions
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                # Create interaction feature
                interaction_feature = X[:, i] * X[:, j]
                X_with_interaction = np.column_stack([X, interaction_feature])

                # Fit model with interaction
                model_with_interaction = type(model)(n_estimators=100, random_state=42)
                model_with_interaction.fit(X_with_interaction, y)

                # Calculate interaction strength
                interaction_importance = model_with_interaction.feature_importances_[-1]

                if interaction_importance > self.min_interaction_strength:
                    interactions.append(
                        {
                            "features": (i, j),
                            "strength": interaction_importance,
                            "type": "multiplicative",
                        }
                    )

        # Sort by strength and limit
        interactions.sort(key=lambda x: x["strength"], reverse=True)
        return interactions[: self.max_interactions]

    def _detect_mutual_info_interactions(self, X, y):
        """Detect interactions using mutual information."""
        if self.task_type_ == "classification":
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        # Get baseline mutual information for individual features
        baseline_mi = mi_func(X, y, random_state=42)

        interactions = []

        # Test pairwise interactions
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                # Create different types of interactions
                interaction_types = {
                    "multiplicative": X[:, i] * X[:, j],
                    "additive": X[:, i] + X[:, j],
                    "ratio": np.divide(X[:, i], X[:, j] + 1e-10),
                    "difference": np.abs(X[:, i] - X[:, j]),
                }

                for interaction_type, interaction_feature in interaction_types.items():
                    # Calculate mutual information of interaction
                    try:
                        mi_score = mi_func(
                            interaction_feature.reshape(-1, 1), y, random_state=42
                        )[0]

                        # Calculate interaction strength (above what individual features contribute)
                        max_individual_mi = max(baseline_mi[i], baseline_mi[j])
                        interaction_strength = mi_score - max_individual_mi

                        if interaction_strength > self.min_interaction_strength:
                            interactions.append(
                                {
                                    "features": (i, j),
                                    "strength": interaction_strength,
                                    "type": interaction_type,
                                    "mi_score": mi_score,
                                }
                            )
                    except:
                        continue

        # Sort by strength and limit
        interactions.sort(key=lambda x: x["strength"], reverse=True)
        return interactions[: self.max_interactions]

    def _detect_correlation_interactions(self, X, y):
        """Detect interactions using correlation with target."""
        # Calculate correlation between individual features and target
        baseline_corr = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        baseline_corr = np.nan_to_num(baseline_corr)

        interactions = []

        # Test pairwise interactions
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                # Create interaction feature
                interaction_feature = X[:, i] * X[:, j]

                # Calculate correlation with target
                try:
                    interaction_corr = np.corrcoef(interaction_feature, y)[0, 1]
                    interaction_corr = np.nan_to_num(interaction_corr)

                    # Calculate interaction strength
                    max_individual_corr = max(
                        abs(baseline_corr[i]), abs(baseline_corr[j])
                    )
                    interaction_strength = abs(interaction_corr) - max_individual_corr

                    if interaction_strength > self.min_interaction_strength:
                        interactions.append(
                            {
                                "features": (i, j),
                                "strength": interaction_strength,
                                "type": "multiplicative",
                                "correlation": interaction_corr,
                            }
                        )
                except:
                    continue

        # Sort by strength and limit
        interactions.sort(key=lambda x: x["strength"], reverse=True)
        return interactions[: self.max_interactions]

    def _detect_statistical_interactions(self, X, y):
        """Detect interactions using statistical tests."""
        from scipy.stats import f_oneway

        interactions = []

        # For each pair of features, test for interaction effect
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                try:
                    # Create bins for features to test interaction
                    x1_bins = pd.qcut(X[:, i], q=3, duplicates="drop")
                    x2_bins = pd.qcut(X[:, j], q=3, duplicates="drop")

                    # Create interaction groups
                    interaction_groups = []
                    for cat1 in x1_bins.cat.categories:
                        for cat2 in x2_bins.cat.categories:
                            mask = (x1_bins == cat1) & (x2_bins == cat2)
                            if np.sum(mask) > 5:  # Minimum sample size
                                interaction_groups.append(y[mask])

                    if len(interaction_groups) >= 4:  # Minimum number of groups
                        # Test for significant differences between interaction groups
                        if self.task_type_ == "regression":
                            f_stat, p_value = f_oneway(*interaction_groups)
                            interaction_strength = f_stat if not np.isnan(f_stat) else 0
                        else:
                            # For classification, use chi-square or similar
                            interaction_strength = 0

                        if interaction_strength > self.min_interaction_strength:
                            interactions.append(
                                {
                                    "features": (i, j),
                                    "strength": interaction_strength,
                                    "type": "statistical",
                                    "p_value": p_value
                                    if "p_value" in locals()
                                    else None,
                                }
                            )
                except:
                    continue

        # Sort by strength and limit
        interactions.sort(key=lambda x: x["strength"], reverse=True)
        return interactions[: self.max_interactions]

    def get_interactions(self):
        """Get detected interactions."""
        check_is_fitted(self)
        return self.interactions_

    def transform(self, X):
        """This detector doesn't transform data, use InteractionGenerator for that."""
        check_is_fitted(self)
        return X


class InteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Generate interaction features based on detected interactions or specified rules.

    Parameters
    ----------
    interactions : list, default=None
        List of interactions to generate. If None, generates all pairwise interactions.
    interaction_types : list, default=['multiply', 'add', 'subtract', 'divide']
        Types of interactions to generate
    max_interactions : int, default=100
        Maximum number of interaction features to generate
    include_self_interactions : bool, default=False
        Whether to include self-interactions (e.g., x^2)
    """

    def __init__(
        self,
        interactions=None,
        interaction_types=None,
        max_interactions=100,
        include_self_interactions=False,
    ):
        if interaction_types is None:
            interaction_types = ["multiply", "add"]
        self.interactions = interactions
        self.interaction_types = interaction_types
        self.max_interactions = max_interactions
        self.include_self_interactions = include_self_interactions

    def fit(self, X, y=None):
        """Fit the interaction generator."""
        X = check_array(X, ensure_all_finite="allow-nan")
        self.n_features_in_ = X.shape[1]

        if self.interactions is None:
            # Generate all pairwise interactions
            self.feature_pairs_ = list(
                itertools.combinations(range(self.n_features_in_), 2)
            )
            if self.include_self_interactions:
                self.feature_pairs_.extend([(i, i) for i in range(self.n_features_in_)])
        else:
            # Use provided interactions
            self.feature_pairs_ = [
                (int(pair[0]), int(pair[1])) for pair in self.interactions
            ]

        # Limit number of interactions
        self.feature_pairs_ = self.feature_pairs_[: self.max_interactions]

        return self

    def transform(self, X):
        """Generate interaction features."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")

        interaction_features = []
        self.feature_names_ = []

        for i, j in self.feature_pairs_:
            for interaction_type in self.interaction_types:
                if interaction_type == "multiply":
                    feature = X[:, i] * X[:, j]
                    name = f"x{i}_mult_x{j}"
                elif interaction_type == "add":
                    feature = X[:, i] + X[:, j]
                    name = f"x{i}_add_x{j}"
                elif interaction_type == "subtract":
                    feature = X[:, i] - X[:, j]
                    name = f"x{i}_sub_x{j}"
                elif interaction_type == "divide":
                    feature = np.divide(X[:, i], X[:, j] + 1e-10)  # Add small epsilon
                    name = f"x{i}_div_x{j}"
                elif interaction_type == "power":
                    if i == j:  # Self-interaction
                        feature = X[:, i] ** 2
                        name = f"x{i}_squared"
                    else:
                        continue  # Skip power for different features
                elif interaction_type == "min":
                    feature = np.minimum(X[:, i], X[:, j])
                    name = f"x{i}_min_x{j}"
                elif interaction_type == "max":
                    feature = np.maximum(X[:, i], X[:, j])
                    name = f"x{i}_max_x{j}"
                else:
                    continue

                interaction_features.append(feature)
                self.feature_names_.append(name)

        if not interaction_features:
            return X

        # Combine original features with interactions
        interaction_matrix = np.column_stack(interaction_features)
        return np.column_stack([X, interaction_matrix])

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        return list(input_features) + self.feature_names_


class PolynomialInteractions(BaseEstimator, TransformerMixin):
    """
    Generate polynomial interaction features with intelligent selection.

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of polynomial features
    interaction_only : bool, default=True
        Only create interaction terms, not individual powers
    include_bias : bool, default=False
        Include bias column
    feature_selection : str, default='variance'
        Method for selecting best polynomial features: 'variance', 'correlation', None
    max_features : int, default=None
        Maximum number of features to keep after generation
    """

    def __init__(
        self,
        degree=2,
        interaction_only=True,
        include_bias=False,
        feature_selection="variance",
        max_features=None,
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.feature_selection = feature_selection
        self.max_features = max_features

    def fit(self, X, y=None):
        """Fit the polynomial interaction generator."""
        from sklearn.preprocessing import PolynomialFeatures

        X = check_array(X, ensure_all_finite="allow-nan")

        # Generate polynomial features
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        X_poly = self.poly_.fit_transform(X)

        # Apply feature selection if requested
        if self.feature_selection and y is not None:
            self.selected_features_ = self._select_features(X_poly, y)
        else:
            self.selected_features_ = np.arange(X_poly.shape[1])

        return self

    def _select_features(self, X, y):
        """Select polynomial features based on specified criteria."""
        n_features = X.shape[1]

        if self.feature_selection == "variance":
            # Select features with highest variance
            variances = np.var(X, axis=0)
            if self.max_features and self.max_features < n_features:
                selected = np.argsort(variances)[-self.max_features :]
            else:
                # Remove features with very low variance
                threshold = np.percentile(variances, 25)
                selected = np.where(variances > threshold)[0]

        elif self.feature_selection == "correlation" and y is not None:
            # Select features with highest correlation to target
            correlations = [
                abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(n_features)
            ]
            correlations = np.nan_to_num(correlations)
            if self.max_features and self.max_features < n_features:
                selected = np.argsort(correlations)[-self.max_features :]
            else:
                # Remove features with very low correlation
                threshold = np.percentile(correlations, 50)
                selected = np.where(correlations > threshold)[0]

        else:
            selected = np.arange(n_features)

        return selected

    def transform(self, X):
        """Transform data using polynomial interactions."""
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
