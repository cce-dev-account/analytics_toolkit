"""
Utility functions for PyTorch regression module.
"""


import numpy as np
import pandas as pd
import torch


def to_tensor(
    data: np.ndarray | pd.DataFrame | pd.Series | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert various data types to PyTorch tensor.

    Parameters
    ----------
    data : array-like
        Data to convert.
    device : torch.device
        Target device.
    dtype : torch.dtype, default=torch.float32
        Target dtype.

    Returns
    -------
    tensor : torch.Tensor
        Converted tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return torch.tensor(data.values, device=device, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device, dtype=dtype)
    else:
        return torch.tensor(data, device=device, dtype=dtype)


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect categorical columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    categorical_cols : list
        List of categorical column names.
    """
    categorical_cols = []

    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            categorical_cols.append(col)
        elif df[col].dtype in ["int64", "int32"] and df[col].nunique() <= 10:
            # Consider integer columns with few unique values as categorical
            categorical_cols.append(col)

    return categorical_cols


def create_dummy_variables(
    df: pd.DataFrame,
    categorical_cols: list[str],
    encoding_mappings: dict | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Create dummy variables for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_cols : list
        List of categorical column names.
    encoding_mappings : dict, optional
        Pre-existing encoding mappings for consistent encoding.

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with dummy variables.
    feature_names : list
        Updated feature names.
    mappings : dict
        Encoding mappings for future use.
    """
    df_encoded = df.copy()
    mappings = encoding_mappings.copy() if encoding_mappings else {}

    for col in categorical_cols:
        if col in df_encoded.columns:
            # Get unique categories
            if col in mappings:
                # Use existing mappings for consistency
                categories = mappings[col]
                # Handle unseen categories by mapping to all zeros (reference category)
                def map_categories(x, cats=categories):
                    return x if x in cats else cats[0]
                df_col = df_encoded[col].map(map_categories)
            else:
                categories = sorted(df_encoded[col].unique())
                mappings[col] = categories
                df_col = df_encoded[col]

            # Create dummy variables (drop first category as reference)
            # If we have existing mappings, ensure consistent columns
            if col in mappings:
                # Create all possible dummy columns based on training data
                expected_categories = mappings[col][1:]  # Skip first (reference) category
                expected_columns = [f"{col}_{cat}" for cat in expected_categories]

                # Create dummies for current data
                dummies = pd.get_dummies(df_col, prefix=col, drop_first=True, dtype=int)

                # Ensure all expected columns exist (fill missing with zeros)
                for expected_col in expected_columns:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0

                # Reorder columns to match expected order
                dummies = dummies.reindex(columns=expected_columns, fill_value=0)
            else:
                dummies = pd.get_dummies(df_col, prefix=col, drop_first=True, dtype=int)

            # Remove original column and add dummies
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

    feature_names = df_encoded.columns.tolist()
    return df_encoded, feature_names, mappings


def calculate_vif(
    X: np.ndarray | pd.DataFrame, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for features.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    feature_names : list, optional
        Feature names.

    Returns
    -------
    vif_df : pd.DataFrame
        DataFrame with features and their VIF values.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    elif feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    n_features = X.shape[1]
    vif_values = []

    for i in range(n_features):
        # Use other features to predict feature i
        X_others = np.delete(X, i, axis=1)
        y_target = X[:, i]

        # Fit linear regression (using numpy for simplicity)
        try:
            # Add intercept
            X_others_with_intercept = np.column_stack(
                [np.ones(X_others.shape[0]), X_others]
            )

            # Solve normal equations
            coef = np.linalg.solve(
                X_others_with_intercept.T @ X_others_with_intercept,
                X_others_with_intercept.T @ y_target,
            )

            # Calculate R-squared
            y_pred = X_others_with_intercept @ coef
            ss_res = np.sum((y_target - y_pred) ** 2)
            ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate VIF
            vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
            vif_values.append(vif)

        except np.linalg.LinAlgError:
            # Singular matrix - perfect multicollinearity
            vif_values.append(np.inf)

    vif_df = pd.DataFrame({"feature": feature_names, "VIF": vif_values})

    # Add warnings
    moderate_multicollinearity = (vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)
    high_multicollinearity = vif_df["VIF"] > 10

    if moderate_multicollinearity.any():
        print("Warning: Moderate multicollinearity detected (VIF > 5):")
        print(vif_df[moderate_multicollinearity][["feature", "VIF"]])

    if high_multicollinearity.any():
        print("Warning: High multicollinearity detected (VIF > 10):")
        print(vif_df[high_multicollinearity][["feature", "VIF"]])

    return vif_df


def add_polynomial_features(
    X: np.ndarray | pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
) -> np.ndarray | pd.DataFrame:
    """
    Generate polynomial features.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    degree : int, default=2
        Maximum degree of polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    include_bias : bool, default=False
        If True, include a bias column (all ones).

    Returns
    -------
    X_poly : array-like
        Polynomial features.
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        return poly.fit_transform(X)
    except ImportError as err:
        raise ImportError("scikit-learn is required for polynomial features") from err


def standardize_features(
    X: np.ndarray | pd.DataFrame, fit_params: dict | None = None
) -> tuple[np.ndarray | pd.DataFrame, dict]:
    """
    Standardize features (z-score normalization).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    fit_params : dict, optional
        Pre-computed mean and std for consistent scaling.

    Returns
    -------
    X_scaled : array-like
        Standardized features.
    params : dict
        Scaling parameters (mean and std).
    """
    if isinstance(X, pd.DataFrame):
        if fit_params is None:
            mean = X.mean()
            std = X.std()
            params = {"mean": mean, "std": std}
        else:
            mean = fit_params["mean"]
            std = fit_params["std"]
            params = fit_params

        X_scaled = (X - mean) / std
        return X_scaled, params
    else:
        if fit_params is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            params = {"mean": mean, "std": std}
        else:
            mean = fit_params["mean"]
            std = fit_params["std"]
            params = fit_params

        X_scaled = (X - mean) / std
        return X_scaled, params


def check_input_consistency(
    X_train: np.ndarray | pd.DataFrame, X_test: np.ndarray | pd.DataFrame
) -> None:
    """
    Check that training and test data have consistent structure.

    Parameters
    ----------
    X_train : array-like
        Training features.
    X_test : array-like
        Test features.

    Raises
    ------
    ValueError
        If data structures are inconsistent.
    """
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if not X_train.columns.equals(X_test.columns):
            raise ValueError("Training and test DataFrames must have the same columns")
    elif isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray):
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                "Training and test arrays must have the same number of features"
            )
    else:
        raise ValueError("Training and test data must be of the same type")


def split_features_target(
    data: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of target column.
    feature_cols : list, optional
        List of feature column names. If None, use all columns except target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    else:
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")

    X = data[feature_cols].copy()
    y = data[target_col].copy()

    return X, y
