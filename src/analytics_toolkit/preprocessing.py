"""Data preprocessing utilities for analytics toolkit."""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""

    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.fitted = False

    def fit_transform(self,
                     data: pd.DataFrame,
                     target_column: Optional[str] = None,
                     numerical_columns: Optional[List[str]] = None,
                     categorical_columns: Optional[List[str]] = None,
                     scaling_method: str = 'standard') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit preprocessor and transform data."""
        data_copy = data.copy()

        # Auto-detect column types if not specified
        if numerical_columns is None:
            numerical_columns = list(data_copy.select_dtypes(include=[np.number]).columns)
            if target_column and target_column in numerical_columns:
                numerical_columns.remove(target_column)

        if categorical_columns is None:
            categorical_columns = list(data_copy.select_dtypes(include=['object', 'category']).columns)
            if target_column and target_column in categorical_columns:
                categorical_columns.remove(target_column)

        # Handle missing values
        for col in numerical_columns:
            data_copy[col] = data_copy[col].fillna(data_copy[col].median())

        for col in categorical_columns:
            data_copy[col] = data_copy[col].fillna(data_copy[col].mode().iloc[0] if not data_copy[col].mode().empty else 'Unknown')

        # Scale numerical features
        if numerical_columns:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")

            self.scalers['numerical'] = scaler
            data_copy[numerical_columns] = scaler.fit_transform(data_copy[numerical_columns])

        # Encode categorical features
        for col in categorical_columns:
            encoder = LabelEncoder()
            self.encoders[col] = encoder
            data_copy[col] = encoder.fit_transform(data_copy[col].astype(str))

        self.fitted = True

        # Handle target
        target = None
        if target_column:
            target = data_copy[target_column]
            data_copy = data_copy.drop(columns=[target_column])

        return data_copy, target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        data_copy = data.copy()

        # Apply scaling
        if 'numerical' in self.scalers:
            numerical_columns = [col for col in data_copy.columns if col in self.scalers['numerical'].feature_names_in_]
            if numerical_columns:
                data_copy[numerical_columns] = self.scalers['numerical'].transform(data_copy[numerical_columns])

        # Apply encoding
        for col, encoder in self.encoders.items():
            if col in data_copy.columns:
                # Handle new categories
                data_copy[col] = data_copy[col].astype(str)
                mask = data_copy[col].isin(encoder.classes_)
                data_copy.loc[mask, col] = encoder.transform(data_copy.loc[mask, col])
                data_copy.loc[~mask, col] = -1  # Unknown category

        return data_copy


def create_train_test_split(X: pd.DataFrame,
                           y: Optional[pd.Series] = None,
                           test_size: float = 0.2,
                           random_state: int = 42,
                           stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Create train-test split with optional stratification."""
    stratify_param = y if stratify and y is not None else None

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, None, None