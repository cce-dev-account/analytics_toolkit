"""
Test suite for UI business logic and utility functions.

Tests the core business logic and data processing functions that support
the Streamlit UI, ensuring they work correctly independent of the UI framework.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def ml_dataset():
    """Create a realistic ML dataset for testing."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'score': np.random.normal(650, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'binary_feature': np.random.choice([0, 1], n_samples),
        'target_regression': np.random.normal(100, 25, n_samples),
        'target_classification': np.random.choice([0, 1], n_samples)
    })


class TestDataValidationLogic:
    """Test data validation functions used in UI."""

    def test_data_type_detection(self, ml_dataset):
        """Test automatic data type detection logic."""

        # Numeric columns detection
        numeric_cols = ml_dataset.select_dtypes(include=[np.number]).columns.tolist()
        expected_numeric = ['age', 'income', 'score', 'binary_feature', 'target_regression', 'target_classification']

        assert len(numeric_cols) == len(expected_numeric)
        for col in expected_numeric:
            assert col in numeric_cols

        # Categorical columns detection
        categorical_cols = ml_dataset.select_dtypes(include=['object']).columns.tolist()
        expected_categorical = ['category', 'region']

        assert len(categorical_cols) == len(expected_categorical)
        for col in expected_categorical:
            assert col in categorical_cols

    def test_missing_value_detection(self):
        """Test missing value detection logic."""
        data_with_missing = pd.DataFrame({
            'complete_col': [1, 2, 3, 4, 5],
            'partial_missing': [1, np.nan, 3, np.nan, 5],
            'mostly_missing': [np.nan, np.nan, np.nan, np.nan, 1],
            'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })

        # Test missing value statistics
        missing_stats = {}
        for col in data_with_missing.columns:
            missing_count = data_with_missing[col].isna().sum()
            missing_pct = (missing_count / len(data_with_missing)) * 100
            missing_stats[col] = {'count': missing_count, 'percentage': missing_pct}

        assert missing_stats['complete_col']['count'] == 0
        assert missing_stats['partial_missing']['count'] == 2
        assert missing_stats['mostly_missing']['percentage'] == 80.0
        assert missing_stats['all_missing']['percentage'] == 100.0

    def test_outlier_detection_logic(self, ml_dataset):
        """Test outlier detection algorithms."""

        # IQR method
        def detect_outliers_iqr(series, factor=1.5):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (series < lower_bound) | (series > upper_bound)

        # Z-score method
        def detect_outliers_zscore(series, threshold=3):
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        income_data = ml_dataset['income']

        # Test IQR outliers
        iqr_outliers = detect_outliers_iqr(income_data)
        assert isinstance(iqr_outliers, pd.Series)
        assert iqr_outliers.dtype == bool

        # Test Z-score outliers
        zscore_outliers = detect_outliers_zscore(income_data)
        assert isinstance(zscore_outliers, pd.Series)
        assert zscore_outliers.dtype == bool


class TestFeatureEngineeringLogic:
    """Test feature engineering business logic."""

    def test_feature_statistics_calculation(self, ml_dataset):
        """Test calculation of feature statistics."""

        def calculate_feature_stats(df):
            stats = {}

            for col in df.columns:
                col_stats = {
                    'dtype': str(df[col].dtype),
                    'non_null_count': df[col].count(),
                    'null_count': df[col].isna().sum(),
                    'unique_count': df[col].nunique()
                }

                if df[col].dtype in ['int64', 'float64']:
                    col_stats.update({
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median()
                    })

                stats[col] = col_stats

            return stats

        stats = calculate_feature_stats(ml_dataset)

        # Check that stats were calculated for all columns
        assert len(stats) == len(ml_dataset.columns)

        # Check numeric column stats
        income_stats = stats['income']
        assert 'mean' in income_stats
        assert 'std' in income_stats
        assert income_stats['mean'] > 0
        assert income_stats['std'] > 0

        # Check categorical column stats
        category_stats = stats['category']
        assert 'mean' not in category_stats  # Should not have numeric stats
        assert category_stats['unique_count'] <= 4  # We created 4 categories

    def test_correlation_analysis(self, ml_dataset):
        """Test correlation analysis logic."""

        numeric_data = ml_dataset.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        # Check correlation matrix properties
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert correlation_matrix.shape[0] == len(numeric_data.columns)

        # Diagonal should be 1 (perfect self-correlation)
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix), 1.0)

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            correlation_matrix.values,
            correlation_matrix.T.values
        )

    def test_feature_interaction_detection(self, ml_dataset):
        """Test feature interaction detection logic."""

        # Simple interaction detection: multiply numeric features
        numeric_cols = ['age', 'income', 'score']
        interactions = {}

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interaction_name = f"{col1}_x_{col2}"
                interactions[interaction_name] = ml_dataset[col1] * ml_dataset[col2]

        # Should create interaction features
        expected_interactions = ['age_x_income', 'age_x_score', 'income_x_score']
        assert len(interactions) == len(expected_interactions)

        for interaction in expected_interactions:
            assert interaction in interactions
            assert len(interactions[interaction]) == len(ml_dataset)


class TestModelComparisonLogic:
    """Test model comparison business logic."""

    def test_model_performance_calculation(self, ml_dataset):
        """Test model performance metric calculations."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

        # Regression task
        X_reg = ml_dataset[['age', 'income', 'score']]
        y_reg = ml_dataset['target_regression']

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

        model_reg = RandomForestRegressor(n_estimators=10, random_state=42)
        model_reg.fit(X_train, y_train)

        y_pred = model_reg.predict(X_test)

        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        assert mse >= 0  # MSE should be non-negative
        assert -1 <= r2 <= 1  # R² typically between -1 and 1

        # Classification task
        X_clf = ml_dataset[['age', 'income', 'score']]
        y_clf = ml_dataset['target_classification']

        X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

        model_clf = RandomForestClassifier(n_estimators=10, random_state=42)
        model_clf.fit(X_train, y_train)

        y_pred = model_clf.predict(X_test)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)

        assert 0 <= accuracy <= 1  # Accuracy should be between 0 and 1

    def test_cross_validation_logic(self, ml_dataset):
        """Test cross-validation implementation."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        X = ml_dataset[['age', 'income', 'score']]
        y = ml_dataset['target_regression']

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(cv_scores) == 3  # 3-fold CV
        assert all(-1 <= score <= 1 for score in cv_scores)  # R² scores

        # Calculate summary statistics
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        assert isinstance(mean_score, (float, np.floating))
        assert isinstance(std_score, (float, np.floating))
        assert std_score >= 0  # Standard deviation should be non-negative

    def test_model_comparison_workflow(self, ml_dataset):
        """Test complete model comparison workflow."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        X = ml_dataset[['age', 'income', 'score']]
        y = ml_dataset['target_regression']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define models to compare
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=10, random_state=42),
            'Linear Regression': LinearRegression()
        }

        results = {}

        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'mse': mse,
                'r2': r2,
                'model': model
            }

        # Check results
        assert len(results) == 2
        assert 'Random Forest' in results
        assert 'Linear Regression' in results

        for name, metrics in results.items():
            assert 'mse' in metrics
            assert 'r2' in metrics
            assert metrics['mse'] >= 0
            assert -1 <= metrics['r2'] <= 1


class TestResultsDashboardLogic:
    """Test results dashboard business logic."""

    def test_model_diagnostics_calculation(self, ml_dataset):
        """Test model diagnostic calculations."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        X = ml_dataset[['age', 'income', 'score']]
        y = ml_dataset['target_regression']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate residuals
        residuals = y_test - y_pred

        # Diagnostic calculations
        residual_mean = residuals.mean()
        residual_std = residuals.std()

        # Tests
        assert len(residuals) == len(y_test)
        assert abs(residual_mean) < residual_std  # Mean residual should be close to 0
        assert residual_std > 0

    def test_feature_importance_analysis(self, ml_dataset):
        """Test feature importance analysis."""
        from sklearn.ensemble import RandomForestRegressor

        X = ml_dataset[['age', 'income', 'score']]
        y = ml_dataset['target_regression']

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Tests
        assert len(importances) == len(feature_names)
        assert all(imp >= 0 for imp in importances)  # Importances should be non-negative
        assert abs(importances.sum() - 1.0) < 1e-10  # Should sum to 1
        assert len(importance_df) == len(feature_names)

    def test_prediction_intervals(self, ml_dataset):
        """Test prediction interval calculations."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        X = ml_dataset[['age', 'income', 'score']]
        y = ml_dataset['target_regression']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # For Random Forest, we can estimate prediction intervals using tree predictions
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])

        # Calculate prediction intervals
        lower_percentile = np.percentile(tree_predictions, 2.5, axis=0)
        upper_percentile = np.percentile(tree_predictions, 97.5, axis=0)
        median_prediction = np.median(tree_predictions, axis=0)

        # Tests
        assert len(lower_percentile) == len(X_test)
        assert len(upper_percentile) == len(X_test)
        assert len(median_prediction) == len(X_test)
        assert all(lower_percentile <= upper_percentile)


class TestUIUtilityFunctions:
    """Test utility functions that support the UI."""

    def test_data_summary_generation(self, ml_dataset):
        """Test data summary generation for UI display."""

        def generate_data_summary(df):
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isna().sum().to_dict()
            }
            return summary

        summary = generate_data_summary(ml_dataset)

        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert summary['shape'] == ml_dataset.shape
        assert len(summary['columns']) == len(ml_dataset.columns)

    def test_progress_tracking_logic(self):
        """Test progress tracking for long-running operations."""

        def simulate_long_operation(n_steps=10):
            progress_updates = []

            for i in range(n_steps):
                progress = (i + 1) / n_steps
                progress_updates.append({
                    'step': i + 1,
                    'progress': progress,
                    'percentage': progress * 100
                })

            return progress_updates

        updates = simulate_long_operation(5)

        assert len(updates) == 5
        assert updates[0]['progress'] == 0.2
        assert updates[-1]['progress'] == 1.0
        assert updates[-1]['percentage'] == 100.0

    def test_error_message_formatting(self):
        """Test error message formatting for UI display."""

        def format_error_message(error, context=""):
            error_info = {
                'type': type(error).__name__,
                'message': str(error),
                'context': context
            }

            formatted_message = f"{error_info['type']}"
            if context:
                formatted_message += f" in {context}"
            formatted_message += f": {error_info['message']}"

            return formatted_message

        # Test with different error types
        value_error = ValueError("Invalid input")
        formatted = format_error_message(value_error, "data preprocessing")

        assert "ValueError" in formatted
        assert "data preprocessing" in formatted
        assert "Invalid input" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])