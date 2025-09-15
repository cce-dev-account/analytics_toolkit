"""
Tests for utility functions.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ..utils import (
    to_tensor,
    detect_categorical_columns,
    create_dummy_variables,
    calculate_vif,
    standardize_features,
    check_input_consistency,
    split_features_target
)


class TestUtils:
    """Test suite for utility functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'continuous1': np.random.randn(n_samples),
            'continuous2': np.random.randn(n_samples),
            'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical2': np.random.choice([1, 2, 3], n_samples),
            'binary': np.random.choice([0, 1], n_samples),
            'target': np.random.randn(n_samples)
        }

        return pd.DataFrame(data)

    def test_to_tensor(self):
        """Test tensor conversion function."""
        device = torch.device('cpu')

        # Test numpy array
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        tensor = to_tensor(arr, device)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device == device
        assert tensor.dtype == torch.float32
        np.testing.assert_array_equal(tensor.cpu().numpy(), arr)

        # Test pandas DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        tensor_df = to_tensor(df, device)
        assert isinstance(tensor_df, torch.Tensor)
        assert tensor_df.shape == (3, 2)

        # Test pandas Series
        series = pd.Series([1, 2, 3])
        tensor_series = to_tensor(series, device)
        assert isinstance(tensor_series, torch.Tensor)
        assert tensor_series.shape == (3,)

        # Test torch tensor (should return as-is but moved to device)
        existing_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
        converted_tensor = to_tensor(existing_tensor, device, dtype=torch.float32)
        assert converted_tensor.device == device
        assert converted_tensor.dtype == torch.float32

        # Test list
        list_data = [[1, 2], [3, 4]]
        tensor_list = to_tensor(list_data, device)
        assert isinstance(tensor_list, torch.Tensor)
        assert tensor_list.shape == (2, 2)

    def test_detect_categorical_columns(self, sample_data):
        """Test categorical column detection."""
        categorical_cols = detect_categorical_columns(sample_data)

        # Should detect string categorical and small-cardinality integer
        expected_categorical = ['categorical1', 'categorical2', 'binary']
        assert set(categorical_cols) == set(expected_categorical)

    def test_create_dummy_variables(self, sample_data):
        """Test dummy variable creation."""
        categorical_cols = ['categorical1', 'categorical2']

        # Test initial encoding
        df_encoded, feature_names, mappings = create_dummy_variables(
            sample_data, categorical_cols
        )

        # Check that original categorical columns are removed
        for col in categorical_cols:
            assert col not in df_encoded.columns

        # Check that dummy variables are created
        assert 'categorical1_B' in df_encoded.columns
        assert 'categorical1_C' in df_encoded.columns
        assert 'categorical2_2' in df_encoded.columns
        assert 'categorical2_3' in df_encoded.columns

        # Check that mappings are stored
        assert 'categorical1' in mappings
        assert 'categorical2' in mappings

        # Test consistent encoding with existing mappings
        new_data = sample_data.iloc[:10].copy()
        df_encoded_new, _, _ = create_dummy_variables(
            new_data, categorical_cols, mappings
        )

        # Should have same dummy columns
        expected_dummies = [col for col in df_encoded.columns
                           if col.startswith(('categorical1_', 'categorical2_'))]
        actual_dummies = [col for col in df_encoded_new.columns
                         if col.startswith(('categorical1_', 'categorical2_'))]

        assert set(expected_dummies) == set(actual_dummies)

    def test_calculate_vif(self, sample_data):
        """Test VIF calculation."""
        # Create some collinear features
        X = sample_data[['continuous1', 'continuous2']].copy()
        X['collinear'] = X['continuous1'] + 0.1 * np.random.randn(len(X))

        vif_results = calculate_vif(X.values, X.columns.tolist())

        assert isinstance(vif_results, pd.DataFrame)
        assert len(vif_results) == 3
        assert 'feature' in vif_results.columns
        assert 'VIF' in vif_results.columns

        # Collinear feature should have high VIF
        collinear_vif = vif_results[vif_results['feature'] == 'collinear']['VIF'].iloc[0]
        assert collinear_vif > 5  # Should indicate multicollinearity

    def test_calculate_vif_perfect_collinearity(self):
        """Test VIF with perfect collinearity."""
        # Create perfectly collinear features
        X = np.random.randn(50, 2)
        X = np.column_stack([X, X[:, 0]])  # Third column is identical to first

        vif_results = calculate_vif(X)

        # Should have infinite VIF for collinear features
        assert any(np.isinf(vif_results['VIF']))

    def test_standardize_features(self, sample_data):
        """Test feature standardization."""
        # Test with numpy array
        X = sample_data[['continuous1', 'continuous2']].values

        X_scaled, params = standardize_features(X)

        # Check that mean is approximately 0 and std is approximately 1
        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)

        # Test with pandas DataFrame
        df = sample_data[['continuous1', 'continuous2']]
        df_scaled, params_df = standardize_features(df)

        assert isinstance(df_scaled, pd.DataFrame)
        assert df_scaled.columns.equals(df.columns)

        # Test consistent scaling with existing parameters
        new_data = sample_data[['continuous1', 'continuous2']].iloc[:10]
        new_scaled, _ = standardize_features(new_data, params_df)

        # Should use same scaling parameters
        expected_scaled = (new_data - params_df['mean']) / params_df['std']
        pd.testing.assert_frame_equal(new_scaled, expected_scaled)

    def test_check_input_consistency(self, sample_data):
        """Test input consistency checking."""
        X_train = sample_data[['continuous1', 'continuous2']]
        X_test = sample_data[['continuous1', 'continuous2']].iloc[:10]

        # Should not raise error for consistent data
        check_input_consistency(X_train, X_test)

        # Test with inconsistent columns
        X_test_bad = sample_data[['continuous1', 'categorical1']].iloc[:10]
        with pytest.raises(ValueError):
            check_input_consistency(X_train, X_test_bad)

        # Test with numpy arrays
        X_train_np = X_train.values
        X_test_np = X_test.values

        check_input_consistency(X_train_np, X_test_np)

        # Test with different number of features
        X_test_bad_np = X_test_np[:, :1]  # Only one feature
        with pytest.raises(ValueError):
            check_input_consistency(X_train_np, X_test_bad_np)

        # Test with mixed types
        with pytest.raises(ValueError):
            check_input_consistency(X_train, X_test_np)

    def test_split_features_target(self, sample_data):
        """Test feature-target splitting."""
        # Test basic splitting
        X, y = split_features_target(sample_data, 'target')

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'target' not in X.columns
        assert len(X) == len(y) == len(sample_data)

        # Test with specific feature columns
        feature_cols = ['continuous1', 'continuous2']
        X_subset, y_subset = split_features_target(
            sample_data, 'target', feature_cols
        )

        assert list(X_subset.columns) == feature_cols
        assert len(X_subset.columns) == 2

        # Test error cases
        with pytest.raises(ValueError):
            split_features_target(sample_data, 'nonexistent_target')

        with pytest.raises(ValueError):
            split_features_target(sample_data, 'target', ['nonexistent_feature'])

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty data
        empty_df = pd.DataFrame()
        categorical_cols = detect_categorical_columns(empty_df)
        assert categorical_cols == []

        # Test single column
        single_col_df = pd.DataFrame({'a': [1, 2, 3]})
        categorical_cols = detect_categorical_columns(single_col_df)
        assert categorical_cols == []

        # Test all categorical
        all_cat_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        categorical_cols = detect_categorical_columns(all_cat_df)
        assert len(categorical_cols) == 2

    def test_vif_edge_cases(self):
        """Test VIF calculation edge cases."""
        # Test with too few samples
        X_small = np.random.randn(2, 3)
        with pytest.warns(None):  # Should handle gracefully
            vif_results = calculate_vif(X_small)
            assert len(vif_results) == 3

        # Test with single feature
        X_single = np.random.randn(50, 1)
        vif_results = calculate_vif(X_single)
        assert len(vif_results) == 1
        # VIF for single feature should be 1 (or inf if constant)

    def test_standardize_edge_cases(self):
        """Test standardization edge cases."""
        # Test with constant features
        X_constant = np.ones((50, 2))
        X_constant[:, 1] = np.random.randn(50)

        X_scaled, params = standardize_features(X_constant)

        # Constant feature should remain constant (NaN handling)
        assert np.all(np.isnan(X_scaled[:, 0]) | (X_scaled[:, 0] == 0))

        # Non-constant feature should be standardized
        np.testing.assert_allclose(np.std(X_scaled[:, 1]), 1, atol=1e-10)

    def test_categorical_with_missing_categories(self):
        """Test dummy variable creation with missing categories in new data."""
        # Original data with three categories
        df_train = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A', 'B'],
            'x': [1, 2, 3, 4, 5]
        })

        df_encoded, _, mappings = create_dummy_variables(df_train, ['cat'])

        # New data missing category 'C'
        df_test = pd.DataFrame({
            'cat': ['A', 'B', 'A'],
            'x': [6, 7, 8]
        })

        df_test_encoded, _, _ = create_dummy_variables(df_test, ['cat'], mappings)

        # Should have same dummy columns as training, but with zeros for missing category
        expected_columns = df_encoded.columns
        actual_columns = df_test_encoded.columns

        assert set(expected_columns) == set(actual_columns)

    def test_unseen_categories(self):
        """Test handling of unseen categories in test data."""
        # Original data
        df_train = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'B'],
            'x': [1, 2, 3, 4]
        })

        df_encoded, _, mappings = create_dummy_variables(df_train, ['cat'])

        # New data with unseen category 'C'
        df_test = pd.DataFrame({
            'cat': ['A', 'B', 'C'],  # 'C' was not in training
            'x': [5, 6, 7]
        })

        df_test_encoded, _, _ = create_dummy_variables(df_test, ['cat'], mappings)

        # Unseen category should be treated as reference category (all zeros)
        # Check that we have the expected structure
        assert len(df_test_encoded) == 3
        assert 'cat_B' in df_test_encoded.columns

        # Row with unseen category 'C' should have zeros for all dummy variables
        c_row = df_test[df_test['cat'] == 'C'].index[0]
        dummy_cols = [col for col in df_test_encoded.columns if col.startswith('cat_')]
        for col in dummy_cols:
            assert df_test_encoded.loc[c_row, col] == 0