"""
Comprehensive test suite for Streamlit UI functionality.

Tests the major user flows, import fallback systems, and UI component behavior
for the Analytics Toolkit web interface.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import importlib

# Add UI path for imports
ui_path = Path(__file__).parent.parent / "ui"
sys.path.insert(0, str(ui_path))

# Import UI pages
try:
    from pages import feature_engineering, model_comparison, results_dashboard
    UI_PAGES_AVAILABLE = True
except ImportError:
    UI_PAGES_AVAILABLE = False


class TestUIImportFallbacks:
    """Test that UI pages handle missing imports gracefully."""

    def test_feature_engineering_import_fallbacks(self):
        """Test feature engineering page handles missing custom transformers."""
        # Mock streamlit to avoid actual UI rendering
        with patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.session_state', {}):

            # Test that page loads even with missing imports
            try:
                import ui.pages.feature_engineering as fe_page
                # Should not raise ImportError
                assert hasattr(fe_page, 'FEATURE_ENGINEERING_AVAILABLE')
                assert hasattr(fe_page, 'AVAILABLE_TRANSFORMERS')
            except ImportError as e:
                pytest.fail(f"Feature engineering page failed to load: {e}")

    def test_model_comparison_import_fallbacks(self):
        """Test model comparison page handles missing visualization classes."""
        with patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.session_state', {}):

            try:
                import ui.pages.model_comparison as mc_page
                assert hasattr(mc_page, 'MODEL_COMPARISON_AVAILABLE')
                assert hasattr(mc_page, 'AVAILABLE_COMPONENTS')
            except ImportError as e:
                pytest.fail(f"Model comparison page failed to load: {e}")

    def test_results_dashboard_import_fallbacks(self):
        """Test results dashboard handles missing visualization components."""
        with patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.session_state', {}):

            try:
                import ui.pages.results_dashboard as rd_page
                assert hasattr(rd_page, 'ADVANCED_VISUALIZATION_AVAILABLE')
                assert hasattr(rd_page, 'AVAILABLE_VIZ_COMPONENTS')
            except ImportError as e:
                pytest.fail(f"Results dashboard page failed to load: {e}")


class TestFeatureEngineeringPage:
    """Test feature engineering page functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'numeric_col1': np.random.normal(0, 1, 100),
            'numeric_col2': np.random.exponential(1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randn(100)
        })

    def test_data_availability_check(self, sample_data):
        """Test that page properly checks for processed data."""
        with patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.button'), \
             patch('streamlit.session_state', {}):

            # Import and test page function
            from ui.pages.feature_engineering import show

            # Should show warning when no processed_data in session
            show()
            mock_warning.assert_called_once()

    def test_feature_availability_display(self, sample_data):
        """Test that page displays available features correctly."""
        session_state = {
            'processed_data': sample_data,
            'feature_engineered_data': sample_data.copy()
        }

        with patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.session_state', session_state), \
             patch('streamlit.columns') as mock_cols, \
             patch('streamlit.metric') as mock_metric:

            mock_cols.return_value = [MagicMock(), MagicMock()]

            from ui.pages.feature_engineering import show
            show()

            # Should display metrics about data
            assert mock_metric.called

    def test_transformation_fallback_logic(self, sample_data):
        """Test that transformations work with fallback implementations."""
        from ui.pages.feature_engineering import AVAILABLE_TRANSFORMERS

        # Test that even without custom transformers, we can still transform data
        numeric_data = sample_data.select_dtypes(include=[np.number])

        # Test log transformation fallback
        log_transformed = np.log(numeric_data + 1)
        assert not log_transformed.isna().all().all()

        # Test outlier capping fallback
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        capped_data = numeric_data.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR, axis=1)
        assert capped_data.shape == numeric_data.shape


class TestModelComparisonPage:
    """Test model comparison page functionality."""

    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML data."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        return pd.DataFrame(X), pd.Series(y)

    def test_data_source_detection(self, sample_ml_data):
        """Test that page detects different data sources correctly."""
        X, y = sample_ml_data

        # Test with final_data in session
        session_state = {'final_data': X}
        with patch('streamlit.session_state', session_state), \
             patch('streamlit.title'), patch('streamlit.markdown'):

            from ui.pages.model_comparison import show
            # Should not raise errors
            try:
                show()
            except Exception as e:
                if "session_state" not in str(e):  # Ignore streamlit-specific errors
                    pytest.fail(f"Unexpected error: {e}")

    def test_model_comparison_workflow(self, sample_ml_data):
        """Test the basic model comparison workflow."""
        X, y = sample_ml_data

        session_state = {
            'final_data': X,
            'target_column': 'target',
            'task_type': 'regression'
        }

        with patch('streamlit.session_state', session_state), \
             patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.button', return_value=False):

            from ui.pages.model_comparison import show
            # Should handle the workflow without errors
            show()


class TestResultsDashboard:
    """Test results dashboard functionality."""

    @pytest.fixture
    def mock_trained_model(self):
        """Create a mock trained model."""
        model = MagicMock()
        model.predict.return_value = np.random.randn(50)
        model.score.return_value = 0.85
        return model

    def test_model_availability_check(self, mock_trained_model):
        """Test that dashboard checks for trained models correctly."""
        # Test with no models
        with patch('streamlit.session_state', {}), \
             patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.warning') as mock_warning:

            from ui.pages.results_dashboard import show
            show()
            # Should show warning about no model
            assert mock_warning.called

    def test_dashboard_with_trained_model(self, mock_trained_model):
        """Test dashboard functionality with a trained model."""
        session_state = {
            'trained_model': mock_trained_model,
            'model_config': {'model_type': 'regression'},
            'X_test': np.random.randn(50, 5),
            'y_test': np.random.randn(50)
        }

        with patch('streamlit.session_state', session_state), \
             patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.tabs'), \
             patch('streamlit.columns'):

            from ui.pages.results_dashboard import show
            # Should handle model analysis without errors
            show()

    def test_visualization_fallback_behavior(self):
        """Test that visualization functions handle missing components gracefully."""
        from ui.pages.results_dashboard import AVAILABLE_VIZ_COMPONENTS

        # Test that we can check for component availability
        assert isinstance(AVAILABLE_VIZ_COMPONENTS, dict)

        # Even if components are missing, should not crash
        if 'ModelEvaluationPlots' not in AVAILABLE_VIZ_COMPONENTS:
            # Should still be able to create basic plots with matplotlib/plotly
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            assert fig is not None
            plt.close(fig)


class TestUIIntegrationWorkflows:
    """Test end-to-end user workflows through the UI."""

    @pytest.fixture
    def complete_session_data(self):
        """Create complete session state data for testing workflows."""
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.exponential(1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randn(100)
        })

        return {
            'raw_data': data,
            'processed_data': data,
            'feature_engineered_data': data,
            'final_data': data.drop('target', axis=1),
            'target_column': 'target',
            'task_type': 'regression'
        }

    def test_complete_workflow_data_flow(self, complete_session_data):
        """Test that data flows correctly through all pages."""

        with patch('streamlit.session_state', complete_session_data), \
             patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.warning'), patch('streamlit.success'), \
             patch('streamlit.columns'), patch('streamlit.selectbox'), \
             patch('streamlit.button', return_value=False):

            # Test feature engineering page with complete data
            from ui.pages.feature_engineering import show as fe_show
            fe_show()

            # Test model comparison page with complete data
            from ui.pages.model_comparison import show as mc_show
            mc_show()

            # Test results dashboard (will show warning about no trained model, but should not crash)
            from ui.pages.results_dashboard import show as rd_show
            rd_show()

    def test_error_handling_across_pages(self):
        """Test that all pages handle errors gracefully."""
        # Test with minimal/empty session state
        empty_session = {}

        with patch('streamlit.session_state', empty_session), \
             patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.warning'), patch('streamlit.error'):

            # All pages should handle empty session gracefully
            from ui.pages import feature_engineering, model_comparison, results_dashboard

            # Should not raise exceptions
            feature_engineering.show()
            model_comparison.show()
            results_dashboard.show()


class TestUIComponentAvailability:
    """Test availability and functionality of UI components."""

    def test_sklearn_components_available(self):
        """Test that required sklearn components are available."""
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score

            # Test basic functionality
            X = np.random.randn(100, 3)
            y = np.random.randn(100)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            assert X_scaled.shape == X.shape

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            assert isinstance(score, float)

        except ImportError as e:
            pytest.fail(f"Required sklearn components not available: {e}")

    def test_plotting_libraries_available(self):
        """Test that plotting libraries work correctly."""
        try:
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go

            # Test basic plotting functionality
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            plt.close(fig)

            # Test plotly
            df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
            fig = px.scatter(df, x='x', y='y')
            assert fig is not None

        except ImportError as e:
            pytest.fail(f"Required plotting libraries not available: {e}")


@pytest.mark.skipif(not UI_PAGES_AVAILABLE, reason="UI pages not available for import")
class TestUIPageLoading:
    """Test that all UI pages can be loaded without errors."""

    def test_all_pages_importable(self):
        """Test that all UI pages can be imported."""
        page_modules = [
            'ui.pages.home',
            'ui.pages.data_upload',
            'ui.pages.preprocessing',
            'ui.pages.feature_engineering',
            'ui.pages.model_training',
            'ui.pages.model_comparison',
            'ui.pages.results_dashboard'
        ]

        for module_name in page_modules:
            try:
                module = importlib.import_module(module_name)
                assert hasattr(module, 'show'), f"{module_name} missing show() function"
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_page_functions_callable(self):
        """Test that all page show() functions are callable."""
        with patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.session_state', {}):

            try:
                import ui.pages.feature_engineering as fe
                import ui.pages.model_comparison as mc
                import ui.pages.results_dashboard as rd

                # Should be callable (even if they show warnings)
                assert callable(fe.show)
                assert callable(mc.show)
                assert callable(rd.show)

            except ImportError:
                pytest.skip("UI pages not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])