"""
Comprehensive test suite for the visualization module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from analytics_toolkit.visualization import (
    BokehBackend,
    ClassificationPlots,
    CorrelationPlots,
    DarkTheme,
    # Data Profiling
    DataProfiler,
    # Themes
    DefaultTheme,
    DistributionPlots,
    # Interactive Plots
    InteractivePlotter,
    MinimalTheme,
    # Model Evaluation
    ModelEvaluationPlots,
    PlotlyBackend,
    RegressionPlots,
    # Statistical Plots
    StatisticalPlots,
    apply_theme,
    generate_profile_report,
    get_theme_colors,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_distribution,
    plot_pairwise_relationships,
    plot_roc_curve,
)


class TestThemes:
    """Test theme system."""

    def test_default_theme_creation(self):
        """Test default theme creation."""
        theme = DefaultTheme()
        assert theme.name == "default"
        assert "primary" in theme.colors
        assert theme.font_family == "Arial"
        assert theme.font_size == 10

    def test_minimal_theme_creation(self):
        """Test minimal theme creation."""
        theme = MinimalTheme()
        assert theme.name == "minimal"
        assert "primary" in theme.colors
        assert theme.font_family == "Helvetica"

    def test_dark_theme_creation(self):
        """Test dark theme creation."""
        theme = DarkTheme()
        assert theme.name == "dark"
        assert theme.figure_facecolor == "#2C3E50"
        assert theme.text_color == "#ECF0F1"

    def test_apply_theme_by_name(self):
        """Test applying theme by name."""
        theme = apply_theme("minimal")
        assert isinstance(theme, MinimalTheme)

        theme = apply_theme("dark")
        assert isinstance(theme, DarkTheme)

    def test_apply_theme_by_instance(self):
        """Test applying theme by instance."""
        original_theme = DarkTheme()
        applied_theme = apply_theme(original_theme)
        assert applied_theme is original_theme

    def test_get_theme_colors(self):
        """Test getting theme colors."""
        colors = get_theme_colors("default", n_colors=5)
        assert len(colors) == 5
        assert all(isinstance(color, str) for color in colors)

    def test_color_palette_generation(self):
        """Test color palette generation."""
        theme = DefaultTheme()
        palette = theme.get_color_palette(10)
        assert len(palette) == 10

        palette = theme.get_color_palette(3)
        assert len(palette) == 3

    def test_invalid_theme_name(self):
        """Test handling of invalid theme name."""
        with pytest.raises(ValueError):
            apply_theme("nonexistent_theme")


class TestStatisticalPlots:
    """Test statistical visualization components."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "numeric1": np.random.normal(0, 1, 1000),
                "numeric2": np.random.exponential(1, 1000),
                "categorical": np.random.choice(["A", "B", "C"], 1000),
                "binary": np.random.choice([0, 1], 1000),
            }
        )
        return data

    def test_statistical_plots_initialization(self):
        """Test StatisticalPlots initialization."""
        plotter = StatisticalPlots()
        assert hasattr(plotter, "theme")

        plotter = StatisticalPlots(theme="minimal")
        assert plotter.theme.name == "minimal"

    def test_plot_distribution(self, sample_data):
        """Test distribution plotting."""
        plotter = StatisticalPlots()
        fig = plotter.plot_distribution(sample_data["numeric1"])

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplots
        plt.close(fig)

    def test_plot_distribution_with_options(self, sample_data):
        """Test distribution plotting with options."""
        plotter = StatisticalPlots()
        fig = plotter.plot_distribution(
            sample_data["numeric1"],
            column="test_column",
            bins=20,
            kde=True,
            normal_overlay=True,
            title="Test Distribution",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_matrix(self, sample_data):
        """Test correlation matrix plotting."""
        plotter = StatisticalPlots()
        numeric_data = sample_data[["numeric1", "numeric2", "binary"]]
        fig = plotter.plot_correlation_matrix(numeric_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pairwise_relationships(self, sample_data):
        """Test pairwise relationships plotting."""
        plotter = StatisticalPlots()
        numeric_data = sample_data[["numeric1", "numeric2", "binary"]]
        fig = plotter.plot_pairwise_relationships(numeric_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_convenience_functions(self, sample_data):
        """Test convenience functions."""
        fig1 = plot_distribution(sample_data["numeric1"])
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        numeric_data = sample_data[["numeric1", "numeric2"]]
        fig2 = plot_correlation_matrix(numeric_data)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        fig3 = plot_pairwise_relationships(numeric_data)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)


class TestDistributionPlots:
    """Test distribution analysis plots."""

    def test_compare_distributions(self):
        """Test distribution comparison."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0.5, 1.2, 1000)

        plotter = DistributionPlots()
        fig = plotter.compare_distributions(data1, data2, labels=("Group A", "Group B"))

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_compare_distributions_without_tests(self):
        """Test distribution comparison without statistical tests."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)

        plotter = DistributionPlots()
        fig = plotter.compare_distributions(data1, data2, statistical_test=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCorrelationPlots:
    """Test correlation analysis plots."""

    @pytest.fixture
    def correlation_data(self):
        """Create data with known correlations."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 500)
        x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, 500)  # Strong correlation
        x3 = -0.6 * x1 + 0.4 * np.random.normal(
            0, 1, 500
        )  # Moderate negative correlation
        x4 = np.random.normal(0, 1, 500)  # Independent

        return pd.DataFrame({"var1": x1, "var2": x2, "var3": x3, "var4": x4})

    def test_plot_correlation_network(self, correlation_data):
        """Test correlation network plot."""
        plotter = CorrelationPlots()
        fig = plotter.plot_correlation_network(correlation_data, threshold=0.3)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_network_no_correlations(self):
        """Test correlation network with no strong correlations."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "var1": np.random.normal(0, 1, 100),
                "var2": np.random.normal(0, 1, 100),
                "var3": np.random.normal(0, 1, 100),
            }
        )

        plotter = CorrelationPlots()
        fig = plotter.plot_correlation_network(data, threshold=0.8)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestInteractivePlots:
    """Test interactive plotting capabilities."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 100),
                "y": np.random.normal(0, 1, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "size": np.random.uniform(5, 20, 100),
            }
        )

    def test_plotly_backend_availability(self):
        """Test Plotly backend availability."""
        backend = PlotlyBackend()
        # Should not raise error regardless of availability
        assert hasattr(backend, "available")

    def test_bokeh_backend_availability(self):
        """Test Bokeh backend availability."""
        backend = BokehBackend()
        # Should not raise error regardless of availability
        assert hasattr(backend, "available")

    def test_interactive_plotter_initialization(self):
        """Test InteractivePlotter initialization."""
        plotter = InteractivePlotter(backend="plotly")
        assert plotter.backend_name == "plotly"

        with pytest.raises(ValueError):
            InteractivePlotter(backend="invalid_backend")

    @pytest.mark.skipif(
        True, reason="Plotly/Bokeh may not be available in test environment"
    )
    def test_scatter_plot(self, sample_data):
        """Test interactive scatter plot creation."""
        plotter = InteractivePlotter(backend="plotly")
        if plotter.backend.available:
            plot = plotter.scatter_plot(sample_data, "x", "y", color="category")
            assert plot is not None

    @pytest.mark.skipif(
        True, reason="Plotly/Bokeh may not be available in test environment"
    )
    def test_line_plot(self, sample_data):
        """Test interactive line plot creation."""
        plotter = InteractivePlotter(backend="plotly")
        if plotter.backend.available:
            plot = plotter.line_plot(sample_data.sort_values("x"), "x", "y")
            assert plot is not None

    @pytest.mark.skipif(
        True, reason="Plotly/Bokeh may not be available in test environment"
    )
    def test_histogram(self, sample_data):
        """Test interactive histogram creation."""
        plotter = InteractivePlotter(backend="plotly")
        if plotter.backend.available:
            plot = plotter.histogram(sample_data, "x")
            assert plot is not None


class TestDataProfiler:
    """Test data profiling capabilities."""

    @pytest.fixture
    def complex_data(self):
        """Create complex dataset for profiling."""
        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame(
            {
                "numeric_normal": np.random.normal(50, 15, n_samples),
                "numeric_skewed": np.random.exponential(2, n_samples),
                "categorical_high": np.random.choice(
                    ["cat1", "cat2", "cat3", "cat4"], n_samples
                ),
                "categorical_low": np.random.choice(
                    ["A", "B"], n_samples, p=[0.8, 0.2]
                ),
                "binary": np.random.choice([0, 1], n_samples),
                "with_nulls": np.random.choice(
                    [1, 2, 3, np.nan], n_samples, p=[0.3, 0.3, 0.3, 0.1]
                ),
                "datetime": pd.date_range("2020-01-01", periods=n_samples, freq="H"),
                "string_lengths": ["short", "medium_length", "very_long_string_example"]
                * (n_samples // 3)
                + ["short"],
            }
        )

        return data

    def test_data_profiler_initialization(self):
        """Test DataProfiler initialization."""
        profiler = DataProfiler()
        assert hasattr(profiler, "theme")

        profiler = DataProfiler(theme="dark")
        assert profiler.theme.name == "dark"

    def test_generate_profile_basic(self, complex_data):
        """Test basic profile generation."""
        profiler = DataProfiler()
        profile = profiler.generate_profile(complex_data, include_plots=False)

        assert hasattr(profile, "dataset_info")
        assert hasattr(profile, "column_profiles")
        assert hasattr(profile, "correlations")
        assert hasattr(profile, "missing_data")
        assert hasattr(profile, "duplicates")

        # Check dataset info
        assert profile.dataset_info["n_rows"] == len(complex_data)
        assert profile.dataset_info["n_columns"] == len(complex_data.columns)

    def test_generate_profile_with_plots(self, complex_data):
        """Test profile generation with plots."""
        profiler = DataProfiler()
        profile = profiler.generate_profile(complex_data, include_plots=True)

        assert hasattr(profile, "figures")
        assert isinstance(profile.figures, dict)

        # Clean up figures
        for fig in profile.figures.values():
            if isinstance(fig, plt.Figure):
                plt.close(fig)

    def test_generate_profile_with_target(self, complex_data):
        """Test profile generation with target variable."""
        profiler = DataProfiler()
        profile = profiler.generate_profile(
            complex_data, target_column="binary", include_plots=True
        )

        assert "target_analysis" in profile.figures
        plt.close("all")

    def test_generate_profile_with_sampling(self):
        """Test profile generation with sampling for large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "col1": np.random.normal(0, 1, 5000),
                "col2": np.random.choice(["A", "B", "C"], 5000),
            }
        )

        profiler = DataProfiler()
        with pytest.warns(UserWarning):
            profile = profiler.generate_profile(
                large_data, sample_size=1000, include_plots=False
            )

        assert profile.dataset_info["n_rows"] == 1000  # Should use sample

    def test_convenience_function(self, complex_data):
        """Test convenience function."""
        profile = generate_profile_report(complex_data, include_plots=False)
        assert hasattr(profile, "dataset_info")


class TestModelEvaluationPlots:
    """Test model evaluation visualization."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data and model."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=3,
            n_informative=8,
            n_redundant=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "model": model,
        }

    @pytest.fixture
    def regression_data(self):
        """Create regression data and model."""
        X, y = make_regression(
            n_samples=1000, n_features=10, noise=0.1, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "model": model,
        }

    def test_plot_confusion_matrix(self, classification_data):
        """Test confusion matrix plotting."""
        plotter = ModelEvaluationPlots()
        fig = plotter.plot_confusion_matrix(
            classification_data["y_test"], classification_data["y_pred"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_roc_curve_multiclass(self, classification_data):
        """Test ROC curve for multiclass classification."""
        plotter = ModelEvaluationPlots()
        fig = plotter.plot_roc_curve(
            classification_data["y_test"], classification_data["y_proba"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_precision_recall_curve(self, classification_data):
        """Test precision-recall curve."""
        plotter = ModelEvaluationPlots()
        fig = plotter.plot_precision_recall_curve(
            classification_data["y_test"], classification_data["y_proba"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_learning_curve(self, classification_data):
        """Test learning curve plotting."""
        plotter = ModelEvaluationPlots()
        fig = plotter.plot_learning_curve(
            classification_data["model"],
            classification_data["X_train"],
            classification_data["y_train"],
            cv=3,  # Reduced for faster testing
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_validation_curve(self, classification_data):
        """Test validation curve plotting."""
        plotter = ModelEvaluationPlots()
        fig = plotter.plot_validation_curve(
            classification_data["model"],
            classification_data["X_train"][:100],  # Reduced for faster testing
            classification_data["y_train"][:100],
            param_name="n_estimators",
            param_range=[5, 10, 15],
            cv=3,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestClassificationPlots:
    """Test classification-specific plots."""

    @pytest.fixture
    def binary_classification_data(self):
        """Create binary classification data."""
        X, y = make_classification(
            n_samples=500, n_features=5, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        return {
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": model.predict(X_test),
            "y_proba": model.predict_proba(X_test)[:, 1],
        }

    def test_classification_report_heatmap(self, binary_classification_data):
        """Test classification report heatmap."""
        plotter = ClassificationPlots()
        fig = plotter.plot_classification_report_heatmap(
            binary_classification_data["y_test"], binary_classification_data["y_pred"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_class_distribution(self, binary_classification_data):
        """Test class distribution plotting."""
        plotter = ClassificationPlots()
        fig = plotter.plot_class_distribution(
            binary_classification_data["y_test"], binary_classification_data["y_pred"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_convenience_functions(self, binary_classification_data):
        """Test convenience functions."""
        fig1 = plot_confusion_matrix(
            binary_classification_data["y_test"], binary_classification_data["y_pred"]
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_roc_curve(
            binary_classification_data["y_test"], binary_classification_data["y_proba"]
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)


class TestRegressionPlots:
    """Test regression-specific plots."""

    @pytest.fixture
    def regression_data(self):
        """Create regression data."""
        X, y = make_regression(n_samples=300, n_features=5, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": model.predict(X_test),
            "model": model,
        }

    def test_predictions_vs_actual(self, regression_data):
        """Test predictions vs actual plot."""
        plotter = RegressionPlots()
        fig = plotter.plot_predictions_vs_actual(
            regression_data["y_test"], regression_data["y_pred"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_residuals(self, regression_data):
        """Test residuals plotting."""
        plotter = RegressionPlots()
        fig = plotter.plot_residuals(
            regression_data["y_test"], regression_data["y_pred"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_feature_importance(self, regression_data):
        """Test feature importance plotting."""
        plotter = RegressionPlots()
        feature_names = [
            f"feature_{i}" for i in range(regression_data["X_train"].shape[1])
        ]

        fig = plotter.plot_feature_importance(
            regression_data["model"].feature_importances_, feature_names
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clean up matplotlib plots after each test."""
    yield
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__])
