"""
Advanced Visualization Suite for Analytics Toolkit.

This module provides comprehensive visualization capabilities including:
- Interactive plotting with Plotly and Bokeh integration
- Statistical visualization toolkit with distributions and correlations
- Model performance dashboards and evaluation plots
- Data exploration and profiling visualizations
- Customizable themes and styling options

Key Components:
- interactive: Interactive plotting with Plotly/Bokeh backends
- statistical: Statistical plots, distributions, and hypothesis testing visualizations
- model_evaluation: Model performance metrics, confusion matrices, ROC curves
- data_profiling: Automated data exploration and summary visualizations
- dashboards: Interactive dashboards for comprehensive analysis
- themes: Consistent styling and theming system
"""

from .data_profiling import (
    DataProfiler,
    ProfileReport,
    generate_profile_report,
)
from .interactive import (
    BokehBackend,
    InteractivePlotter,
    PlotlyBackend,
)
from .model_evaluation import (
    ClassificationPlots,
    ModelEvaluationPlots,
    RegressionPlots,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_roc_curve,
)
from .statistical import (
    CorrelationPlots,
    DistributionPlots,
    StatisticalPlots,
    plot_correlation_matrix,
    plot_distribution,
    plot_pairwise_relationships,
)
from .themes import (
    DarkTheme,
    DefaultTheme,
    MinimalTheme,
    PlotTheme,
    apply_theme,
    get_theme_colors,
)

__all__ = [
    # Interactive Plotting
    "InteractivePlotter",
    "PlotlyBackend",
    "BokehBackend",
    # Statistical Plots
    "StatisticalPlots",
    "DistributionPlots",
    "CorrelationPlots",
    "plot_distribution",
    "plot_correlation_matrix",
    "plot_pairwise_relationships",
    # Model Evaluation
    "ModelEvaluationPlots",
    "ClassificationPlots",
    "RegressionPlots",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_learning_curve",
    # Data Profiling
    "DataProfiler",
    "ProfileReport",
    "generate_profile_report",
    # Themes
    "PlotTheme",
    "DefaultTheme",
    "MinimalTheme",
    "DarkTheme",
    "apply_theme",
    "get_theme_colors",
]
