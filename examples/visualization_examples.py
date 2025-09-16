"""
Comprehensive examples demonstrating the visualization module capabilities.

This script showcases all major visualization features including:
- Theme system and styling
- Statistical analysis plots
- Interactive visualizations
- Automated data profiling
- Model evaluation dashboards
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analytics_toolkit.visualization import (
    ClassificationPlots,
    CorrelationPlots,
    DataProfiler,
    DistributionPlots,
    InteractivePlotter,
    ModelEvaluationPlots,
    RegressionPlots,
    StatisticalPlots,
    apply_theme,
    generate_profile_report,
    get_theme_colors,
)
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demonstrate_themes():
    """Demonstrate different theme options."""
    print("=== Theme System Demonstration ===")

    # Create sample data
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = 2 * x + np.random.normal(0, 0.5, 1000)

    themes = ["default", "minimal", "dark"]

    for theme_name in themes:
        print(f"\nApplying {theme_name} theme...")

        # Apply theme
        apply_theme(theme_name)

        # Create a simple plot to show theme
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.6, s=30)
        ax.set_title(f"Scatter Plot with {theme_name.capitalize()} Theme")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")

        # Save or display
        plt.savefig(f"theme_{theme_name}_example.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Show color palette
        colors = get_theme_colors(theme_name, n_colors=7)
        print(f"Color palette for {theme_name}: {colors[:3]}...")  # Show first 3 colors


def demonstrate_statistical_plots():
    """Demonstrate statistical visualization capabilities."""
    print("\n=== Statistical Plots Demonstration ===")

    # Create diverse sample data
    np.random.seed(42)
    n_samples = 1500

    data = pd.DataFrame(
        {
            "normal_dist": np.random.normal(50, 15, n_samples),
            "exponential_dist": np.random.exponential(2, n_samples),
            "uniform_dist": np.random.uniform(0, 100, n_samples),
            "categorical": np.random.choice(
                ["Category A", "Category B", "Category C", "Category D"],
                n_samples,
                p=[0.4, 0.3, 0.2, 0.1],
            ),
            "correlated_var": None,  # Will be filled below
            "target": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        }
    )

    # Create correlated variable
    data["correlated_var"] = 0.7 * data["normal_dist"] + 0.3 * np.random.normal(
        0, 10, n_samples
    )

    # Initialize statistical plotter
    plotter = StatisticalPlots(theme="default")

    print("\n1. Distribution Analysis")
    # Single variable distribution
    dist_fig = plotter.plot_distribution(
        data["normal_dist"],
        column="Normal Distribution",
        bins=40,
        kde=True,
        normal_overlay=True,
        title="Comprehensive Distribution Analysis",
    )
    plt.savefig("distribution_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(dist_fig)

    print("   - Created distribution analysis plot")

    print("\n2. Correlation Matrix")
    # Correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])
    corr_fig = plotter.plot_correlation_matrix(
        numeric_data, method="pearson", title="Feature Correlation Matrix"
    )
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(corr_fig)

    print("   - Created correlation matrix heatmap")

    print("\n3. Pairwise Relationships")
    # Pairwise relationships
    pairwise_fig = plotter.plot_pairwise_relationships(
        numeric_data[["normal_dist", "correlated_var", "uniform_dist"]],
        hue=None,
        kind="scatter",
    )
    plt.savefig("pairwise_relationships.png", dpi=150, bbox_inches="tight")
    plt.close(pairwise_fig)

    print("   - Created pairwise relationship plots")

    print("\n4. Distribution Comparison")
    # Compare distributions
    dist_plotter = DistributionPlots(theme="minimal")
    group_a = data[data["categorical"] == "Category A"]["normal_dist"]
    group_b = data[data["categorical"] == "Category B"]["normal_dist"]

    comparison_fig = dist_plotter.compare_distributions(
        group_a, group_b, labels=("Category A", "Category B"), statistical_test=True
    )
    plt.savefig("distribution_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(comparison_fig)

    print("   - Created distribution comparison with statistical tests")

    print("\n5. Correlation Network")
    # Correlation network
    corr_plotter = CorrelationPlots()
    corr_fig = corr_plotter.plot_correlation_network(numeric_data, threshold=0.3)
    plt.savefig("correlation_network.png", dpi=150, bbox_inches="tight")
    plt.close(corr_fig)

    print("   - Created correlation network visualization")


def demonstrate_interactive_plots():
    """Demonstrate interactive plotting capabilities."""
    print("\n=== Interactive Plots Demonstration ===")

    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "x": np.random.normal(0, 1, 500),
            "y": np.random.normal(0, 1, 500),
            "category": np.random.choice(["Group 1", "Group 2", "Group 3"], 500),
            "size_var": np.random.uniform(5, 25, 500),
            "color_var": np.random.uniform(0, 100, 500),
        }
    )

    # Add some correlation
    data["y"] += 0.5 * data["x"] + np.random.normal(0, 0.5, 500)

    try:
        print("\n1. Interactive Scatter Plot (Plotly)")
        plotter = InteractivePlotter(backend="plotly")

        if plotter.backend.available:
            # Interactive scatter plot
            scatter_fig = plotter.scatter_plot(
                data,
                "x",
                "y",
                color="color_var",
                size="size_var",
                title="Interactive Scatter Plot with Color and Size Encoding",
            )

            # Save as HTML (if Plotly available)
            scatter_fig.write_html("interactive_scatter.html")
            print("   - Created interactive scatter plot (saved as HTML)")

            # Interactive histogram
            hist_fig = plotter.histogram(
                data, "x", bins=30, title="Interactive Histogram"
            )
            hist_fig.write_html("interactive_histogram.html")
            print("   - Created interactive histogram")

            # Correlation heatmap
            numeric_data = data.select_dtypes(include=[np.number])
            heatmap_fig = plotter.correlation_heatmap(
                numeric_data, title="Interactive Correlation Heatmap"
            )
            heatmap_fig.write_html("interactive_correlation.html")
            print("   - Created interactive correlation heatmap")

        else:
            print("   - Plotly not available, skipping interactive plots")

    except Exception as e:
        print(f"   - Interactive plots not available: {e}")

    try:
        print("\n2. Interactive Plots (Bokeh)")
        bokeh_plotter = InteractivePlotter(backend="bokeh")

        if bokeh_plotter.backend.available:
            bokeh_plotter.scatter_plot(data, "x", "y", title="Bokeh Scatter Plot")
            print("   - Created Bokeh scatter plot")
        else:
            print("   - Bokeh not available")

    except Exception as e:
        print(f"   - Bokeh plots not available: {e}")


def demonstrate_data_profiling():
    """Demonstrate automated data profiling."""
    print("\n=== Data Profiling Demonstration ===")

    # Create complex dataset for profiling
    np.random.seed(42)
    n_samples = 2000

    # Create realistic dataset with various data types and issues
    data = pd.DataFrame(
        {
            # Numeric variables
            "revenue": np.random.lognormal(10, 1, n_samples),
            "age": np.random.normal(35, 12, n_samples).clip(18, 80),
            "score": np.random.beta(2, 5, n_samples) * 100,
            # Categorical variables
            "department": np.random.choice(
                ["Sales", "Marketing", "Engineering", "HR", "Finance"],
                n_samples,
                p=[0.3, 0.2, 0.25, 0.1, 0.15],
            ),
            "region": np.random.choice(["North", "South", "East", "West"], n_samples),
            "performance": np.random.choice(
                ["Excellent", "Good", "Average", "Poor"],
                n_samples,
                p=[0.2, 0.4, 0.3, 0.1],
            ),
            # Variable with missing data
            "optional_field": np.random.choice(
                [1, 2, 3, np.nan], n_samples, p=[0.25, 0.25, 0.25, 0.25]
            ),
            # Boolean variable
            "is_active": np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
            # Date variable
            "join_date": pd.date_range("2020-01-01", periods=n_samples, freq="6H"),
            # String variable with varying lengths
            "comments": [
                "Short",
                "Medium length comment",
                "Very long comment with lots of detail",
            ]
            * (n_samples // 3)
            + ["Short"],
        }
    )

    # Add some duplicates
    data = pd.concat([data, data.iloc[:50]], ignore_index=True)

    print(f"\nCreated dataset with {len(data)} rows and {len(data.columns)} columns")

    # Generate comprehensive profile
    profiler = DataProfiler(theme="default")

    print("\n1. Basic Profiling (without plots)")
    profile = profiler.generate_profile(data, include_plots=False)

    # Display key statistics
    print("   Dataset Info:")
    print(f"   - Rows: {profile.dataset_info['n_rows']:,}")
    print(f"   - Columns: {profile.dataset_info['n_columns']}")
    print(
        f"   - Memory Usage: {profile.dataset_info['memory_usage'] / 1024 / 1024:.2f} MB"
    )
    print(f"   - Missing Values: {profile.missing_data['total_missing']:,}")
    print(f"   - Duplicate Rows: {profile.duplicates['total_duplicates']:,}")

    # Show column profiles for a few columns
    print("\n   Sample Column Profiles:")
    for col in ["revenue", "department", "optional_field"]:
        if col in profile.column_profiles:
            col_profile = profile.column_profiles[col]
            print(f"   - {col}:")
            print(f"     Type: {col_profile['dtype']}")
            print(f"     Missing: {col_profile['null_percentage']:.1f}%")
            print(f"     Unique: {col_profile['unique_count']:,}")
            if "mean" in col_profile:
                print(f"     Mean: {col_profile['mean']:.2f}")

    print("\n2. Full Profiling (with visualizations)")
    profile_with_plots = profiler.generate_profile(
        data.drop(["join_date"], axis=1),  # Remove datetime for simpler visualization
        target_column="performance",
        include_plots=True,
    )

    # Save generated plots
    plot_names = {
        "overview": "profile_overview.png",
        "missing_data": "profile_missing_data.png",
        "correlations": "profile_correlations.png",
        "numeric_distributions": "profile_numeric_distributions.png",
        "categorical_distributions": "profile_categorical_distributions.png",
        "target_analysis": "profile_target_analysis.png",
    }

    for plot_key, filename in plot_names.items():
        if plot_key in profile_with_plots.figures:
            profile_with_plots.figures[plot_key].savefig(
                filename, dpi=150, bbox_inches="tight"
            )
            plt.close(profile_with_plots.figures[plot_key])
            print(f"   - Saved {plot_key} plot")

    print("\n3. Convenience Function")
    # Test convenience function
    generate_profile_report(
        data[["revenue", "age", "department", "performance"]],
        target_column="performance",
        include_plots=False,
    )
    print("   - Generated quick profile using convenience function")


def demonstrate_model_evaluation():
    """Demonstrate model evaluation visualizations."""
    print("\n=== Model Evaluation Demonstration ===")

    # Classification Example
    print("\n1. Classification Model Evaluation")

    # Create classification dataset
    X_cls, y_cls = make_classification(
        n_samples=2000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_cls, y_train_cls)

    y_pred_cls = clf.predict(X_test_cls)
    y_proba_cls = clf.predict_proba(X_test_cls)

    # Initialize evaluation plotter
    eval_plotter = ClassificationPlots(theme="default")

    # Confusion Matrix
    confusion_matrix_fig = eval_plotter.plot_confusion_matrix(
        y_test_cls,
        y_pred_cls,
        class_names=["Class 0", "Class 1", "Class 2"],
        title="Multi-class Confusion Matrix",
    )
    plt.savefig("classification_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(confusion_matrix_fig)
    print("   - Created confusion matrix")

    # ROC Curves
    roc_curves_fig = eval_plotter.plot_roc_curve(
        y_test_cls,
        y_proba_cls,
        class_names=["Class 0", "Class 1", "Class 2"],
        title="Multi-class ROC Curves",
    )
    plt.savefig("classification_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(roc_curves_fig)
    print("   - Created ROC curves")

    # Precision-Recall Curves
    pr_curves_fig = eval_plotter.plot_precision_recall_curve(
        y_test_cls,
        y_proba_cls,
        class_names=["Class 0", "Class 1", "Class 2"],
        title="Multi-class Precision-Recall Curves",
    )
    plt.savefig("classification_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(pr_curves_fig)
    print("   - Created precision-recall curves")

    # Classification Report Heatmap
    report_fig = eval_plotter.plot_classification_report_heatmap(
        y_test_cls, y_pred_cls, class_names=["Class 0", "Class 1", "Class 2"]
    )
    plt.savefig("classification_report_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(report_fig)
    print("   - Created classification report heatmap")

    # Learning Curve
    learning_fig = eval_plotter.plot_learning_curve(
        clf, X_train_cls, y_train_cls, cv=5, scoring="accuracy"
    )
    plt.savefig("classification_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(learning_fig)
    print("   - Created learning curve")

    # Class Distribution
    class_dist_fig = eval_plotter.plot_class_distribution(
        y_test_cls, y_pred_cls, class_names=["Class 0", "Class 1", "Class 2"]
    )
    plt.savefig("classification_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(class_dist_fig)
    print("   - Created class distribution comparison")

    # Regression Example
    print("\n2. Regression Model Evaluation")

    # Create regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1500, n_features=15, noise=0.1, random_state=42
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    # Train regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train_reg, y_train_reg)

    y_pred_reg = regressor.predict(X_test_reg)

    # Initialize regression plotter
    reg_plotter = RegressionPlots(theme="minimal")

    # Predictions vs Actual
    pred_fig = reg_plotter.plot_predictions_vs_actual(
        y_test_reg, y_pred_reg, title="Regression: Predictions vs Actual Values"
    )
    plt.savefig("regression_predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(pred_fig)
    print("   - Created predictions vs actual plot")

    # Residual Analysis
    residual_fig = reg_plotter.plot_residuals(
        y_test_reg, y_pred_reg, title="Regression: Residual Analysis"
    )
    plt.savefig("regression_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(residual_fig)
    print("   - Created residual analysis")

    # Feature Importance
    feature_names = [f"Feature_{i+1}" for i in range(X_reg.shape[1])]
    importance_fig = reg_plotter.plot_feature_importance(
        regressor.feature_importances_, feature_names, top_n=10
    )
    plt.savefig("regression_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(importance_fig)
    print("   - Created feature importance plot")

    # Validation Curve
    reg_eval_plotter = ModelEvaluationPlots()
    validation_fig = reg_eval_plotter.plot_validation_curve(
        regressor,
        X_train_reg[:300],
        y_train_reg[:300],  # Subset for faster computation
        param_name="n_estimators",
        param_range=[10, 25, 50, 75, 100],
        cv=3,
        scoring="r2",
    )
    plt.savefig("regression_validation_curve.png", dpi=150, bbox_inches="tight")
    plt.close(validation_fig)
    print("   - Created validation curve")


def demonstrate_real_world_example():
    """Demonstrate a complete real-world analysis workflow."""
    print("\n=== Real-World Example: Iris Dataset Analysis ===")

    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["species"] = [iris.target_names[i] for i in iris.target]
    iris_df["target"] = iris.target

    print(
        f"\nAnalyzing Iris dataset: {len(iris_df)} samples, {len(iris_df.columns)} features"
    )

    # 1. Data Profiling
    print("\n1. Automated Data Profiling")
    apply_theme("default")

    profiler = DataProfiler()
    profile = profiler.generate_profile(
        iris_df.drop(["target"], axis=1),  # Don't include numeric target in profiling
        target_column="species",
        include_plots=True,
    )

    # Save profiling results
    for plot_name, fig in profile.figures.items():
        fig.savefig(f"iris_{plot_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("   - Generated comprehensive data profile")

    # 2. Statistical Analysis
    print("\n2. Statistical Analysis")
    apply_theme("minimal")

    stat_plotter = StatisticalPlots()

    # Correlation analysis
    numeric_features = iris_df.select_dtypes(include=[np.number]).drop(
        ["target"], axis=1
    )
    iris_correlation_fig = stat_plotter.plot_correlation_matrix(
        numeric_features, title="Iris Features Correlation Matrix"
    )
    plt.savefig("iris_correlation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(iris_correlation_fig)

    # Pairwise relationships with species coloring
    iris_pairwise_fig = stat_plotter.plot_pairwise_relationships(
        numeric_features, title="Iris Features Pairwise Relationships"
    )
    plt.savefig("iris_pairwise_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(iris_pairwise_fig)
    print("   - Created statistical analysis plots")

    # 3. Model Training and Evaluation
    print("\n3. Model Training and Evaluation")
    apply_theme("dark")

    # Prepare data
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Evaluation plots
    eval_plotter = ClassificationPlots()

    # Confusion Matrix
    iris_confusion_fig = eval_plotter.plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=iris.target_names,
        title="Iris Classification: Confusion Matrix",
    )
    plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(iris_confusion_fig)

    # ROC Curves
    iris_roc_fig = eval_plotter.plot_roc_curve(
        y_test,
        y_proba,
        class_names=iris.target_names,
        title="Iris Classification: ROC Curves",
    )
    plt.savefig("iris_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(iris_roc_fig)

    # Feature Importance
    reg_plotter = RegressionPlots()
    iris_importance_fig = reg_plotter.plot_feature_importance(
        model.feature_importances_,
        iris.feature_names,
        title="Iris Classification: Feature Importance",
    )
    plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(iris_importance_fig)
    print("   - Created model evaluation plots")

    # 4. Interactive Analysis (if available)
    print("\n4. Interactive Analysis")
    try:
        interactive_plotter = InteractivePlotter(backend="plotly")
        if interactive_plotter.backend.available:
            # Interactive scatter plot
            iris_df["species_numeric"] = iris_df["target"]
            scatter_fig = interactive_plotter.scatter_plot(
                iris_df,
                "sepal length (cm)",
                "sepal width (cm)",
                color="species_numeric",
                title="Interactive Iris Analysis: Sepal Measurements",
            )
            scatter_fig.write_html("iris_interactive_scatter.html")
            print("   - Created interactive scatter plot")
        else:
            print("   - Interactive plots not available (Plotly not installed)")
    except Exception as e:
        print(f"   - Interactive analysis skipped: {e}")

    print("\n   Complete Iris analysis workflow finished!")
    print("   Generated plots showcase the full visualization pipeline")


def main():
    """Run all visualization examples."""
    print("Analytics Toolkit - Visualization Module Examples")
    print("=" * 60)

    try:
        # Set up clean environment
        plt.style.use("default")  # Reset any existing style

        # Run all demonstrations
        demonstrate_themes()
        demonstrate_statistical_plots()
        demonstrate_interactive_plots()
        demonstrate_data_profiling()
        demonstrate_model_evaluation()
        demonstrate_real_world_example()

        print("\n" + "=" * 60)
        print("All visualization examples completed successfully!")
        print("\nGenerated files:")
        print("- Theme examples: theme_*.png")
        print(
            "- Statistical plots: distribution_*.png, correlation_*.png, pairwise_*.png"
        )
        print("- Interactive plots: interactive_*.html (if Plotly available)")
        print("- Data profiling: profile_*.png")
        print("- Model evaluation: classification_*.png, regression_*.png")
        print("- Real-world example: iris_*.png, iris_*.html")
        print("\nThe visualization module provides comprehensive tools for:")
        print("- Professional data visualization with consistent theming")
        print("- Interactive plots for exploratory data analysis")
        print("- Automated data profiling and quality assessment")
        print("- Model evaluation and performance monitoring")
        print("- Statistical analysis and hypothesis testing")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This may be due to missing optional dependencies (Plotly, Bokeh)")
        print("Core matplotlib/seaborn functionality should still work")

    finally:
        plt.close("all")  # Clean up any remaining plots


if __name__ == "__main__":
    main()
