"""
Model comparison page for the Analytics Toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

# Add src to path for imports
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import components with fallbacks
MODEL_COMPARISON_AVAILABLE = True
AVAILABLE_COMPONENTS = {}

try:
    from analytics_toolkit.automl import AutoMLPipeline
    AVAILABLE_COMPONENTS['AutoMLPipeline'] = AutoMLPipeline
except ImportError:
    pass

try:
    from analytics_toolkit.pytorch_regression import LinearRegression, LogisticRegression
    AVAILABLE_COMPONENTS['LinearRegression'] = LinearRegression
    AVAILABLE_COMPONENTS['LogisticRegression'] = LogisticRegression
except ImportError:
    pass

try:
    from analytics_toolkit.visualization import ModelEvaluationPlots
    AVAILABLE_COMPONENTS['ModelEvaluationPlots'] = ModelEvaluationPlots
except ImportError:
    try:
        # Try alternative import names
        from analytics_toolkit.visualization import ModelEvaluationPlotter
        AVAILABLE_COMPONENTS['ModelEvaluationPlots'] = ModelEvaluationPlotter
    except ImportError:
        pass

# Import sklearn components (should always be available)
try:
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR, SVC
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
except ImportError as e:
    MODEL_COMPARISON_AVAILABLE = False
    st.error(f"Sklearn dependencies not available: {e}")

def show():
    """Display the model comparison page."""

    st.title("⚖️ Model Comparison")
    st.markdown("Compare multiple models side-by-side with comprehensive analysis")

    if not MODEL_COMPARISON_AVAILABLE:
        st.error("❌ Model comparison modules not available. Please check your installation.")
        return

    # Check if we have data
    data_available = False
    if 'final_data' in st.session_state:
        data = st.session_state.final_data
        data_available = True
    elif 'feature_engineered_data' in st.session_state:
        data = st.session_state.feature_engineered_data
        data_available = True
    elif 'processed_data' in st.session_state:
        data = st.session_state.processed_data
        data_available = True

    if not data_available:
        st.warning("📊 No processed data available. Please complete data preprocessing first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Go to Data Upload"):
                st.switch_page("pages/data_upload.py")
        with col2:
            if st.button("🔧 Go to Preprocessing"):
                st.switch_page("pages/preprocessing.py")
        return

    # Initialize session state for model comparison
    if 'comparison_models' not in st.session_state:
        st.session_state.comparison_models = {}

    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}

    # Main interface
    tab1, tab2, tab3 = st.tabs(["🏗️ Setup Models", "📊 Compare Results", "📈 Advanced Analysis"])

    with tab1:
        setup_model_comparison(data)

    with tab2:
        if st.session_state.comparison_results:
            display_model_comparison()
        else:
            st.info("⚙️ Set up and train models first to see comparisons.")

    with tab3:
        if st.session_state.comparison_results:
            advanced_model_analysis()
        else:
            st.info("⚙️ Train models first to see advanced analysis.")


def setup_model_comparison(data):
    """Set up models for comparison."""

    st.markdown("### 🎯 Problem Setup")

    # Target column selection
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns available for modeling.")
        return

    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox(
            "Select target column:",
            numeric_cols,
            key="comparison_target"
        )

    with col2:
        # Determine problem type
        if target_column:
            unique_values = data[target_column].nunique()
            if unique_values <= 10:
                problem_type = st.selectbox(
                    "Problem type:",
                    ["classification", "regression"],
                    index=0,
                    key="comparison_problem_type"
                )
            else:
                problem_type = st.selectbox(
                    "Problem type:",
                    ["regression", "classification"],
                    index=0,
                    key="comparison_problem_type"
                )

    if not target_column:
        return

    # Feature selection
    feature_cols = [col for col in data.columns if col != target_column]
    selected_features = st.multiselect(
        "Select features (leave empty for all):",
        feature_cols,
        key="comparison_features"
    )

    if not selected_features:
        selected_features = feature_cols

    # Model selection
    st.markdown("### 🤖 Model Selection")

    available_models = get_available_models(problem_type)

    selected_models = st.multiselect(
        "Select models to compare:",
        list(available_models.keys()),
        default=list(available_models.keys())[:3],
        key="comparison_selected_models"
    )

    # Training configuration
    st.markdown("### ⚙️ Training Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Test size:", 0.1, 0.5, 0.2, key="comparison_test_size")

    with col2:
        cv_folds = st.slider("CV folds:", 3, 10, 5, key="comparison_cv_folds")

    with col3:
        random_state = st.number_input("Random state:", value=42, key="comparison_random_state")

    # Train models
    if selected_models and st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models..."):
            train_comparison_models(
                data, target_column, selected_features, selected_models,
                available_models, problem_type, test_size, cv_folds, random_state
            )
        st.success(f"✅ Successfully trained {len(selected_models)} models!")
        st.rerun()


def get_available_models(problem_type: str) -> Dict[str, Any]:
    """Get available models for the problem type."""

    if problem_type == "regression":
        return {
            "PyTorch Linear": LinearRegression,
            "Random Forest": RandomForestRegressor,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "Elastic Net": ElasticNet,
            "SVR": SVR,
            "AutoML": "automl"
        }
    else:  # classification
        return {
            "PyTorch Logistic": LogisticRegression,
            "Random Forest": RandomForestClassifier,
            "SVC": SVC,
            "AutoML": "automl"
        }


def train_comparison_models(data, target_column, feature_cols, selected_models,
                          available_models, problem_type, test_size, cv_folds, random_state):
    """Train all selected models."""

    # Prepare data
    X = data[feature_cols]
    y = data[target_column]

    # Handle missing values (simple imputation)
    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {}
    results = {}

    for model_name in selected_models:
        try:
            st.write(f"Training {model_name}...")

            if model_name == "AutoML":
                # Use AutoML pipeline
                model = AutoMLPipeline(
                    task_type=problem_type,
                    time_limit=60,  # 1 minute for demo
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            elif available_models[model_name] in [LinearRegression, LogisticRegression]:
                # PyTorch models
                model = available_models[model_name](
                    fit_intercept=True,
                    compute_stats=True
                )
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)

            else:
                # Sklearn models
                model_class = available_models[model_name]
                if model_name == "Random Forest":
                    model = model_class(n_estimators=100, random_state=random_state)
                elif model_name in ["Ridge", "Lasso", "Elastic Net"]:
                    model = model_class(alpha=1.0, random_state=random_state)
                else:
                    model = model_class(random_state=random_state)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            if problem_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # Cross validation
                cv_scores = cross_val_score(
                    model if hasattr(model, 'fit') else None,
                    X_train, y_train, cv=cv_folds,
                    scoring='r2' if hasattr(model, 'fit') else None
                )

                model_results = {
                    'model': model,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'y_train': y_train,
                    'X_test': X_test,
                    'X_train': X_train,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_scores': cv_scores if hasattr(model, 'fit') else [r2] * cv_folds,
                    'problem_type': problem_type
                }

            else:  # classification
                accuracy = accuracy_score(y_test, y_pred)

                # Cross validation
                cv_scores = cross_val_score(
                    model if hasattr(model, 'fit') else None,
                    X_train, y_train, cv=cv_folds,
                    scoring='accuracy' if hasattr(model, 'fit') else None
                )

                model_results = {
                    'model': model,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'y_train': y_train,
                    'X_test': X_test,
                    'X_train': X_train,
                    'accuracy': accuracy,
                    'cv_scores': cv_scores if hasattr(model, 'fit') else [accuracy] * cv_folds,
                    'problem_type': problem_type
                }

            models[model_name] = model
            results[model_name] = model_results

        except Exception as e:
            st.error(f"❌ Failed to train {model_name}: {str(e)}")
            continue

    # Store results
    st.session_state.comparison_models = models
    st.session_state.comparison_results = results


def display_model_comparison():
    """Display comprehensive model comparison."""

    results = st.session_state.comparison_results

    if not results:
        st.info("No model results available.")
        return

    st.markdown("### 📊 Model Performance Comparison")

    # Create metrics DataFrame
    metrics_data = []
    problem_type = list(results.values())[0]['problem_type']

    for model_name, result in results.items():
        if problem_type == "regression":
            metrics_data.append({
                'Model': model_name,
                'RMSE': f"{result['rmse']:.4f}",
                'R²': f"{result['r2']:.4f}",
                'CV Mean': f"{np.mean(result['cv_scores']):.4f}",
                'CV Std': f"{np.std(result['cv_scores']):.4f}"
            })
        else:
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'CV Mean': f"{np.mean(result['cv_scores']):.4f}",
                'CV Std': f"{np.std(result['cv_scores']):.4f}"
            })

    metrics_df = pd.DataFrame(metrics_data)

    # Display metrics table
    st.dataframe(metrics_df, use_container_width=True)

    # Performance visualization
    col1, col2 = st.columns(2)

    with col1:
        # Main metric comparison
        if problem_type == "regression":
            main_metric = [result['r2'] for result in results.values()]
            metric_name = "R² Score"
        else:
            main_metric = [result['accuracy'] for result in results.values()]
            metric_name = "Accuracy"

        fig = px.bar(
            x=list(results.keys()),
            y=main_metric,
            title=f"{metric_name} Comparison",
            labels={'x': 'Model', 'y': metric_name}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cross-validation scores
        cv_data = []
        for model_name, result in results.items():
            for i, score in enumerate(result['cv_scores']):
                cv_data.append({
                    'Model': model_name,
                    'Fold': i + 1,
                    'Score': score
                })

        cv_df = pd.DataFrame(cv_data)
        fig = px.box(
            cv_df, x='Model', y='Score',
            title="Cross-Validation Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Prediction vs Actual plots
    if problem_type == "regression":
        st.markdown("#### 🎯 Prediction vs Actual Comparison")

        n_models = len(results)
        cols = st.columns(min(n_models, 3))

        for i, (model_name, result) in enumerate(results.items()):
            with cols[i % 3]:
                fig = px.scatter(
                    x=result['y_test'],
                    y=result['predictions'],
                    title=f"{model_name}",
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    trendline="ols"
                )

                # Add perfect prediction line
                min_val = min(min(result['y_test']), min(result['predictions']))
                max_val = max(max(result['y_test']), max(result['predictions']))
                fig.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


def advanced_model_analysis():
    """Display advanced model analysis."""

    results = st.session_state.comparison_results

    st.markdown("### 🔬 Advanced Analysis")

    # Model selection recommendations
    st.markdown("#### 🏆 Model Recommendations")

    problem_type = list(results.values())[0]['problem_type']

    if problem_type == "regression":
        # Rank by R²
        ranked_models = sorted(
            results.items(),
            key=lambda x: x[1]['r2'],
            reverse=True
        )
        best_metric = "R²"
    else:
        # Rank by accuracy
        ranked_models = sorted(
            results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        best_metric = "Accuracy"

    for i, (model_name, result) in enumerate(ranked_models[:3]):
        if i == 0:
            st.success(f"🥇 **Best Model: {model_name}** - {best_metric}: {result.get('r2' if problem_type == 'regression' else 'accuracy', 0):.4f}")
        elif i == 1:
            st.info(f"🥈 **Runner-up: {model_name}** - {best_metric}: {result.get('r2' if problem_type == 'regression' else 'accuracy', 0):.4f}")
        else:
            st.warning(f"🥉 **Third: {model_name}** - {best_metric}: {result.get('r2' if problem_type == 'regression' else 'accuracy', 0):.4f}")

    # Statistical significance testing
    st.markdown("#### 📈 Statistical Analysis")

    with st.expander("Cross-Validation Statistical Tests", expanded=False):
        perform_statistical_tests(results)

    # Feature importance comparison (if available)
    st.markdown("#### 🎯 Feature Importance Analysis")

    with st.expander("Feature Importance Comparison", expanded=False):
        display_feature_importance_comparison(results)

    # Residual analysis for regression
    if problem_type == "regression":
        st.markdown("#### 🔍 Residual Analysis")

        with st.expander("Residual Plots", expanded=False):
            display_residual_analysis(results)


def perform_statistical_tests(results):
    """Perform statistical significance tests."""

    from scipy import stats

    model_names = list(results.keys())
    cv_scores = {name: result['cv_scores'] for name, result in results.items()}

    st.markdown("**Pairwise t-tests (p-values):**")

    # Create pairwise comparison matrix
    n_models = len(model_names)
    p_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            _, p_value = stats.ttest_rel(cv_scores[model_names[i]], cv_scores[model_names[j]])
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value

    # Display as heatmap
    fig = px.imshow(
        p_matrix,
        x=model_names,
        y=model_names,
        color_continuous_scale="RdYlBu_r",
        title="P-values for Pairwise Model Comparisons",
        text_auto=True
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("*Lower p-values (red) indicate statistically significant differences*")


def display_feature_importance_comparison(results):
    """Display feature importance comparison across models."""

    importance_data = []

    for model_name, result in results.items():
        model = result['model']

        try:
            # Try to get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = result['X_train'].columns
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
                feature_names = result['X_train'].columns
            else:
                continue

            for feature, importance in zip(feature_names, importances):
                importance_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                })

        except Exception:
            continue

    if importance_data:
        importance_df = pd.DataFrame(importance_data)

        # Create grouped bar chart
        fig = px.bar(
            importance_df,
            x='Feature',
            y='Importance',
            color='Model',
            title="Feature Importance Comparison",
            barmode='group'
        )

        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for the selected models.")


def display_residual_analysis(results):
    """Display residual analysis for regression models."""

    for model_name, result in results.items():
        st.markdown(f"**{model_name} Residuals:**")

        residuals = result['y_test'] - result['predictions']

        col1, col2 = st.columns(2)

        with col1:
            # Residual vs Predicted
            fig = px.scatter(
                x=result['predictions'],
                y=residuals,
                title=f"Residuals vs Predicted - {model_name}",
                labels={'x': 'Predicted Values', 'y': 'Residuals'}
            )

            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="red")

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Residual histogram
            fig = px.histogram(
                residuals,
                title=f"Residual Distribution - {model_name}",
                labels={'value': 'Residuals', 'count': 'Frequency'}
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")