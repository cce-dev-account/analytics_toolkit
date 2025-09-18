"""
Results dashboard page for the Analytics Toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import components with fallbacks
ADVANCED_VISUALIZATION_AVAILABLE = True
AVAILABLE_VIZ_COMPONENTS = {}

try:
    from analytics_toolkit.visualization import ModelEvaluationPlots
    AVAILABLE_VIZ_COMPONENTS['ModelEvaluationPlots'] = ModelEvaluationPlots
except ImportError:
    try:
        from analytics_toolkit.visualization import ModelEvaluationPlotter
        AVAILABLE_VIZ_COMPONENTS['ModelEvaluationPlots'] = ModelEvaluationPlotter
    except ImportError:
        pass

try:
    from analytics_toolkit.visualization import StatisticalPlots
    AVAILABLE_VIZ_COMPONENTS['StatisticalPlots'] = StatisticalPlots
except ImportError:
    try:
        from analytics_toolkit.visualization import StatisticalPlotter
        AVAILABLE_VIZ_COMPONENTS['StatisticalPlots'] = StatisticalPlotter
    except ImportError:
        pass

try:
    from analytics_toolkit.pytorch_regression.stats import (
        compute_model_statistics, format_summary_table
    )
    AVAILABLE_VIZ_COMPONENTS['compute_model_statistics'] = compute_model_statistics
    AVAILABLE_VIZ_COMPONENTS['format_summary_table'] = format_summary_table
except ImportError:
    pass

try:
    from sklearn.inspection import permutation_importance
except ImportError as e:
    ADVANCED_VISUALIZATION_AVAILABLE = False
    st.error(f"Required sklearn components not available: {e}")

def show():
    """Display the results dashboard page."""

    st.title("📈 Results Dashboard")
    st.markdown("Comprehensive model performance analysis and diagnostics")

    # Check if model results are available from different sources
    model_available = False
    model = None
    model_config = {}

    # Check for trained model
    if 'trained_model' in st.session_state:
        model = st.session_state.trained_model
        model_config = st.session_state.get('model_config', {})
        model_available = True

    # Check for comparison results (from model comparison page)
    elif 'comparison_results' in st.session_state and st.session_state.comparison_results:
        st.info("📊 Multiple models available from comparison. Select a model to analyze:")

        model_names = list(st.session_state.comparison_results.keys())
        selected_model = st.selectbox("Choose model for detailed analysis:", model_names)

        if selected_model:
            model_result = st.session_state.comparison_results[selected_model]
            model = model_result['model']

            # Create mock session state for compatibility
            st.session_state.X_train = model_result['X_train']
            st.session_state.X_test = model_result['X_test']
            st.session_state.y_train = model_result['y_train']
            st.session_state.y_test = model_result['y_test']

            model_config = {'model_name': selected_model, 'model_type': model_result['problem_type']}
            model_available = True

    if not model_available:
        st.warning("🧠 No trained model available. Please train a model first.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Go to Model Training"):
                st.switch_page("pages/model_training.py")
        with col2:
            if st.button("⚖️ Go to Model Comparison"):
                st.switch_page("pages/model_comparison.py")
        return

    # Main dashboard with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Diagnostics", "🎯 Feature Analysis",
        "📈 Performance", "🔬 Advanced"
    ])

    with tab1:
        display_model_overview_enhanced(model, model_config)
        display_performance_summary(model)

    with tab2:
        display_diagnostic_plots_enhanced(model)

    with tab3:
        display_feature_analysis_enhanced(model)

    with tab4:
        display_performance_analysis_enhanced(model)

    with tab5:
        display_improvement_suggestions(model)


def display_model_overview_enhanced(model, config):
    """Enhanced model overview with comprehensive information."""

    st.markdown("### 🧠 Model Overview")

    # Basic model information
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_name = model_config.get('model_name', type(model).__name__)
        st.metric("🤖 Model Type", model_name)

    with col2:
        task_type = model_config.get('model_type', 'Unknown').title()
        st.metric("🎯 Task Type", task_type)

    with col3:
        if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None:
            n_features = st.session_state.X_train.shape[1]
        else:
            n_features = "Unknown"
        st.metric("📊 Features", n_features)

    with col4:
        if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None:
            n_samples = st.session_state.X_train.shape[0]
        else:
            n_samples = "Unknown"
        st.metric("📈 Training Samples", n_samples)

    # Model complexity and parameters
    with st.expander("🔧 Model Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Parameters:**")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for key, value in list(params.items())[:8]:  # Show first 8 params
                    st.markdown(f"• **{key}**: {value}")
            else:
                # PyTorch model parameters
                st.markdown(f"• **Fit Intercept**: {getattr(model, 'fit_intercept', 'N/A')}")
                st.markdown(f"• **Penalty**: {getattr(model, 'penalty', 'N/A')}")
                st.markdown(f"• **Alpha**: {getattr(model, 'alpha', 'N/A')}")
                st.markdown(f"• **Device**: {getattr(model, 'device', 'CPU')}")

        with col2:
            st.markdown("**Training Information:**")
            if hasattr(model, 'n_iter_'):
                st.markdown(f"• **Iterations**: {getattr(model, 'n_iter_', 'N/A')}")
            if hasattr(model, 'score'):
                try:
                    train_score = model.score(st.session_state.X_train, st.session_state.y_train)
                    st.markdown(f"• **Training Score**: {train_score:.4f}")
                except (AttributeError, ValueError, KeyError):
                    pass

    # Model summary for PyTorch models
    if hasattr(model, 'summary'):
        with st.expander("📋 Statistical Summary", expanded=False):
            try:
                summary = model.summary()
                st.text(summary)
            except Exception as e:
                st.warning(f"Summary not available: {str(e)}")


def display_performance_summary(model):
    """Display performance summary metrics."""

    st.markdown("### 📊 Performance Summary")

    if not hasattr(st.session_state, 'X_test') or st.session_state.X_test is None:
        st.warning("Test data not available for performance evaluation.")
        return

    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # Determine if regression or classification
    is_regression = pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 10

    try:
        # Get predictions
        y_train_pred = model.predict(st.session_state.X_train)
        y_test_pred = model.predict(st.session_state.X_test)

        if is_regression:
            display_regression_summary(y_train, y_test, y_train_pred, y_test_pred)
        else:
            display_classification_summary(y_train, y_test, y_train_pred, y_test_pred, model)

    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")


def display_regression_summary(y_train, y_test, y_train_pred, y_test_pred):
    """Display regression performance summary."""

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🎯 Test R²", f"{test_r2:.4f}", f"{test_r2 - train_r2:.4f}")

    with col2:
        st.metric("📊 Test RMSE", f"{test_rmse:.4f}", f"{test_rmse - train_rmse:.4f}")

    with col3:
        st.metric("📈 Test MAE", f"{test_mae:.4f}", f"{test_mae - train_mae:.4f}")

    with col4:
        # Bias calculation
        bias = np.mean(y_test_pred - y_test)
        st.metric("⚖️ Bias", f"{bias:.4f}")

    with col5:
        # Model assessment
        if test_r2 > 0.9:
            st.metric("🏆 Assessment", "Excellent", "✅")
        elif test_r2 > 0.8:
            st.metric("🏆 Assessment", "Good", "✅")
        elif test_r2 > 0.6:
            st.metric("🏆 Assessment", "Fair", "⚠️")
        else:
            st.metric("🏆 Assessment", "Poor", "❌")

    # Performance insights
    st.markdown("#### 🔍 Performance Insights")

    insights = []
    overfitting = train_r2 - test_r2

    if overfitting > 0.1:
        insights.append(
            "⚠️ **Overfitting detected**: Training performance significantly exceeds test performance"
        )
    elif overfitting < -0.05:
        insights.append(
            "🔄 **Underfitting possible**: Test performance exceeds training performance"
        )
    else:
        insights.append(
            "✅ **Good generalization**: Training and test performance are well-balanced"
        )

    if abs(bias) > 0.1 * np.std(y_test):
        insights.append(f"📊 **Bias detected**: Model predictions are systematically {'high' if bias > 0 else 'low'}")
    else:
        insights.append("⚖️ **Unbiased predictions**: Model shows minimal systematic bias")

    for insight in insights:
        st.markdown(f"• {insight}")


def display_classification_summary(y_train, y_test, y_train_pred, y_test_pred, model):
    """Display classification performance summary."""

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Handle binary vs multiclass
    average = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'

    precision = precision_score(y_test, y_test_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_test_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)

    # Calculate AUC if binary classification and probabilities available
    auc = None
    if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(st.session_state.X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except (AttributeError, ValueError, IndexError):
            pass

    # Display metrics
    cols = st.columns(6 if auc is not None else 5)

    with cols[0]:
        st.metric("🎯 Accuracy", f"{test_acc:.4f}", f"{test_acc - train_acc:.4f}")

    with cols[1]:
        st.metric("📊 Precision", f"{precision:.4f}")

    with cols[2]:
        st.metric("📈 Recall", f"{recall:.4f}")

    with cols[3]:
        st.metric("⚖️ F1-Score", f"{f1:.4f}")

    with cols[4]:
        # Class balance
        class_balance = len(np.unique(y_test))
        st.metric("🔢 Classes", f"{class_balance}")

    if auc is not None:
        with cols[5]:
            st.metric("📈 AUC-ROC", f"{auc:.4f}")

    # Performance insights
    st.markdown("#### 🔍 Performance Insights")

    insights = []
    overfitting = train_acc - test_acc

    if overfitting > 0.05:
        insights.append("⚠️ **Overfitting detected**: Training accuracy significantly exceeds test accuracy")
    elif test_acc > 0.95:
        insights.append("🏆 **Excellent performance**: Very high accuracy achieved")
    elif test_acc > 0.85:
        insights.append("✅ **Good performance**: High accuracy with room for improvement")
    else:
        insights.append("📊 **Moderate performance**: Consider feature engineering or different algorithms")

    if precision < 0.7:
        insights.append("⚠️ **Low precision**: Many false positives - consider adjusting decision threshold")

    if recall < 0.7:
        insights.append("📈 **Low recall**: Missing many positive cases - consider rebalancing or different metrics")

    for insight in insights:
        st.markdown(f"• {insight}")


def display_diagnostic_plots_enhanced(model):
    """Enhanced diagnostic plots with multiple visualization options."""

    st.markdown("### 🔍 Diagnostic Plots")

    if not hasattr(st.session_state, 'X_test') or st.session_state.X_test is None:
        st.warning("Test data not available for diagnostic plots.")
        return

    y_test = st.session_state.y_test
    is_regression = pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 10

    try:
        y_test_pred = model.predict(st.session_state.X_test)

        if is_regression:
            display_regression_diagnostics_enhanced(y_test, y_test_pred)
        else:
            display_classification_diagnostics_enhanced(y_test, y_test_pred, model)

    except Exception as e:
        st.error(f"Error generating diagnostic plots: {str(e)}")


def display_regression_diagnostics_enhanced(y_true, y_pred):
    """Enhanced regression diagnostic plots."""

    # Main diagnostic plots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Predictions vs Actual', 'Residuals vs Predicted', 'Residual Distribution',
            'Q-Q Plot', 'Residuals vs Order', 'Scale-Location Plot'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    residuals = y_true - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # 1. Predictions vs Actual
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                  marker=dict(color='blue', opacity=0.6, size=6)),
        row=1, col=1
    )
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash'), showlegend=False),
        row=1, col=1
    )

    # 2. Residuals vs Predicted
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                  marker=dict(color='green', opacity=0.6, size=6), showlegend=False),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    # 3. Residual Distribution
    fig.add_trace(
        go.Histogram(x=residuals, name='Residual Dist', opacity=0.7,
                    marker=dict(color='purple'), showlegend=False),
        row=1, col=3
    )

    # 4. Q-Q Plot
    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(standardized_residuals)

    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                  mode='markers', name='Q-Q Plot',
                  marker=dict(color='orange', opacity=0.6, size=6), showlegend=False),
        row=2, col=1
    )
    # Q-Q reference line
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                  mode='lines', line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )

    # 5. Residuals vs Order (to check for patterns)
    order = np.arange(len(residuals))
    fig.add_trace(
        go.Scatter(x=order, y=residuals, mode='markers',
                  marker=dict(color='brown', opacity=0.6, size=6), showlegend=False),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    # 6. Scale-Location Plot (sqrt of standardized residuals vs fitted)
    sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
    fig.add_trace(
        go.Scatter(x=y_pred, y=sqrt_abs_resid, mode='markers',
                  marker=dict(color='teal', opacity=0.6, size=6), showlegend=False),
        row=2, col=3
    )

    fig.update_layout(height=800, title_text="Comprehensive Regression Diagnostics")
    st.plotly_chart(fig, width='stretch')

    # Diagnostic insights
    st.markdown("#### 🔍 Diagnostic Insights")

    insights = []

    # Check residual patterns
    residual_std = np.std(residuals)
    if np.abs(np.mean(residuals)) > 0.1 * residual_std:
        insights.append("⚠️ **Biased predictions**: Residuals have non-zero mean")

    # Check homoscedasticity
    pred_sorted = np.argsort(y_pred)
    first_half_var = np.var(residuals[pred_sorted[:len(residuals)//2]])
    second_half_var = np.var(residuals[pred_sorted[len(residuals)//2:]])
    if max(first_half_var, second_half_var) / min(first_half_var, second_half_var) > 2:
        insights.append("📊 **Heteroscedasticity detected**: Residual variance is not constant")

    # Check normality (simple test)
    _, p_value = stats.normaltest(residuals)
    if p_value < 0.05:
        insights.append("📈 **Non-normal residuals**: Consider data transformation or robust methods")
    else:
        insights.append("✅ **Normally distributed residuals**: Good assumption for statistical inference")

    if len(insights) == 0:
        insights.append("✅ **Good diagnostic results**: No major issues detected")

    for insight in insights:
        st.markdown(f"• {insight}")


def display_classification_diagnostics_enhanced(y_true, y_pred, model):
    """Enhanced classification diagnostic plots."""

    from sklearn.metrics import confusion_matrix, classification_report

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate percentages
        cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        fig = px.imshow(
            cm_percent,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix (%)",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues"
        )

        # Add count annotations
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                        showarrow=False,
                        font=dict(color="white" if cm_percent[i, j] > 50 else "black")
                    )
                )

        fig.update_layout(annotations=annotations)
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Class distribution comparison
        true_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().reindex(true_counts.index, fill_value=0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=true_counts.index, y=true_counts.values,
                            name='Actual Distribution', opacity=0.7))
        fig.add_trace(go.Bar(x=pred_counts.index, y=pred_counts.values,
                            name='Predicted Distribution', opacity=0.7))
        fig.update_layout(title="Class Distribution Comparison", barmode='group')
        st.plotly_chart(fig, width='stretch')

    # ROC Curve and Precision-Recall for binary classification
    if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve

            y_proba = model.predict_proba(st.session_state.X_test)[:, 1]

            col1, col2 = st.columns(2)

            with col1:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {roc_auc:.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                       line=dict(dash='dash'), name='Random'))
                fig.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall, precision)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, name=f'PR (AUC = {pr_auc:.3f})'))
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision'
                )
                st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.warning(f"Could not generate ROC/PR curves: {str(e)}")

    # Classification report
    with st.expander("📋 Detailed Classification Report", expanded=False):
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), width='stretch')


def display_feature_analysis_enhanced(model):
    """Enhanced feature analysis with multiple perspectives."""

    st.markdown("### 🎯 Feature Analysis")

    if not hasattr(st.session_state, 'X_train') or st.session_state.X_train is None:
        st.warning("Training data not available for feature analysis.")
        return

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # Feature importance analysis
    display_feature_importance_detailed(model)

    # Feature correlation analysis
    st.markdown("#### 🔗 Feature Correlations")
    try:
        if hasattr(X_train, 'corr'):
            corr_matrix = X_train.corr()
            if len(corr_matrix) > 0:
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Heatmap"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No correlation data available")
        else:
            st.info("Correlation analysis not available for this data type")
    except Exception as e:
        st.warning(f"Could not display correlation analysis: {str(e)}")

    # Feature distribution analysis
    st.markdown("#### 📊 Feature Distributions")
    try:
        numeric_features = X_train.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5
        if len(numeric_features) > 0:
            for feature in numeric_features:
                fig = px.histogram(
                    x=X_train[feature],
                    title=f"Distribution of {feature}",
                    nbins=30
                )
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No numeric features available for distribution analysis")
    except Exception as e:
        st.warning(f"Could not display feature distributions: {str(e)}")


def display_performance_analysis_enhanced(model):
    """Display enhanced performance analysis."""
    st.markdown("### 📊 Performance Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_name = getattr(model, '__class__', type(model)).__name__
        st.metric("🤖 Model Type", model_name)

    with col2:
        # Try to determine task type from model
        task_type = "Classification" if hasattr(model, 'classes_') else "Regression"
        st.metric("🎯 Task Type", task_type)

    with col3:
        n_features = len(st.session_state.X_train.columns)
        st.metric("📊 Features", n_features)

    with col4:
        device = str(getattr(model, 'device', 'Unknown'))
        st.metric("💻 Device", device.split(':')[0])

    # Model parameters
    with st.expander("🔧 Model Parameters"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Core Parameters:**")
            st.markdown(f"• Fit Intercept: {getattr(model, 'fit_intercept', 'N/A')}")
            st.markdown(f"• Penalty: {getattr(model, 'penalty', 'N/A')}")
            st.markdown(f"• Alpha: {getattr(model, 'alpha', 'N/A')}")

        with col2:
            st.markdown("**Training Parameters:**")
            st.markdown(f"• Max Iterations: {getattr(model, 'max_iter', 'N/A')}")
            st.markdown(f"• Tolerance: {getattr(model, 'tol', 'N/A')}")
            if hasattr(model, 'n_iter_') and model.n_iter_ is not None:
                st.markdown(f"• Converged in: {model.n_iter_} iterations")

def display_performance_metrics():
    """Display comprehensive performance metrics."""

    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")

    model = st.session_state.trained_model
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # Calculate scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Determine if regression or classification
    is_regression = pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 20

    if is_regression:
        display_regression_metrics(model, train_score, test_score)
    else:
        display_classification_metrics(model, train_score, test_score)

def display_regression_metrics(model, train_score, test_score):
    """Display regression-specific metrics."""

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    y_train_pred = model.predict(st.session_state.X_train)
    y_test_pred = model.predict(st.session_state.X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🏋️ Train R²", f"{train_score:.4f}")
        st.metric("🧪 Test R²", f"{test_score:.4f}")

    with col2:
        st.metric("📊 Train MSE", f"{train_mse:.4f}")
        st.metric("📊 Test MSE", f"{test_mse:.4f}")

    with col3:
        st.metric("📈 Train MAE", f"{train_mae:.4f}")
        st.metric("📈 Test MAE", f"{test_mae:.4f}")

    with col4:
        st.metric("🎯 RMSE", f"{np.sqrt(test_mse):.4f}")

        # Overfitting check
        if train_score - test_score > 0.1:
            st.error("⚠️ Overfitting")
        elif test_score > 0.8:
            st.success("✅ Good Fit")
        else:
            st.warning("📊 Moderate Fit")

def display_classification_metrics(model, train_score, test_score):
    """Display classification-specific metrics."""

    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

    y_test = st.session_state.y_test
    y_test_pred = model.predict(st.session_state.X_test)

    # Handle multiclass vs binary
    average = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'

    # Calculate metrics
    precision = precision_score(y_test, y_test_pred, average=average)
    recall = recall_score(y_test, y_test_pred, average=average)
    f1 = f1_score(y_test, y_test_pred, average=average)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Accuracy", f"{test_score:.4f}")
        st.metric("🏋️ Train Acc", f"{train_score:.4f}")

    with col2:
        st.metric("📊 Precision", f"{precision:.4f}")

    with col3:
        st.metric("📈 Recall", f"{recall:.4f}")

    with col4:
        st.metric("⚖️ F1-Score", f"{f1:.4f}")

        # Performance assessment
        if test_score > 0.9:
            st.success("✅ Excellent")
        elif test_score > 0.8:
            st.success("✅ Good")
        elif test_score > 0.7:
            st.warning("📊 Moderate")
        else:
            st.error("⚠️ Poor")

    # Classification report
    with st.expander("📋 Detailed Classification Report"):
        report = classification_report(y_test, y_test_pred)
        st.code(report)

def display_diagnostic_plots():
    """Display diagnostic plots and visualizations."""

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Visualizations")

    model = st.session_state.trained_model
    y_test = st.session_state.y_test
    y_test_pred = model.predict(st.session_state.X_test)

    # Determine plot type based on task
    is_regression = pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 20

    if is_regression:
        display_regression_plots(y_test, y_test_pred)
    else:
        display_classification_plots(y_test, y_test_pred)

def display_regression_plots(y_true, y_pred):
    """Display regression diagnostic plots."""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Predictions vs Actual', 'Residual Plot',
                       'Residual Distribution', 'Q-Q Plot'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Predictions vs Actual
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                  marker=dict(color='blue', opacity=0.6)),
        row=1, col=1
    )
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # 2. Residual plot
    residuals = y_true - y_pred
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                  marker=dict(color='green', opacity=0.6)),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    # 3. Residual distribution
    fig.add_trace(
        go.Histogram(x=residuals, name='Residual Distribution',
                    marker=dict(color='purple', opacity=0.7)),
        row=2, col=1
    )

    # 4. Q-Q plot (simplified)
    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)

    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                  mode='markers', name='Q-Q Plot',
                  marker=dict(color='orange', opacity=0.6)),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(height=700, showlegend=False, title_text="Regression Diagnostics")
    fig.update_xaxes(title_text="True Values", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)

    st.plotly_chart(fig, width='stretch')

def display_classification_plots(y_true, y_pred):
    """Display classification diagnostic plots."""

    from sklearn.metrics import confusion_matrix

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       title="Confusion Matrix",
                       labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Class distribution comparison
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=true_counts.index, y=true_counts.values,
                            name='True Distribution', opacity=0.7))
        fig.add_trace(go.Bar(x=pred_counts.index, y=pred_counts.values,
                            name='Predicted Distribution', opacity=0.7))
        fig.update_layout(title="Class Distribution Comparison", barmode='group')
        st.plotly_chart(fig, width='stretch')

    # Probability distribution (if available)
    if hasattr(st.session_state.trained_model, 'predict_proba'):
        try:
            y_proba = st.session_state.trained_model.predict_proba(st.session_state.X_test)

            # Plot probability distribution
            fig = px.histogram(x=y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1),
                             title="Prediction Probability Distribution",
                             nbins=30)
            st.plotly_chart(fig, width='stretch')
        except (AttributeError, ValueError, IndexError):
            pass

def display_model_insights(model):
    """Display model insights and interpretation."""

    st.markdown("---")
    st.markdown("### 🔍 Model Insights")

    # Feature importance
    if hasattr(model, 'coef_') and model.coef_ is not None:
        display_feature_importance_detailed(model)

    # Statistical inference
    if hasattr(model, 'standard_errors_') and model.standard_errors_ is not None:
        display_statistical_inference(model)

    # Model comparison suggestions
    display_improvement_suggestions(model)

def display_feature_importance_detailed(model):
    """Display detailed feature importance analysis."""

    st.markdown("#### 🎯 Feature Importance Analysis")

    try:
        X_train = st.session_state.X_train
        coef_values = model.coef_.detach().cpu().numpy()

        # Get feature names
        feature_names = list(X_train.columns)
        if model.fit_intercept:
            feature_names = ['Intercept'] + feature_names

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef_values,
            'Abs_Coefficient': np.abs(coef_values)
        })

        # Sort by absolute importance
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=True)

        # Interactive plot
        fig = px.bar(
            importance_df.tail(15),  # Top 15 features
            x='Coefficient',
            y='Feature',
            orientation='h',
            title="Feature Coefficients (Top 15)",
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

        # Feature importance insights
        col1, col2 = st.columns(2)

        with col1:
            most_important = importance_df.loc[importance_df['Abs_Coefficient'].idxmax()]
            st.metric("🏆 Most Important Feature", most_important['Feature'])

        with col2:
            n_significant = len(importance_df[importance_df['Abs_Coefficient'] > 0.1])
            st.metric("📊 Significant Features", f"{n_significant}/{len(importance_df)}")

    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")

def display_statistical_inference(model):
    """Display statistical inference results."""

    st.markdown("#### 📊 Statistical Inference")

    try:
        # Confidence intervals
        conf_int = model.conf_int()

        st.markdown("**95% Confidence Intervals:**")

        # Format confidence intervals nicely
        conf_display = conf_int.copy()
        conf_display['Significant'] = ~((conf_display['lower'] <= 0) & (conf_display['upper'] >= 0))

        # Color code significant features
        def highlight_significant(row):
            return ['background-color: lightgreen' if row['Significant'] else ''
                   for _ in row.index]

        st.dataframe(
            conf_display.style.apply(highlight_significant, axis=1),
            width='stretch'
        )

        # Statistical summary
        n_significant = conf_display['Significant'].sum()
        st.info(f"📈 **Statistical Significance:** {n_significant}/{len(conf_display)} features have confidence intervals that don't include zero (highlighted in green)")

    except Exception as e:
        st.warning(f"Statistical inference not available: {str(e)}")

def display_improvement_suggestions(model):
    """Display suggestions for model improvement."""

    st.markdown("#### 💡 Improvement Suggestions")

    suggestions = []

    # Performance-based suggestions
    test_score = model.score(st.session_state.X_test, st.session_state.y_test)
    train_score = model.score(st.session_state.X_train, st.session_state.y_train)

    if train_score - test_score > 0.1:
        suggestions.append("⚠️ **Overfitting detected**: Consider regularization (Ridge/Lasso) or feature selection")

    if test_score < 0.7:
        suggestions.append("📊 **Low performance**: Try feature engineering, polynomial features, or different algorithms")

    # Feature-based suggestions
    if hasattr(model, 'coef_'):
        n_features = len(st.session_state.X_train.columns)
        if n_features > 20:
            suggestions.append("🔍 **Many features**: Consider feature selection or dimensionality reduction")

    # Model-specific suggestions
    if getattr(model, 'penalty', 'none') == 'none' and test_score < 0.8:
        suggestions.append("🔧 **No regularization**: Try Ridge or Lasso regression for better generalization")

    if suggestions:
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")
    else:
        st.success("✅ Model performance looks good! Consider trying ensemble methods for further improvement.")

    # Action buttons
    st.markdown("#### 🚀 Next Steps")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⚖️ Compare with Other Models", width='stretch'):
            st.switch_page("pages/model_comparison.py")

    with col2:
        if st.button("🔧 Retrain with Different Settings", width='stretch'):
            st.switch_page("pages/model_training.py")