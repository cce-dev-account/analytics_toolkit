"""
Results dashboard page for the Analytics Toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show():
    """Display the results dashboard page."""

    st.title("📈 Results Dashboard")
    st.markdown("Comprehensive model performance analysis and diagnostics")

    # Check if model results are available
    if 'trained_model' not in st.session_state:
        st.warning("🧠 No trained model available. Please train a model first.")
        if st.button("Go to Model Training"):
            st.switch_page("pages/model_training.py")
        return

    model = st.session_state.trained_model
    model_config = st.session_state.get('model_config', {})

    # Display model overview
    display_model_overview(model, model_config)

    # Performance metrics
    display_performance_metrics()

    # Visualizations
    display_diagnostic_plots()

    # Model insights
    display_model_insights(model)

def display_model_overview(model, config):
    """Display model overview information."""

    st.markdown("### 🧠 Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_name = config.get('model_name', 'Unknown Model')
        st.metric("🤖 Model Type", model_name)

    with col2:
        task_type = config.get('model_type', 'Unknown').title()
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

    st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

    # Probability distribution (if available)
    if hasattr(st.session_state.trained_model, 'predict_proba'):
        try:
            y_proba = st.session_state.trained_model.predict_proba(st.session_state.X_test)

            # Plot probability distribution
            fig = px.histogram(x=y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1),
                             title="Prediction Probability Distribution",
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        except:
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
        st.plotly_chart(fig, use_container_width=True)

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
            use_container_width=True
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
        if st.button("⚖️ Compare with Other Models", use_container_width=True):
            st.switch_page("pages/model_comparison.py")

    with col2:
        if st.button("🔧 Retrain with Different Settings", use_container_width=True):
            st.switch_page("pages/model_training.py")