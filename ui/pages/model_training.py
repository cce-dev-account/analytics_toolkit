"""
Model training page for the Analytics Toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def show():
    """Display the model training page."""

    st.title("🧠 Model Training")
    st.markdown("Train PyTorch statistical models with comprehensive inference")

    # Check if preprocessed data is available
    if not all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
        st.warning("🔧 Please complete data preprocessing first.")
        if st.button("Go to Preprocessing"):
            st.switch_page("pages/preprocessing.py")
        return

    # Display data info
    display_training_data_info()

    # Model selection and configuration
    model_config = configure_model_training()

    # Train model button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        trained_model = train_selected_model(model_config)

        if trained_model is not None:
            st.session_state.trained_model = trained_model
            st.session_state.model_config = model_config
            st.success("✅ Model training completed!")

            # Display training results
            display_training_results(trained_model)

def display_training_data_info():
    """Display information about the training data."""

    st.markdown("### 📊 Training Data Overview")

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🏋️ Training Samples", f"{len(X_train):,}")

    with col2:
        st.metric("🧪 Test Samples", f"{len(X_test):,}")

    with col3:
        st.metric("📈 Features", len(X_train.columns))

    with col4:
        # Determine task type
        if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 20:
            task_type = "Regression"
        else:
            task_type = f"Classification ({y_train.nunique()} classes)"
        st.metric("🎯 Task Type", task_type)

def configure_model_training():
    """Configure model training parameters."""

    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")

    config = {}
    y_train = st.session_state.y_train

    # Determine task type
    is_regression = pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 20

    # Model selection
    if is_regression:
        st.info("🎯 **Task Type:** Regression")
        config['model_type'] = 'regression'

        model_options = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
        config['model_name'] = st.selectbox("Select Model", model_options)

    else:
        st.info("🎯 **Task Type:** Classification")
        config['model_type'] = 'classification'

        model_options = ['Logistic Regression', 'Ridge Classification']
        config['model_name'] = st.selectbox("Select Model", model_options)

    # Model parameters
    st.markdown("#### 🔧 Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        config['fit_intercept'] = st.checkbox("Fit Intercept", value=True)

        if 'Ridge' in config['model_name'] or 'Lasso' in config['model_name']:
            config['alpha'] = st.slider(
                "Regularization Strength (α)",
                min_value=0.001,
                max_value=10.0,
                value=1.0,
                step=0.001,
                format="%.3f"
            )

    with col2:
        config['max_iter'] = st.slider("Max Iterations", 100, 5000, 1000)
        config['tol'] = st.selectbox(
            "Tolerance",
            [1e-3, 1e-4, 1e-5, 1e-6],
            index=1,
            format_func=lambda x: f"{x:.0e}"
        )

    # Advanced options
    with st.expander("🔬 Advanced Options"):
        config['device'] = st.selectbox("Computation Device", ['cpu', 'auto'], index=1)

        if config['model_type'] == 'regression':
            config['solver'] = st.selectbox("Solver", ['auto', 'qr', 'normal_equation'])
        else:
            config['solver'] = st.selectbox("Solver", ['lbfgs', 'adam', 'sgd'])

    return config

def train_selected_model(config):
    """Train the selected model."""

    try:
        # Import models
        from analytics_toolkit.pytorch_regression import LinearRegression, LogisticRegression

        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        with st.spinner("🔄 Training model..."):
            # Initialize model based on type
            if config['model_type'] == 'regression':
                if 'Ridge' in config['model_name']:
                    model = LinearRegression(
                        fit_intercept=config['fit_intercept'],
                        penalty='l2',
                        alpha=config.get('alpha', 1.0),
                        max_iter=config['max_iter'],
                        tol=config['tol'],
                        device=config['device'],
                        solver=config.get('solver', 'auto')
                    )
                elif 'Lasso' in config['model_name']:
                    model = LinearRegression(
                        fit_intercept=config['fit_intercept'],
                        penalty='l1',
                        alpha=config.get('alpha', 1.0),
                        max_iter=config['max_iter'],
                        tol=config['tol'],
                        device=config['device'],
                        solver='gradient_descent'  # L1 requires iterative solver
                    )
                else:  # Linear Regression
                    model = LinearRegression(
                        fit_intercept=config['fit_intercept'],
                        penalty='none',
                        max_iter=config['max_iter'],
                        tol=config['tol'],
                        device=config['device'],
                        solver=config.get('solver', 'auto')
                    )

            else:  # Classification
                if 'Ridge' in config['model_name']:
                    model = LogisticRegression(
                        fit_intercept=config['fit_intercept'],
                        penalty='l2',
                        alpha=config.get('alpha', 1.0),
                        max_iter=config['max_iter'],
                        tol=config['tol'],
                        device=config['device'],
                        solver=config.get('solver', 'lbfgs')
                    )
                else:  # Logistic Regression
                    model = LogisticRegression(
                        fit_intercept=config['fit_intercept'],
                        penalty='none',
                        max_iter=config['max_iter'],
                        tol=config['tol'],
                        device=config['device'],
                        solver=config.get('solver', 'lbfgs')
                    )

            # Train the model
            model.fit(X_train, y_train)

            return model

    except ImportError:
        st.error("❌ PyTorch regression models not available. Please install the full Analytics Toolkit.")
        return None
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        return None

def display_training_results(model):
    """Display training results and model performance."""

    st.markdown("---")
    st.markdown("### 📊 Training Results")

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🏋️ Training Score", f"{train_score:.4f}")

    with col2:
        st.metric("🧪 Test Score", f"{test_score:.4f}")

        # Check for overfitting
        if train_score - test_score > 0.1:
            st.warning("⚠️ Potential overfitting detected")

    # Model-specific results
    if hasattr(model, 'log_likelihood_'):
        col1, col2, col3 = st.columns(3)

        with col1:
            if model.log_likelihood_ is not None:
                st.metric("📈 Log Likelihood", f"{model.log_likelihood_:.2f}")

        with col2:
            if model.aic_ is not None:
                st.metric("📊 AIC", f"{model.aic_:.2f}")

        with col3:
            if model.bic_ is not None:
                st.metric("📊 BIC", f"{model.bic_:.2f}")

    # Convergence info for iterative models
    if hasattr(model, 'n_iter_'):
        if model.n_iter_ is not None:
            st.info(f"🔄 **Convergence:** Model converged in {model.n_iter_} iterations")

    # Statistical summary
    if hasattr(model, 'summary'):
        try:
            with st.expander("📊 Statistical Summary", expanded=False):
                summary = model.summary()
                st.code(summary, language=None)
        except:
            st.info("Statistical summary not available")

    # Feature importance (coefficients)
    if hasattr(model, 'coef_') and model.coef_ is not None:
        display_feature_importance(model)

    # Store predictions for results page
    st.session_state.y_train_pred = y_train_pred
    st.session_state.y_test_pred = y_test_pred

    # Navigation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 View Results Dashboard", use_container_width=True):
            st.switch_page("pages/results_dashboard.py")

    with col2:
        if st.button("⚖️ Compare Models", use_container_width=True):
            st.switch_page("pages/model_comparison.py")

def display_feature_importance(model):
    """Display feature importance based on model coefficients."""

    st.markdown("#### 🎯 Feature Importance")

    try:
        X_train = st.session_state.X_train
        coef_values = model.coef_.detach().cpu().numpy()

        # Get feature names
        if model.fit_intercept:
            feature_names = ['Intercept'] + list(X_train.columns)
        else:
            feature_names = list(X_train.columns)

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef_values,
            'Abs_Coefficient': np.abs(coef_values)
        })

        # Sort by absolute importance
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)

        # Plot top features
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)

        fig = px.bar(
            top_features,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title=f"Top {top_n} Feature Coefficients",
            color='Coefficient',
            color_continuous_scale='RdBu'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Show coefficient table
        st.dataframe(importance_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")