"""
Home page for the Analytics Toolkit Streamlit interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def show():
    """Display the home page."""

    # Main title and description
    st.title("🧬 Analytics Toolkit")
    st.markdown("### *Advanced Machine Learning with Statistical Rigor*")

    st.markdown("---")

    # Introduction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome to the Analytics Toolkit! 🚀

        A comprehensive Python toolkit for data analytics and machine learning that combines
        the power of **PyTorch**, **scikit-learn**, and statistical inference to provide
        production-ready ML solutions with academic-level rigor.

        ### ✨ Key Features:

        - **🔧 Automated Preprocessing** - Intelligent data cleaning, encoding, and scaling
        - **🔬 Advanced Feature Engineering** - Transformations, selection, and interaction detection
        - **🧠 PyTorch Statistical Models** - Linear & logistic regression with full statistical inference
        - **🤖 AutoML Pipeline** - Automated model selection and hyperparameter optimization
        - **📊 Interactive Visualizations** - Comprehensive model diagnostics and performance metrics
        - **⚖️ Model Comparison** - Side-by-side evaluation of multiple algorithms

        """)

    with col2:
        st.info("""
        **📍 Getting Started:**

        1. Upload your data 📊
        2. Configure preprocessing 🔧
        3. Apply feature engineering 🔬
        4. Train models 🧠
        5. Analyze results 📈
        """)

    # Capabilities overview
    st.markdown("---")
    st.markdown("## 🎯 What You Can Do")

    # Create feature showcase
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Data Analysis
        - **Upload CSV/Excel files**
        - **Automatic data profiling**
        - **Missing value analysis**
        - **Distribution visualization**
        - **Correlation analysis**
        - **Interactive data exploration**
        """)

    with col2:
        st.markdown("""
        ### 🧠 Machine Learning
        - **Linear & Logistic Regression**
        - **Statistical inference (p-values, CI)**
        - **AutoML model selection**
        - **Hyperparameter optimization**
        - **Cross-validation**
        - **Model interpretability**
        """)

    with col3:
        st.markdown("""
        ### 📈 Results & Insights
        - **Interactive dashboards**
        - **Performance metrics**
        - **Residual analysis**
        - **Feature importance**
        - **Model comparison**
        - **Export capabilities**
        """)

    # Sample datasets section
    st.markdown("---")
    st.markdown("## 📋 Try with Sample Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏠 Generate Sample Regression Dataset", use_container_width=True):
            # Generate regression data
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)

            # Create DataFrame
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(8)])
            df['target'] = y

            # Add some categorical features
            categories = np.random.choice(['A', 'B', 'C'], size=500)
            df['category'] = categories

            # Store in session state
            st.session_state.uploaded_data = df
            st.session_state.data_source = "Sample Regression Data"

            st.success("✅ Sample regression dataset loaded! Go to 'Data Upload' to explore it.")

    with col2:
        if st.button("🎯 Generate Sample Classification Dataset", use_container_width=True):
            # Generate classification data
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=500, n_features=10, n_informative=8,
                n_redundant=2, n_clusters_per_class=1, random_state=42
            )

            # Create DataFrame
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(10)])
            df['target'] = y

            # Add categorical features
            categories = np.random.choice(['Group_1', 'Group_2', 'Group_3'], size=500)
            df['category'] = categories

            # Store in session state
            st.session_state.uploaded_data = df
            st.session_state.data_source = "Sample Classification Data"

            st.success("✅ Sample classification dataset loaded! Go to 'Data Upload' to explore it.")

    # Technical details
    st.markdown("---")
    st.markdown("## 🔬 Technical Highlights")

    # Create tabs for technical details
    tab1, tab2, tab3 = st.tabs(["🧠 Statistical Models", "🔬 Feature Engineering", "🤖 AutoML"])

    with tab1:
        st.markdown("""
        ### PyTorch Statistical Regression Models

        Our custom PyTorch implementation provides:

        - **Full statistical inference** with standard errors, t-statistics, p-values
        - **Confidence intervals** for coefficients and predictions
        - **Model diagnostics** including AIC, BIC, log-likelihood
        - **GPU acceleration** for large datasets
        - **Sklearn-compatible API** for easy integration
        - **Advanced regularization** (L1, L2, Elastic Net)
        - **Robust numerical methods** with condition number checking
        """)

    with tab2:
        st.markdown("""
        ### Advanced Feature Engineering

        Comprehensive feature transformation capabilities:

        - **Smart transformations**: Log, Box-Cox, outlier capping
        - **Advanced encoding**: Target encoding, Bayesian encoding
        - **Feature selection**: Variance, correlation, mutual information
        - **Interaction detection**: Tree-based and statistical methods
        - **Temporal features**: Date/time extraction, lag features
        - **Automated binning**: Quantile and uniform binning
        - **Pipeline integration**: Sklearn-compatible transformers
        """)

    with tab3:
        st.markdown("""
        ### AutoML Pipeline

        Automated machine learning workflow:

        - **Model selection**: Automatic algorithm comparison
        - **Hyperparameter optimization**: Bayesian optimization
        - **Feature preprocessing**: Automated feature engineering
        - **Cross-validation**: Robust performance estimation
        - **Ensemble methods**: Model combination strategies
        - **Time limits**: Configurable training time
        - **Interpretability**: Model explanation and feature importance
        """)

    # Performance metrics
    st.markdown("---")
    st.markdown("## 📊 Performance & Capabilities")

    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🧠 Models Available",
            value="5+",
            help="Linear/Logistic Regression + AutoML algorithms"
        )

    with col2:
        st.metric(
            label="🔬 Feature Transformers",
            value="15+",
            help="Advanced feature engineering transformations"
        )

    with col3:
        st.metric(
            label="📊 Visualization Types",
            value="10+",
            help="Interactive plots and diagnostic charts"
        )

    with col4:
        st.metric(
            label="⚡ Processing Speed",
            value="GPU Ready",
            help="PyTorch backend with CUDA support"
        )

    # Call to action
    st.markdown("---")

    st.info("""
    ### 🚀 Ready to Get Started?

    1. **Upload your data** using the 'Data Upload' page
    2. **Configure preprocessing** to clean and prepare your dataset
    3. **Apply feature engineering** to enhance your features
    4. **Train models** using PyTorch statistical regression or AutoML
    5. **Analyze results** with interactive dashboards and comparisons

    Navigate using the sidebar menu to begin your machine learning journey!
    """)

    # Quick links
    st.markdown("### 🔗 Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Upload Data", use_container_width=True):
            st.switch_page("pages/data_upload.py")

    with col2:
        if st.button("🧠 Train Models", use_container_width=True):
            st.switch_page("pages/model_training.py")

    with col3:
        if st.button("📈 View Results", use_container_width=True):
            st.switch_page("pages/results_dashboard.py")