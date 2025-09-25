"""
Analytics Toolkit - Streamlit Web Interface (Simple Version)
===========================================================

A web-based interface for the Analytics Toolkit with basic functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure page
st.set_page_config(
    page_title="Analytics Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application."""

    st.title("🧬 Analytics Toolkit")
    st.markdown("### *Advanced Machine Learning with Statistical Rigor*")

    # Sidebar
    st.sidebar.title("🧬 Analytics Toolkit")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.selectbox(
        "Select Page:",
        ["🏠 Home", "📊 Data Upload", "🔧 Preprocessing", "🧠 Model Training", "📈 Results"]
    )

    # Module availability check
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")

    modules_available = check_modules()
    for module, available in modules_available.items():
        emoji = "✅" if available else "❌"
        st.sidebar.markdown(f"{emoji} {module}")

    # Display selected page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Upload":
        show_data_upload()
    elif page == "🔧 Preprocessing":
        show_enhanced_preprocessing()
    elif page == "🧠 Model Training":
        show_enhanced_model_training()
    elif page == "📈 Results":
        show_results()

def check_modules():
    """Check which modules are available."""
    modules = {}

    try:
        import analytics_toolkit.preprocessing
        modules["Preprocessing"] = True
    except ImportError:
        modules["Preprocessing"] = False

    try:
        import analytics_toolkit.pytorch_regression
        modules["PyTorch Models"] = True
    except ImportError:
        modules["PyTorch Models"] = False

    return modules

def show_home():
    """Home page."""

    st.markdown("---")
    st.markdown("## Welcome to Analytics Toolkit! 🚀")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### ✨ Key Features:

        - **🔧 Automated Preprocessing** - Data cleaning and preparation
        - **🧠 PyTorch Statistical Models** - Linear & logistic regression
        - **📊 Interactive Visualizations** - Comprehensive diagnostics
        - **⚡ Real-time Analysis** - Instant results and insights
        """)

    with col2:
        st.info("""
        **📍 Quick Start:**
        1. Upload data 📊
        2. Preprocess 🔧
        3. Train model 🧠
        4. View results 📈
        """)

    # Sample data generation
    st.markdown("### 🎯 Try Sample Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Generate Regression Data", key="sample_reg"):
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)

            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
            df['target'] = y
            df['category'] = np.random.choice(['A', 'B', 'C'], size=500)

            st.session_state.data = df
            st.success("✅ Regression dataset generated!")

    with col2:
        if st.button("🎯 Generate Classification Data", key="sample_class"):
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=500,
                n_features=5,
                n_informative=3,
                n_redundant=1,
                n_clusters_per_class=1,
                random_state=42
            )

            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
            df['target'] = y
            df['category'] = np.random.choice(['Group_1', 'Group_2'], size=500)

            st.session_state.data = df
            st.success("✅ Classification dataset generated!")

def show_data_upload():
    """Data upload page."""

    st.markdown("## 📊 Data Upload & Preview")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )

    data = None

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.session_state.data = data
            st.success(f"✅ Loaded {len(data)} rows and {len(data.columns)} columns")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    elif 'data' in st.session_state:
        data = st.session_state.data

    if data is not None:
        # Data overview
        st.markdown("### 📋 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", f"{len(data):,}")
        with col2:
            st.metric("📈 Columns", len(data.columns))
        with col3:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("❓ Missing", f"{missing_pct:.1f}%")
        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Memory", f"{memory_mb:.1f} MB")

        # Data preview
        st.markdown("### 👀 Data Preview")
        st.dataframe(data.head(10), width=1200)

        # Basic visualization
        st.markdown("### 📊 Quick Analysis")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column for distribution", numeric_cols)

            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, width=1200)

    else:
        st.info("👆 Upload a file or generate sample data to begin")

def show_preprocessing():
    """Preprocessing page."""

    st.markdown("## 🔧 Data Preprocessing")

    if 'data' not in st.session_state:
        st.warning("📊 Please upload data first")
        return

    data = st.session_state.data

    # Target selection
    st.markdown("### 🎯 Select Target Column")
    target_col = st.selectbox("Target variable", data.columns.tolist())

    # Preprocessing options
    st.markdown("### ⚙️ Preprocessing Options")

    col1, col2 = st.columns(2)

    with col1:
        scaling_method = st.selectbox(
            "Scaling method",
            ["standard", "minmax", "none"],
            help="StandardScaler, MinMaxScaler, or no scaling"
        )

    with col2:
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)

    # Apply preprocessing
    if st.button("🚀 Apply Preprocessing", type="primary"):
        try:
            from analytics_toolkit.preprocessing import DataPreprocessor, create_train_test_split

            with st.spinner("Processing..."):
                # Initialize preprocessor
                preprocessor = DataPreprocessor()

                # Apply preprocessing
                X, y = preprocessor.fit_transform(data, target_column=target_col, scaling_method=scaling_method)

                # Train-test split
                X_train, X_test, y_train, y_test = create_train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Store in session
                st.session_state.update({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'target_col': target_col,
                    'preprocessor': preprocessor
                })

                st.success("✅ Preprocessing completed!")

                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🏋️ Training samples", len(X_train))
                with col2:
                    st.metric("🧪 Test samples", len(X_test))

        except ImportError:
            st.error("❌ Analytics Toolkit preprocessing not available")
        except Exception as e:
            st.error(f"❌ Error: {e}")

def show_enhanced_preprocessing():
    """Enhanced preprocessing page with advanced features."""
    try:
        # Import our enhanced preprocessing page
        import sys
        from pathlib import Path

        # Add ui to path if not already there
        ui_path = Path(__file__).parent / "ui"
        if str(ui_path) not in sys.path:
            sys.path.insert(0, str(ui_path))

        from pages.preprocessing import show
        show()

    except ImportError as e:
        st.error(f"❌ Could not load enhanced preprocessing: {e}")
        # Fallback to basic preprocessing
        show_preprocessing()
    except Exception as e:
        st.error(f"❌ Error in enhanced preprocessing: {e}")
        show_preprocessing()


def show_enhanced_model_training():
    """Enhanced model training page with advanced features."""
    try:
        # Import our enhanced model training page
        import sys
        from pathlib import Path

        # Add ui to path if not already there
        ui_path = Path(__file__).parent / "ui"
        if str(ui_path) not in sys.path:
            sys.path.insert(0, str(ui_path))

        from pages.model_training import show
        show()

    except ImportError as e:
        st.error(f"❌ Could not load enhanced model training: {e}")
        # Fallback to basic model training
        show_basic_model_training()
    except Exception as e:
        st.error(f"❌ Error in enhanced model training: {e}")
        show_basic_model_training()


def show_basic_model_training():
    """Basic model training fallback."""

    st.markdown("## 🧠 Model Training (Basic)")

    if not all(k in st.session_state for k in ['X_train', 'y_train']):
        st.warning("🔧 Please complete preprocessing first")
        return

    # Model selection
    y_train = st.session_state.y_train
    is_regression = pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 10

    task_type = "Regression" if is_regression else "Classification"
    st.info(f"🎯 **Task Type:** {task_type}")

    # Model configuration
    col1, col2 = st.columns(2)

    with col1:
        if is_regression:
            model_type = st.selectbox("Model", ["Linear Regression", "Ridge Regression"])
        else:
            model_type = st.selectbox("Model", ["Logistic Regression"])

    with col2:
        if "Ridge" in model_type:
            alpha = st.slider("Regularization (α)", 0.01, 10.0, 1.0)
        else:
            alpha = 0.0

    # Train model
    if st.button("🚀 Train Model", type="primary"):
        try:
            from analytics_toolkit.pytorch_regression import LinearRegression, LogisticRegression

            X_train = st.session_state.X_train
            y_train = st.session_state.y_train

            with st.spinner("Training..."):
                if is_regression:
                    if "Ridge" in model_type:
                        model = LinearRegression(penalty='l2', alpha=alpha)
                    else:
                        model = LinearRegression()
                else:
                    model = LogisticRegression()

                # Train
                model.fit(X_train, y_train)

                # Store model
                st.session_state.model = model
                st.session_state.model_type = model_type

                st.success("✅ Model training completed!")

                # Show training results
                train_score = model.score(X_train, y_train)
                test_score = model.score(st.session_state.X_test, st.session_state.y_test)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🏋️ Training Score", f"{train_score:.4f}")
                with col2:
                    st.metric("🧪 Test Score", f"{test_score:.4f}")

        except ImportError:
            st.error("❌ PyTorch models not available")
        except Exception as e:
            st.error(f"❌ Training error: {e}")

def show_results():
    """Results page."""

    st.markdown("## 📈 Results Dashboard")

    if 'model' not in st.session_state:
        st.warning("🧠 Please train a model first")
        return

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    model_type = st.session_state.get('model_type', 'Unknown')

    # Performance metrics
    st.markdown("### 📊 Performance Metrics")

    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)

    st.metric("🎯 Model Performance", f"{test_score:.4f}")

    # Determine task type
    is_regression = pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 10

    # Visualizations
    st.markdown("### 📊 Visualizations")

    if is_regression:
        # Regression plots
        col1, col2 = st.columns(2)

        with col1:
            # Predictions vs Actual
            fig1 = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title='Predictions vs Actual'
            )
            fig1.add_shape(
                type="line",
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max(),
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig1, width=600)

        with col2:
            # Residuals
            residuals = y_test - y_pred
            fig2 = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicted', 'y': 'Residuals'},
                title='Residual Plot'
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, width=600)

    else:
        # Classification plots
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)

        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual")
        st.plotly_chart(fig, width=800)

    # Feature importance
    if hasattr(model, 'coef_') and model.coef_ is not None:
        st.markdown("### 🎯 Feature Importance")

        coef_values = model.coef_.detach().cpu().numpy()
        feature_names = list(X_test.columns)

        # Handle multi-dimensional coefficients (e.g., logistic regression)
        if coef_values.ndim > 1:
            coef_values = coef_values.flatten()

        # Ensure arrays have the same length
        if model.fit_intercept and len(coef_values) == len(feature_names) + 1:
            feature_names = ['Intercept'] + feature_names
        elif len(coef_values) != len(feature_names):
            # Fallback to generic names if lengths don't match
            feature_names = [f'Feature_{i}' for i in range(len(coef_values))]

        # Create importance plot
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef_values
        })
        importance_df = importance_df.reindex(
            importance_df['Coefficient'].abs().sort_values(ascending=True).index
        )

        fig = px.bar(
            importance_df.tail(10),
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Coefficients'
        )
        st.plotly_chart(fig, width=1000)

    # Statistical summary
    if hasattr(model, 'summary'):
        with st.expander("📊 Statistical Summary"):
            try:
                summary = model.summary()
                st.code(summary, language=None)
            except:
                st.info("Statistical summary not available")

if __name__ == "__main__":
    main()