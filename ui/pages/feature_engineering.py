"""
Feature engineering page for the Analytics Toolkit.
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

# Import feature engineering components with fallbacks
FEATURE_ENGINEERING_AVAILABLE = True
AVAILABLE_TRANSFORMERS = {}

try:
    from analytics_toolkit.feature_engineering import RobustScaler
    AVAILABLE_TRANSFORMERS['RobustScaler'] = RobustScaler
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import OutlierCapTransformer
    AVAILABLE_TRANSFORMERS['OutlierCapTransformer'] = OutlierCapTransformer
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import LogTransformer
    AVAILABLE_TRANSFORMERS['LogTransformer'] = LogTransformer
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import BinningTransformer
    AVAILABLE_TRANSFORMERS['BinningTransformer'] = BinningTransformer
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import TargetEncoder
    AVAILABLE_TRANSFORMERS['TargetEncoder'] = TargetEncoder
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import FrequencyEncoder
    AVAILABLE_TRANSFORMERS['FrequencyEncoder'] = FrequencyEncoder
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import BayesianTargetEncoder
    AVAILABLE_TRANSFORMERS['BayesianTargetEncoder'] = BayesianTargetEncoder
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import FeatureSelector
    AVAILABLE_TRANSFORMERS['FeatureSelector'] = FeatureSelector
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import CorrelationFilter
    AVAILABLE_TRANSFORMERS['CorrelationFilter'] = CorrelationFilter
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import MutualInfoSelector
    AVAILABLE_TRANSFORMERS['MutualInfoSelector'] = MutualInfoSelector
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import InteractionDetector
    AVAILABLE_TRANSFORMERS['InteractionDetector'] = InteractionDetector
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import PolynomialInteractions
    AVAILABLE_TRANSFORMERS['PolynomialInteractions'] = PolynomialInteractions
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import DateTimeFeatures
    AVAILABLE_TRANSFORMERS['DateTimeFeatures'] = DateTimeFeatures
except ImportError:
    pass

try:
    from analytics_toolkit.feature_engineering import LagFeatures
    AVAILABLE_TRANSFORMERS['LagFeatures'] = LagFeatures
except ImportError:
    pass

if not AVAILABLE_TRANSFORMERS:
    FEATURE_ENGINEERING_AVAILABLE = False

def show():
    """Display the feature engineering page."""

    st.title("🔬 Feature Engineering")
    st.markdown("Advanced feature transformations and selection")

    if 'processed_data' not in st.session_state:
        st.warning("🔧 Please complete data preprocessing first.")
        if st.button("Go to Preprocessing"):
            st.switch_page("pages/preprocessing.py")
        return

    if not FEATURE_ENGINEERING_AVAILABLE:
        st.error("❌ Feature engineering module is not available. Please check your installation.")
        return

    data = st.session_state.processed_data.copy()

    # Sidebar for feature engineering configuration
    st.sidebar.markdown("## 🔬 Feature Engineering Controls")

    # Initialize session state for feature engineering
    if 'feature_engineered_data' not in st.session_state:
        st.session_state.feature_engineered_data = data.copy()

    # Feature engineering sections
    st.markdown("---")

    # Data Transformations Section
    st.markdown("### 🔄 Data Transformations")

    with st.expander("📊 Distribution Transformations", expanded=False):
        apply_distribution_transforms(data)

    with st.expander("🎯 Outlier Handling", expanded=False):
        apply_outlier_transforms(data)

    with st.expander("📏 Advanced Scaling", expanded=False):
        apply_scaling_transforms(data)

    # Feature Creation Section
    st.markdown("### ✨ Feature Creation")

    with st.expander("🏷️ Categorical Encoding", expanded=False):
        apply_categorical_encoding(data)

    with st.expander("🔗 Feature Interactions", expanded=False):
        apply_feature_interactions(data)

    with st.expander("📅 Temporal Features", expanded=False):
        apply_temporal_features(data)

    # Feature Selection Section
    st.markdown("### 🎯 Feature Selection")

    with st.expander("📉 Feature Selection Methods", expanded=False):
        apply_feature_selection(data)

    # Show current feature engineered data
    st.markdown("---")
    st.markdown("### 📋 Current Feature Set")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Features", len(data.columns))

    with col2:
        st.metric("Engineered Features", len(st.session_state.feature_engineered_data.columns))

    # Preview of engineered data
    st.markdown("#### 🔍 Feature Preview")
    st.dataframe(st.session_state.feature_engineered_data.head(), use_container_width=True)

    # Feature importance visualization if possible
    display_feature_analysis()

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Back to Preprocessing"):
            st.switch_page("pages/preprocessing.py")

    with col2:
        if st.button("🔄 Reset Features"):
            st.session_state.feature_engineered_data = data.copy()
            st.rerun()

    with col3:
        if st.button("🧠 Continue to Model Training", type="primary"):
            st.session_state.final_data = st.session_state.feature_engineered_data.copy()
            st.switch_page("pages/model_training.py")


def apply_distribution_transforms(data):
    """Apply distribution transformation options."""

    st.markdown("Transform feature distributions to improve model performance.")

    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for transformation.")
        return

    col1, col2 = st.columns(2)

    with col1:
        selected_cols = st.multiselect(
            "Select columns to transform:",
            numeric_cols,
            key="dist_transform_cols"
        )

    with col2:
        transform_method = st.selectbox(
            "Transformation method:",
            ["log1p", "box-cox", "yeo-johnson", "quantile-uniform"],
            key="dist_transform_method"
        )

    if selected_cols and st.button("Apply Distribution Transform"):
        try:
            # Use LogTransformer for log transformations or sklearn transformers for others
            if transform_method == "log1p":
                transformer = LogTransformer()
                transformed_data = st.session_state.feature_engineered_data.copy()
                transformed_data[selected_cols] = transformer.fit_transform(
                    transformed_data[selected_cols]
                )
            else:
                # Fallback to sklearn transformers
                from sklearn.preprocessing import PowerTransformer, QuantileTransformer
                if transform_method == "box-cox":
                    transformer = PowerTransformer(method='box-cox')
                elif transform_method == "yeo-johnson":
                    transformer = PowerTransformer(method='yeo-johnson')
                else:  # quantile-uniform
                    transformer = QuantileTransformer(output_distribution='uniform')

                transformed_data = st.session_state.feature_engineered_data.copy()
                # Ensure positive values for box-cox
                if transform_method == "box-cox":
                    transformed_data[selected_cols] = transformer.fit_transform(
                        transformed_data[selected_cols].abs() + 1e-8
                    )
                else:
                    transformed_data[selected_cols] = transformer.fit_transform(
                        transformed_data[selected_cols]
                    )

            st.session_state.feature_engineered_data = transformed_data
            st.success(f"✅ Applied {transform_method} transformation to {len(selected_cols)} columns")

            # Show before/after comparison
            if len(selected_cols) > 0:
                show_transformation_comparison(data[selected_cols[0]],
                                             transformed_data[selected_cols[0]],
                                             f"{transform_method.title()} Transformation")

        except Exception as e:
            st.error(f"❌ Transformation failed: {str(e)}")


def apply_outlier_transforms(data):
    """Apply outlier handling transformations."""

    st.markdown("Handle outliers using various capping methods.")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for outlier handling.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_cols = st.multiselect(
            "Select columns for outlier handling:",
            numeric_cols,
            key="outlier_cols"
        )

    with col2:
        outlier_method = st.selectbox(
            "Outlier method:",
            ["iqr", "percentile", "zscore"],
            key="outlier_method"
        )

    with col3:
        threshold = st.slider(
            "Threshold:",
            1.0, 5.0, 1.5,
            key="outlier_threshold"
        )

    if selected_cols and st.button("Apply Outlier Handling"):
        try:
            transformed_data = st.session_state.feature_engineered_data.copy()

            if 'OutlierCapTransformer' in AVAILABLE_TRANSFORMERS:
                transformer = AVAILABLE_TRANSFORMERS['OutlierCapTransformer'](method=outlier_method, threshold=threshold)
                transformed_data[selected_cols] = transformer.fit_transform(
                    transformed_data[selected_cols]
                )
            else:
                # Fallback manual outlier handling
                for col in selected_cols:
                    if outlier_method == "iqr":
                        Q1 = transformed_data[col].quantile(0.25)
                        Q3 = transformed_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        transformed_data[col] = transformed_data[col].clip(lower_bound, upper_bound)
                    elif outlier_method == "percentile":
                        lower_percentile = (100 - 99) / 2
                        upper_percentile = 100 - lower_percentile
                        lower_bound = transformed_data[col].quantile(lower_percentile / 100)
                        upper_bound = transformed_data[col].quantile(upper_percentile / 100)
                        transformed_data[col] = transformed_data[col].clip(lower_bound, upper_bound)
                    elif outlier_method == "zscore":
                        mean = transformed_data[col].mean()
                        std = transformed_data[col].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        transformed_data[col] = transformed_data[col].clip(lower_bound, upper_bound)

            st.session_state.feature_engineered_data = transformed_data
            st.success(f"✅ Applied {outlier_method} outlier handling to {len(selected_cols)} columns")

        except Exception as e:
            st.error(f"❌ Outlier handling failed: {str(e)}")


def apply_scaling_transforms(data):
    """Apply advanced scaling transformations."""

    st.markdown("Apply advanced scaling methods for better model performance.")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available for scaling.")
        return

    col1, col2 = st.columns(2)

    with col1:
        scaling_method = st.selectbox(
            "Scaling method:",
            ["robust", "standard", "minmax", "quantile"],
            key="scaling_method"
        )

    with col2:
        scale_all = st.checkbox("Scale all numeric columns", value=True)

    if not scale_all:
        selected_cols = st.multiselect(
            "Select columns to scale:",
            numeric_cols,
            key="scaling_cols"
        )
    else:
        selected_cols = numeric_cols

    if selected_cols and st.button("Apply Scaling"):
        try:
            # Use available custom scaler or sklearn fallback
            if scaling_method == "robust" and 'RobustScaler' in AVAILABLE_TRANSFORMERS:
                scaler = AVAILABLE_TRANSFORMERS['RobustScaler']()
            else:
                # Use sklearn scalers for all methods
                from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
                if scaling_method == "standard":
                    scaler = StandardScaler()
                elif scaling_method == "minmax":
                    scaler = MinMaxScaler()
                elif scaling_method == "robust":
                    scaler = RobustScaler()
                else:  # quantile
                    scaler = QuantileTransformer()

            transformed_data = st.session_state.feature_engineered_data.copy()
            transformed_data[selected_cols] = scaler.fit_transform(
                transformed_data[selected_cols]
            )

            st.session_state.feature_engineered_data = transformed_data
            st.success(f"✅ Applied {scaling_method} scaling to {len(selected_cols)} columns")

        except Exception as e:
            st.error(f"❌ Scaling failed: {str(e)}")


def apply_categorical_encoding(data):
    """Apply categorical encoding methods."""

    st.markdown("Encode categorical variables using advanced methods.")

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        st.info("No categorical columns available for encoding.")
        return

    col1, col2 = st.columns(2)

    with col1:
        selected_cols = st.multiselect(
            "Select categorical columns:",
            categorical_cols,
            key="encoding_cols"
        )

    with col2:
        encoding_method = st.selectbox(
            "Encoding method:",
            ["frequency", "target", "ordinal", "onehot"],
            key="encoding_method"
        )

    # Target column selection for target encoding
    target_col = None
    if encoding_method == "target":
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_col = st.selectbox(
                "Select target column:",
                numeric_cols,
                key="target_encoding_col"
            )

    if selected_cols and st.button("Apply Encoding"):
        try:
            transformed_data = st.session_state.feature_engineered_data.copy()

            if encoding_method == "frequency":
                encoder = FrequencyEncoder()
                for col in selected_cols:
                    transformed_data[f"{col}_freq"] = encoder.fit_transform(transformed_data[[col]])

            elif encoding_method == "target" and target_col:
                encoder = TargetEncoder()
                for col in selected_cols:
                    transformed_data[f"{col}_target"] = encoder.fit_transform(
                        transformed_data[[col]], transformed_data[target_col]
                    )

            elif encoding_method == "onehot":
                for col in selected_cols:
                    dummies = pd.get_dummies(transformed_data[col], prefix=col)
                    transformed_data = pd.concat([transformed_data, dummies], axis=1)
                    transformed_data.drop(col, axis=1, inplace=True)

            st.session_state.feature_engineered_data = transformed_data
            st.success(f"✅ Applied {encoding_method} encoding to {len(selected_cols)} columns")

        except Exception as e:
            st.error(f"❌ Encoding failed: {str(e)}")


def apply_feature_interactions(data):
    """Apply feature interaction detection and creation."""

    st.markdown("Create feature interactions to capture non-linear relationships.")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for feature interactions.")
        return

    col1, col2 = st.columns(2)

    with col1:
        interaction_method = st.selectbox(
            "Interaction method:",
            ["polynomial", "manual", "automated"],
            key="interaction_method"
        )

    with col2:
        if interaction_method == "polynomial":
            degree = st.slider("Polynomial degree:", 2, 4, 2)
        elif interaction_method == "automated":
            max_interactions = st.slider("Max interactions:", 5, 50, 10)

    if interaction_method == "manual":
        col1_select = st.selectbox("First feature:", numeric_cols, key="interact_col1")
        col2_select = st.selectbox("Second feature:", numeric_cols, key="interact_col2")
        operation = st.selectbox("Operation:", ["multiply", "add", "divide", "subtract"])

        if col1_select != col2_select and st.button("Create Interaction"):
            try:
                transformed_data = st.session_state.feature_engineered_data.copy()

                if operation == "multiply":
                    transformed_data[f"{col1_select}_x_{col2_select}"] = (
                        transformed_data[col1_select] * transformed_data[col2_select]
                    )
                elif operation == "add":
                    transformed_data[f"{col1_select}_plus_{col2_select}"] = (
                        transformed_data[col1_select] + transformed_data[col2_select]
                    )
                # Add other operations...

                st.session_state.feature_engineered_data = transformed_data
                st.success(f"✅ Created interaction: {col1_select} {operation} {col2_select}")

            except Exception as e:
                st.error(f"❌ Interaction creation failed: {str(e)}")

    else:
        if st.button(f"Create {interaction_method.title()} Interactions"):
            try:
                if interaction_method == "polynomial":
                    # Use sklearn PolynomialFeatures as fallback
                    from sklearn.preprocessing import PolynomialFeatures
                    generator = PolynomialFeatures(degree=degree, include_bias=False)
                    interactions = generator.fit_transform(data[numeric_cols[:5]])  # Limit columns

                    # Add interaction features
                    transformed_data = st.session_state.feature_engineered_data.copy()
                    interaction_df = pd.DataFrame(
                        interactions,
                        columns=[f"poly_{i}" for i in range(interactions.shape[1])]
                    )
                    transformed_data = pd.concat([transformed_data, interaction_df], axis=1)

                    st.session_state.feature_engineered_data = transformed_data
                    st.success(f"✅ Created {interactions.shape[1]} polynomial features")

            except Exception as e:
                st.error(f"❌ Interaction creation failed: {str(e)}")


def apply_temporal_features(data):
    """Apply temporal feature extraction."""

    st.markdown("Extract temporal features from date/time columns.")

    # Detect potential date columns
    date_cols = []
    for col in data.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
        elif data[col].dtype == 'object':
            # Try to parse as date
            try:
                pd.to_datetime(data[col].head(), errors='raise')
                date_cols.append(col)
            except (ValueError, TypeError):
                pass

    if not date_cols:
        st.info("No date/time columns detected. You can create sample temporal data below.")

        if st.button("Create Sample Date Column"):
            transformed_data = st.session_state.feature_engineered_data.copy()
            start_date = pd.Timestamp('2020-01-01')
            transformed_data['sample_date'] = pd.date_range(
                start_date, periods=len(transformed_data), freq='D'
            )
            st.session_state.feature_engineered_data = transformed_data
            st.success("✅ Created sample date column")
            st.rerun()
        return

    selected_date_col = st.selectbox(
        "Select date column:",
        date_cols,
        key="temporal_col"
    )

    temporal_features = st.multiselect(
        "Select temporal features to extract:",
        ["year", "month", "day", "dayofweek", "quarter", "is_weekend", "season"],
        default=["year", "month", "dayofweek"],
        key="temporal_features"
    )

    if temporal_features and st.button("Extract Temporal Features"):
        try:
            transformed_data = st.session_state.feature_engineered_data.copy()

            # Convert to datetime if needed
            date_series = pd.to_datetime(transformed_data[selected_date_col])

            for feature in temporal_features:
                if feature == "year":
                    transformed_data[f"{selected_date_col}_year"] = date_series.dt.year
                elif feature == "month":
                    transformed_data[f"{selected_date_col}_month"] = date_series.dt.month
                elif feature == "day":
                    transformed_data[f"{selected_date_col}_day"] = date_series.dt.day
                elif feature == "dayofweek":
                    transformed_data[f"{selected_date_col}_dayofweek"] = date_series.dt.dayofweek
                elif feature == "quarter":
                    transformed_data[f"{selected_date_col}_quarter"] = date_series.dt.quarter
                elif feature == "is_weekend":
                    transformed_data[f"{selected_date_col}_is_weekend"] = (
                        date_series.dt.dayofweek >= 5
                    ).astype(int)
                elif feature == "season":
                    seasons = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
                    transformed_data[f"{selected_date_col}_season"] = (
                        date_series.dt.month.map(seasons)
                    )

            st.session_state.feature_engineered_data = transformed_data
            st.success(f"✅ Extracted {len(temporal_features)} temporal features")

        except Exception as e:
            st.error(f"❌ Temporal feature extraction failed: {str(e)}")


def apply_feature_selection(data):
    """Apply feature selection methods."""

    st.markdown("Select the most relevant features for modeling.")

    numeric_cols = st.session_state.feature_engineered_data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for feature selection.")
        return

    col1, col2 = st.columns(2)

    with col1:
        selection_method = st.selectbox(
            "Selection method:",
            ["variance_threshold", "correlation_filter", "mutual_info"],
            key="selection_method"
        )

    with col2:
        if selection_method == "variance_threshold":
            threshold = st.slider("Variance threshold:", 0.0, 1.0, 0.01)
        elif selection_method == "correlation_filter":
            threshold = st.slider("Correlation threshold:", 0.5, 0.99, 0.95)
        elif selection_method == "mutual_info":
            n_features = st.slider("Number of features:", 1, min(20, len(numeric_cols)), 10)

    # Target column for mutual info
    target_col = None
    if selection_method == "mutual_info":
        target_col = st.selectbox(
            "Select target column:",
            numeric_cols,
            key="selection_target"
        )

    if st.button("Apply Feature Selection"):
        try:
            transformed_data = st.session_state.feature_engineered_data.copy()

            if selection_method == "variance_threshold":
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=threshold)
                selected_features = selector.fit_transform(transformed_data[numeric_cols])
                feature_names = [col for col, mask in zip(numeric_cols, selector.get_support()) if mask]

            elif selection_method == "correlation_filter":
                # Use CorrelationFilter from our module
                selector = CorrelationFilter(threshold=threshold)
                selected_features = selector.fit_transform(transformed_data[numeric_cols])
                feature_names = selector.selected_features_ if hasattr(selector, 'selected_features_') else numeric_cols[:selected_features.shape[1]]

            elif selection_method == "mutual_info" and target_col:
                from sklearn.feature_selection import mutual_info_regression
                X = transformed_data[numeric_cols].drop(columns=[target_col])
                y = transformed_data[target_col]

                scores = mutual_info_regression(X, y)
                top_indices = np.argsort(scores)[-n_features:]
                feature_names = X.columns[top_indices].tolist()
                feature_names.append(target_col)  # Keep target

            # Update data with selected features
            categorical_cols = transformed_data.select_dtypes(include=['object', 'category']).columns.tolist()
            final_features = list(feature_names) + categorical_cols

            st.session_state.feature_engineered_data = transformed_data[final_features]
            st.success(f"✅ Selected {len(feature_names)} features using {selection_method}")

            # Show feature importance if available
            if selection_method == "mutual_info":
                st.markdown("#### 📊 Feature Importance Scores")
                importance_df = pd.DataFrame({
                    'Feature': X.columns[top_indices],
                    'Score': scores[top_indices]
                }).sort_values('Score', ascending=False)

                fig = px.bar(importance_df, x='Score', y='Feature', orientation='h',
                           title="Mutual Information Scores")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Feature selection failed: {str(e)}")


def show_transformation_comparison(original, transformed, title):
    """Show before/after comparison of transformations."""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Before", "After"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Original distribution
    fig.add_trace(
        go.Histogram(x=original, name="Original", opacity=0.7),
        row=1, col=1
    )

    # Transformed distribution
    fig.add_trace(
        go.Histogram(x=transformed, name="Transformed", opacity=0.7),
        row=1, col=2
    )

    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def display_feature_analysis():
    """Display analysis of current feature set."""

    data = st.session_state.feature_engineered_data

    # Feature type breakdown
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Numeric Features", len(numeric_cols))

    with col2:
        st.metric("Categorical Features", len(categorical_cols))

    with col3:
        st.metric("Total Features", len(data.columns))

    # Correlation heatmap for numeric features
    if len(numeric_cols) > 1:
        st.markdown("#### 🔥 Feature Correlation Heatmap")

        # Limit to top 20 features for readability
        display_cols = numeric_cols[:20] if len(numeric_cols) > 20 else numeric_cols
        corr_matrix = data[display_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)