"""
Data preprocessing page for the Analytics Toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def show():
    """Display the preprocessing page."""

    st.title("🔧 Data Preprocessing")
    st.markdown("Configure data preprocessing pipeline")

    # Check if data is available
    if 'uploaded_data' not in st.session_state:
        st.warning("📊 No data available. Please upload data first.")
        if st.button("📁 Go to Data Upload"):
            st.switch_page("pages/data_upload.py")
        return

    data = st.session_state.uploaded_data
    st.success(f"✅ Using dataset: {st.session_state.get('data_source', 'Unknown')}")

    # Data overview
    display_preprocessing_overview(data)

    # Preprocessing configuration
    preprocessing_config = configure_preprocessing(data)

    # Apply preprocessing
    if st.button("🚀 Apply Preprocessing", type="primary", use_container_width=True):
        processed_data = apply_preprocessing(data, preprocessing_config)

        if processed_data is not None:
            st.session_state.processed_data = processed_data
            st.session_state.preprocessing_config = preprocessing_config
            st.success("✅ Preprocessing completed successfully!")

            # Show before/after comparison
            display_preprocessing_results(data, processed_data)


def display_preprocessing_overview(data: pd.DataFrame):
    """Display preprocessing overview and current data state."""

    st.markdown("### 📋 Current Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Rows", f"{len(data):,}")

    with col2:
        st.metric("📈 Columns", len(data.columns))

    with col3:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("❓ Missing Data", f"{missing_pct:.1f}%")

    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")

    # Column types breakdown
    st.markdown("### 📊 Column Type Distribution")

    col_types = {
        'Numeric': len(data.select_dtypes(include=[np.number]).columns),
        'Categorical': len(data.select_dtypes(include=['object', 'category']).columns),
        'Datetime': len(data.select_dtypes(include=['datetime64']).columns),
        'Boolean': len(data.select_dtypes(include=['bool']).columns)
    }

    # Remove types with 0 columns
    col_types = {k: v for k, v in col_types.items() if v > 0}

    if col_types:
        fig = px.pie(
            values=list(col_types.values()),
            names=list(col_types.keys()),
            title="Column Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def configure_preprocessing(data: pd.DataFrame) -> Dict[str, Any]:
    """Configure preprocessing parameters."""

    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing Configuration")

    config = {}

    # Target column selection
    st.markdown("#### 🎯 Target Variable")

    target_candidates = data.columns.tolist()
    current_target = st.session_state.get('target_column', None)

    if current_target and current_target in target_candidates:
        default_idx = target_candidates.index(current_target)
    else:
        default_idx = len(target_candidates) - 1  # Default to last column

    config['target_column'] = st.selectbox(
        "Select target variable",
        target_candidates,
        index=default_idx,
        help="Choose the column you want to predict"
    )

    # Feature columns (exclude target)
    feature_columns = [col for col in data.columns if col != config['target_column']]
    numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()

    # Display feature breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"🔢 **Numeric Features:** {len(numeric_features)}")
        if numeric_features:
            with st.expander("View Numeric Features"):
                for feat in numeric_features:
                    st.markdown(f"• {feat}")

    with col2:
        st.info(f"🏷️ **Categorical Features:** {len(categorical_features)}")
        if categorical_features:
            with st.expander("View Categorical Features"):
                for feat in categorical_features:
                    unique_count = data[feat].nunique()
                    st.markdown(f"• {feat} ({unique_count} unique)")

    # Scaling configuration
    st.markdown("#### 📏 Scaling Configuration")

    config['scaling_method'] = st.selectbox(
        "Scaling method for numeric features",
        ['standard', 'minmax', 'robust', 'none'],
        index=0,
        help={
            'standard': 'StandardScaler: mean=0, std=1',
            'minmax': 'MinMaxScaler: scale to [0,1]',
            'robust': 'RobustScaler: median-based, less sensitive to outliers',
            'none': 'No scaling applied'
        }
    )

    # Missing value handling
    st.markdown("#### ❓ Missing Value Handling")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numeric Features**")
        config['numeric_missing_strategy'] = st.selectbox(
            "Strategy for numeric missing values",
            ['median', 'mean', 'mode', 'drop'],
            index=0,
            help="How to handle missing values in numeric columns"
        )

    with col2:
        st.markdown("**Categorical Features**")
        config['categorical_missing_strategy'] = st.selectbox(
            "Strategy for categorical missing values",
            ['mode', 'constant', 'drop'],
            index=0,
            help="How to handle missing values in categorical columns"
        )

        if config['categorical_missing_strategy'] == 'constant':
            config['categorical_fill_value'] = st.text_input(
                "Fill value for categorical",
                value="Unknown",
                help="Constant value to fill missing categorical data"
            )

    # Encoding configuration
    if categorical_features:
        st.markdown("#### 🏷️ Categorical Encoding")

        config['encoding_method'] = st.selectbox(
            "Encoding method for categorical features",
            ['label', 'onehot', 'target', 'frequency'],
            index=0,
            help={
                'label': 'Label Encoding: Simple numeric mapping',
                'onehot': 'One-Hot Encoding: Binary columns for each category',
                'target': 'Target Encoding: Mean target value per category',
                'frequency': 'Frequency Encoding: Count-based encoding'
            }
        )

        # One-hot encoding specific options
        if config['encoding_method'] == 'onehot':
            config['onehot_drop_first'] = st.checkbox(
                "Drop first category (avoid multicollinearity)",
                value=True,
                help="Drop first category to avoid perfect multicollinearity"
            )

            config['max_categories'] = st.slider(
                "Maximum categories per feature",
                min_value=5,
                max_value=50,
                value=10,
                help="Features with more categories will be label encoded instead"
            )

    # Outlier handling
    if numeric_features:
        st.markdown("#### 🎯 Outlier Handling")

        config['outlier_method'] = st.selectbox(
            "Outlier detection and handling",
            ['none', 'iqr', 'z_score', 'isolation_forest'],
            index=0,
            help={
                'none': 'No outlier handling',
                'iqr': 'Interquartile Range method',
                'z_score': 'Z-score method (assumes normal distribution)',
                'isolation_forest': 'Isolation Forest algorithm'
            }
        )

        if config['outlier_method'] != 'none':
            config['outlier_action'] = st.selectbox(
                "Action for detected outliers",
                ['cap', 'remove', 'transform'],
                index=0,
                help={
                    'cap': 'Cap outliers to threshold values',
                    'remove': 'Remove outlier rows',
                    'transform': 'Apply log transformation'
                }
            )

            if config['outlier_method'] == 'iqr':
                config['iqr_factor'] = st.slider(
                    "IQR multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Higher values are more permissive"
                )

            elif config['outlier_method'] == 'z_score':
                config['z_threshold'] = st.slider(
                    "Z-score threshold",
                    min_value=2.0,
                    max_value=4.0,
                    value=3.0,
                    step=0.1,
                    help="Standard deviations from mean"
                )

    # Feature selection
    st.markdown("#### 🎯 Feature Selection (Optional)")

    config['feature_selection'] = st.checkbox(
        "Enable automatic feature selection",
        value=False,
        help="Apply statistical tests to select most relevant features"
    )

    if config['feature_selection']:
        config['selection_method'] = st.selectbox(
            "Feature selection method",
            ['variance_threshold', 'chi2', 'mutual_info', 'correlation'],
            index=0
        )

        config['n_features'] = st.slider(
            "Number of features to select",
            min_value=1,
            max_value=len(feature_columns),
            value=min(10, len(feature_columns)),
            help="Maximum number of features to keep"
        )

    # Train-test split configuration
    st.markdown("#### 🔄 Train-Test Split")

    col1, col2 = st.columns(2)

    with col1:
        config['test_size'] = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Fraction of data to use for testing"
        )

    with col2:
        config['random_state'] = st.number_input(
            "Random state",
            min_value=1,
            max_value=1000,
            value=42,
            help="Seed for reproducible results"
        )

    # Stratification for classification
    target_data = data[config['target_column']]
    if not pd.api.types.is_numeric_dtype(target_data) or target_data.nunique() <= 20:
        config['stratify'] = st.checkbox(
            "Stratified split (recommended for classification)",
            value=True,
            help="Maintain class proportions in train/test split"
        )

    return config


def apply_preprocessing(data: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Apply preprocessing based on configuration."""

    try:
        # Import required modules
        from analytics_toolkit.preprocessing import DataPreprocessor, create_train_test_split
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        with st.spinner("🔄 Applying preprocessing..."):
            # Create a copy of the data
            processed_data = data.copy()

            # Separate features and target
            target_col = config['target_column']
            feature_cols = [col for col in processed_data.columns if col != target_col]

            # Handle missing values first
            processed_data = handle_missing_values(processed_data, config, feature_cols)

            # Handle outliers if configured
            if config.get('outlier_method', 'none') != 'none':
                processed_data = handle_outliers(processed_data, config, feature_cols)

            # Initialize preprocessor
            preprocessor = DataPreprocessor()

            # Apply preprocessing
            X_processed, y_processed = preprocessor.fit_transform(
                processed_data,
                target_column=target_col,
                scaling_method=config['scaling_method']
            )

            # Create train-test split
            stratify = config.get('stratify', False)
            X_train, X_test, y_train, y_test = create_train_test_split(
                X_processed,
                y_processed,
                test_size=config['test_size'],
                random_state=config['random_state'],
                stratify=stratify
            )

            # Store results in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor

            # Combine processed data for return
            result_data = X_processed.copy()
            result_data[target_col] = y_processed

            return result_data

    except ImportError as e:
        st.error(f"❌ Required module not available: {str(e)}")
        st.info("Some preprocessing features require the full Analytics Toolkit installation.")
        return None

    except Exception as e:
        st.error(f"❌ Preprocessing failed: {str(e)}")
        return None


def handle_missing_values(data: pd.DataFrame, config: Dict[str, Any], feature_cols: list) -> pd.DataFrame:
    """Handle missing values based on configuration."""

    processed_data = data.copy()

    # Numeric features
    numeric_cols = processed_data[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        strategy = config.get('numeric_missing_strategy', 'median')

        for col in numeric_cols:
            if processed_data[col].isnull().any():
                if strategy == 'median':
                    fill_value = processed_data[col].median()
                elif strategy == 'mean':
                    fill_value = processed_data[col].mean()
                elif strategy == 'mode':
                    fill_value = processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 0
                elif strategy == 'drop':
                    processed_data = processed_data.dropna(subset=[col])
                    continue

                processed_data[col] = processed_data[col].fillna(fill_value)

    # Categorical features
    categorical_cols = processed_data[feature_cols].select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        strategy = config.get('categorical_missing_strategy', 'mode')

        for col in categorical_cols:
            if processed_data[col].isnull().any():
                if strategy == 'mode':
                    mode_val = processed_data[col].mode()
                    fill_value = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                elif strategy == 'constant':
                    fill_value = config.get('categorical_fill_value', 'Unknown')
                elif strategy == 'drop':
                    processed_data = processed_data.dropna(subset=[col])
                    continue

                processed_data[col] = processed_data[col].fillna(fill_value)

    return processed_data


def handle_outliers(data: pd.DataFrame, config: Dict[str, Any], feature_cols: list) -> pd.DataFrame:
    """Handle outliers based on configuration."""

    processed_data = data.copy()
    numeric_cols = processed_data[feature_cols].select_dtypes(include=[np.number]).columns

    method = config.get('outlier_method', 'none')
    action = config.get('outlier_action', 'cap')

    if method == 'iqr':
        factor = config.get('iqr_factor', 1.5)

        for col in numeric_cols:
            Q1 = processed_data[col].quantile(0.25)
            Q3 = processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            if action == 'cap':
                processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
            elif action == 'remove':
                mask = (processed_data[col] >= lower_bound) & (processed_data[col] <= upper_bound)
                processed_data = processed_data[mask]

    elif method == 'z_score':
        threshold = config.get('z_threshold', 3.0)

        for col in numeric_cols:
            z_scores = np.abs((processed_data[col] - processed_data[col].mean()) / processed_data[col].std())

            if action == 'cap':
                mean_val = processed_data[col].mean()
                std_val = processed_data[col].std()
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
            elif action == 'remove':
                mask = z_scores <= threshold
                processed_data = processed_data[mask]

    return processed_data


def display_preprocessing_results(original_data: pd.DataFrame, processed_data: pd.DataFrame):
    """Display before/after preprocessing comparison."""

    st.markdown("---")
    st.markdown("### 📊 Preprocessing Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{len(processed_data):,}",
            delta=f"{len(processed_data) - len(original_data):+,}"
        )

    with col2:
        st.metric(
            "Columns",
            len(processed_data.columns),
            delta=len(processed_data.columns) - len(original_data.columns)
        )

    with col3:
        orig_missing = (original_data.isnull().sum().sum() / (len(original_data) * len(original_data.columns))) * 100
        proc_missing = (processed_data.isnull().sum().sum() / (len(processed_data) * len(processed_data.columns))) * 100
        st.metric(
            "Missing Data %",
            f"{proc_missing:.1f}%",
            delta=f"{proc_missing - orig_missing:.1f}%"
        )

    with col4:
        orig_memory = original_data.memory_usage(deep=True).sum() / 1024**2
        proc_memory = processed_data.memory_usage(deep=True).sum() / 1024**2
        st.metric(
            "Memory (MB)",
            f"{proc_memory:.1f}",
            delta=f"{proc_memory - orig_memory:.1f}"
        )

    # Show train/test split info
    if all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
        st.markdown("#### 🔄 Train-Test Split")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"**Training Set**\n{len(st.session_state.X_train):,} samples")

        with col2:
            st.info(f"**Test Set**\n{len(st.session_state.X_test):,} samples")

        with col3:
            test_ratio = len(st.session_state.X_test) / (len(st.session_state.X_train) + len(st.session_state.X_test))
            st.info(f"**Test Ratio**\n{test_ratio:.1%}")

    # Data preview
    st.markdown("#### 👀 Processed Data Preview")
    st.dataframe(processed_data.head(10), use_container_width=True)

    # Ready for next step
    st.success("✅ Data preprocessing completed! Ready for feature engineering or model training.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔬 Go to Feature Engineering", use_container_width=True):
            st.switch_page("pages/feature_engineering.py")

    with col2:
        if st.button("🧠 Go to Model Training", use_container_width=True):
            st.switch_page("pages/model_training.py")