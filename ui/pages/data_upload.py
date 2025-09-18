"""
Data upload and preview page for the Analytics Toolkit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Optional

def show():
    """Display the data upload and preview page."""

    st.title("📊 Data Upload & Preview")
    st.markdown("Upload your dataset and explore its characteristics")

    # Upload section
    st.markdown("### 📁 Upload Dataset")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset in CSV or Excel format"
        )

    with col2:
        st.markdown("**File Requirements:**")
        st.markdown("- Max size: 200MB")
        st.markdown("- Formats: CSV, Excel")
        st.markdown("- Headers recommended")

    # Check if data exists in session state or uploaded
    data = None
    data_source = "No data"

    if uploaded_file is not None:
        data = load_uploaded_file(uploaded_file)
        data_source = f"Uploaded: {uploaded_file.name}"

        if data is not None:
            st.session_state.uploaded_data = data
            st.session_state.data_source = data_source

    elif 'uploaded_data' in st.session_state:
        data = st.session_state.uploaded_data
        data_source = st.session_state.get('data_source', 'Session data')

    # Display data if available
    if data is not None:
        display_data_overview(data, data_source)
        display_data_exploration(data)
        display_data_quality_check(data)
    else:
        display_no_data_message()


def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load data from uploaded file."""

    try:
        with st.spinner("Loading data..."):
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for CSV files
                try:
                    data = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    data = pd.read_csv(uploaded_file, encoding='latin-1')

            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None

            st.success(f"✅ Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            return data

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def display_data_overview(data: pd.DataFrame, data_source: str):
    """Display basic data overview."""

    st.markdown("---")
    st.markdown("### 📋 Dataset Overview")

    # Basic info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Rows", f"{len(data):,}")

    with col2:
        st.metric("📈 Columns", len(data.columns))

    with col3:
        st.metric("💾 Memory", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    with col4:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("❓ Missing Data", f"{missing_pct:.1f}%")

    # Data source info
    st.info(f"📂 **Data Source:** {data_source}")

    # Data preview
    st.markdown("### 👀 Data Preview")

    # Preview options
    col1, col2, col3 = st.columns(3)

    with col1:
        show_rows = st.selectbox("Rows to display", [5, 10, 20, 50], index=1)

    with col2:
        preview_type = st.selectbox("Preview type", ["Head", "Tail", "Random sample"])

    with col3:
        if st.button("🔄 Refresh Preview"):
            st.rerun()

    # Display preview based on selection
    if preview_type == "Head":
        preview_data = data.head(show_rows)
    elif preview_type == "Tail":
        preview_data = data.tail(show_rows)
    else:
        preview_data = data.sample(min(show_rows, len(data)), random_state=42)

    st.dataframe(preview_data, width='stretch')

    # Column information
    st.markdown("### 📊 Column Information")

    col_info = []
    for col in data.columns:
        col_info.append({
            'Column': col,
            'Type': str(data[col].dtype),
            'Non-Null': f"{data[col].count():,}",
            'Missing': f"{data[col].isnull().sum():,}",
            'Missing %': f"{(data[col].isnull().sum() / len(data) * 100):.1f}%",
            'Unique': f"{data[col].nunique():,}"
        })

    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, width='stretch')


def display_data_exploration(data: pd.DataFrame):
    """Display interactive data exploration tools."""

    st.markdown("---")
    st.markdown("### 🔍 Data Exploration")

    # Tabs for different exploration views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "❓ Missing Data", "🎯 Target Analysis"])

    with tab1:
        display_distributions(data)

    with tab2:
        display_correlations(data)

    with tab3:
        display_missing_data_analysis(data)

    with tab4:
        display_target_analysis(data)


def display_distributions(data: pd.DataFrame):
    """Display distribution analysis."""

    st.markdown("#### 📊 Feature Distributions")

    # Select columns for distribution analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        if numeric_cols:
            selected_numeric = st.selectbox("Select numeric column", numeric_cols)

            if selected_numeric:
                # Create histogram
                fig = px.histogram(
                    data,
                    x=selected_numeric,
                    marginal="box",
                    title=f"Distribution of {selected_numeric}"
                )
                st.plotly_chart(fig, width='stretch')

                # Basic statistics
                st.markdown("**Statistics:**")
                stats = data[selected_numeric].describe()
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                    st.metric("Std Dev", f"{stats['std']:.2f}")

                with col_b:
                    st.metric("Min", f"{stats['min']:.2f}")
                    st.metric("Max", f"{stats['max']:.2f}")

    with col2:
        if categorical_cols:
            selected_categorical = st.selectbox("Select categorical column", categorical_cols)

            if selected_categorical:
                # Value counts
                value_counts = data[selected_categorical].value_counts().head(10)

                # Create bar chart
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 10 Values in {selected_categorical}"
                )
                fig.update_xaxes(title=selected_categorical)
                fig.update_yaxes(title="Count")
                st.plotly_chart(fig, width='stretch')

                # Category statistics
                st.markdown("**Category Statistics:**")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Unique Values", data[selected_categorical].nunique())

                with col_b:
                    st.metric("Most Frequent", value_counts.index[0])


def display_correlations(data: pd.DataFrame):
    """Display correlation analysis."""

    st.markdown("#### 🔗 Correlation Analysis")

    numeric_data = data.select_dtypes(include=[np.number])

    if len(numeric_data.columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    # Compute correlation matrix
    corr_matrix = numeric_data.corr()

    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="Correlation Matrix Heatmap"
    )

    fig.update_layout(
        width=700,
        height=600
    )

    st.plotly_chart(fig, width='stretch')

    # Show highly correlated pairs
    st.markdown("#### 🔥 Highly Correlated Features")

    threshold = st.slider("Correlation threshold", 0.5, 0.95, 0.8, 0.05)

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })

    if high_corr_pairs:
        corr_df = pd.DataFrame(high_corr_pairs)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(corr_df, width='stretch')
    else:
        st.info(f"No feature pairs found with correlation >= {threshold}")


def display_missing_data_analysis(data: pd.DataFrame):
    """Display missing data analysis."""

    st.markdown("#### ❓ Missing Data Analysis")

    # Missing data summary
    missing_summary = []
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            missing_summary.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing Percentage': (missing_count / len(data)) * 100,
                'Data Type': str(data[col].dtype)
            })

    if missing_summary:
        missing_df = pd.DataFrame(missing_summary)
        missing_df = missing_df.sort_values('Missing Count', ascending=False)

        # Missing data chart
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Percentage',
            title="Missing Data by Column"
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, width='stretch')

        # Missing data table
        st.dataframe(missing_df, width='stretch')

        # Missing data heatmap (for smaller datasets)
        if len(data) <= 1000:
            st.markdown("#### 🔥 Missing Data Pattern")
            missing_pattern = data.isnull().astype(int)

            fig = px.imshow(
                missing_pattern.T,
                color_continuous_scale=['white', 'red'],
                title="Missing Data Pattern (Red = Missing)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

    else:
        st.success("🎉 No missing data found in the dataset!")


def display_target_analysis(data: pd.DataFrame):
    """Display target variable analysis."""

    st.markdown("#### 🎯 Target Variable Analysis")

    # Let user select potential target column
    target_candidates = data.columns.tolist()

    if not target_candidates:
        st.warning("No columns available for target analysis.")
        return

    selected_target = st.selectbox(
        "Select potential target variable",
        target_candidates,
        help="Choose the column you want to predict"
    )

    if selected_target:
        target_data = data[selected_target]

        # Determine if target is numeric or categorical
        if pd.api.types.is_numeric_dtype(target_data):
            # Numeric target - regression analysis
            st.markdown(f"**Regression Analysis for '{selected_target}'**")

            col1, col2 = st.columns(2)

            with col1:
                # Target distribution
                fig = px.histogram(
                    data,
                    x=selected_target,
                    marginal="box",
                    title=f"Target Distribution: {selected_target}"
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Target statistics
                stats = target_data.describe()
                st.markdown("**Target Statistics:**")

                st.metric("Mean", f"{stats['mean']:.3f}")
                st.metric("Std Dev", f"{stats['std']:.3f}")
                st.metric("Min", f"{stats['min']:.3f}")
                st.metric("Max", f"{stats['max']:.3f}")

        else:
            # Categorical target - classification analysis
            st.markdown(f"**Classification Analysis for '{selected_target}'**")

            col1, col2 = st.columns(2)

            with col1:
                # Class distribution
                class_counts = target_data.value_counts()

                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title=f"Class Distribution: {selected_target}"
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Classification metrics
                st.markdown("**Class Statistics:**")

                st.metric("Number of Classes", target_data.nunique())
                st.metric("Most Frequent Class", class_counts.index[0])
                st.metric("Class Balance Ratio", f"{class_counts.min() / class_counts.max():.2f}")

                # Show class distribution table
                st.markdown("**Class Counts:**")
                class_df = pd.DataFrame({
                    'Class': class_counts.index,
                    'Count': class_counts.values,
                    'Percentage': (class_counts.values / len(target_data) * 100).round(1)
                })
                st.dataframe(class_df, width='stretch')

        # Store target selection in session state
        if st.button(f"✅ Set '{selected_target}' as Target Variable", type="primary"):
            st.session_state.target_column = selected_target
            st.success(f"Target variable set to '{selected_target}'. Ready for preprocessing!")


def display_data_quality_check(data: pd.DataFrame):
    """Display data quality assessment."""

    st.markdown("---")
    st.markdown("### 🔍 Data Quality Assessment")

    quality_issues = []

    # Check for duplicates
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        quality_issues.append({
            'Issue': 'Duplicate Rows',
            'Count': duplicate_count,
            'Severity': 'Medium',
            'Description': f'{duplicate_count} duplicate rows found'
        })

    # Check for missing data
    missing_cols = data.columns[data.isnull().any()].tolist()
    if missing_cols:
        quality_issues.append({
            'Issue': 'Missing Data',
            'Count': len(missing_cols),
            'Severity': 'Low',
            'Description': f'Missing values in {len(missing_cols)} columns'
        })

    # Check for high cardinality categorical columns
    for col in data.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = data[col].nunique() / len(data)
        if unique_ratio > 0.5:
            quality_issues.append({
                'Issue': 'High Cardinality',
                'Count': data[col].nunique(),
                'Severity': 'Medium',
                'Description': f"Column '{col}' has {data[col].nunique()} unique values ({unique_ratio:.1%})"
            })

    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() == 1]
    if constant_cols:
        quality_issues.append({
            'Issue': 'Constant Columns',
            'Count': len(constant_cols),
            'Severity': 'High',
            'Description': f'Columns with single value: {", ".join(constant_cols)}'
        })

    if quality_issues:
        st.warning(f"⚠️ Found {len(quality_issues)} data quality issues")

        quality_df = pd.DataFrame(quality_issues)
        st.dataframe(quality_df, width='stretch')

        st.markdown("**Recommendations:**")
        for issue in quality_issues:
            if issue['Issue'] == 'Duplicate Rows':
                st.markdown("- Consider removing duplicate rows in preprocessing")
            elif issue['Issue'] == 'Missing Data':
                st.markdown("- Configure missing value handling in preprocessing")
            elif issue['Issue'] == 'High Cardinality':
                st.markdown("- Consider target encoding or feature selection for high cardinality features")
            elif issue['Issue'] == 'Constant Columns':
                st.markdown("- Remove constant columns as they provide no information")

    else:
        st.success("✅ No major data quality issues detected!")

    # Overall quality score
    quality_score = max(0, 100 - len(quality_issues) * 15)
    st.metric("📊 Overall Data Quality Score", f"{quality_score}/100")


def display_no_data_message():
    """Display message when no data is available."""

    st.info("👆 Please upload a dataset to begin analysis")

    # Sample data generation
    st.markdown("### 🔧 Or Generate Sample Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Regression Dataset**")
        if st.button("Generate Regression Data", key="reg_data"):
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42)

            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(8)])
            df['target'] = y
            df['category'] = np.random.choice(['A', 'B', 'C'], size=1000)

            st.session_state.uploaded_data = df
            st.session_state.data_source = "Generated Regression Data"
            st.rerun()

    with col2:
        st.markdown("**Classification Dataset**")
        if st.button("Generate Classification Data", key="class_data"):
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=1000, n_features=10, n_informative=8,
                n_redundant=2, random_state=42
            )

            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(10)])
            df['target'] = y
            df['category'] = np.random.choice(['Group_1', 'Group_2', 'Group_3'], size=1000)

            st.session_state.uploaded_data = df
            st.session_state.data_source = "Generated Classification Data"
            st.rerun()