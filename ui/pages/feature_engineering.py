"""
Feature engineering page for the Analytics Toolkit.
"""

import streamlit as st

def show():
    """Display the feature engineering page."""

    st.title("🔬 Feature Engineering")
    st.markdown("Advanced feature transformations and selection")

    if 'processed_data' not in st.session_state:
        st.warning("🔧 Please complete data preprocessing first.")
        if st.button("Go to Preprocessing"):
            st.switch_page("pages/preprocessing.py")
        return

    st.info("🚧 Feature Engineering UI coming soon! This will include:")
    st.markdown("""
    - **Transformations**: Log, Box-Cox, outlier capping
    - **Encoding**: Target encoding, frequency encoding
    - **Selection**: Variance threshold, correlation filter
    - **Interactions**: Automated feature interaction detection
    - **Temporal**: Date/time feature extraction
    """)

    # For now, skip to model training
    if st.button("🧠 Continue to Model Training", type="primary"):
        st.switch_page("pages/model_training.py")