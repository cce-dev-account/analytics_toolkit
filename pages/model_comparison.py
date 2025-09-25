"""
Model comparison page for the Analytics Toolkit.
"""

import streamlit as st

def show():
    """Display the model comparison page."""

    st.title("⚖️ Model Comparison")
    st.markdown("Compare multiple models side-by-side")

    st.info("🚧 Model Comparison UI coming soon! This will include:")
    st.markdown("""
    - **Side-by-side metrics** comparison
    - **Performance benchmarking** across different algorithms
    - **Statistical significance testing** between models
    - **ROC curves** and performance plots
    - **Cross-validation** results
    - **Model selection** recommendations
    """)

    if 'trained_model' in st.session_state:
        st.success("✅ You have a trained model ready for comparison!")

        # Placeholder for future implementation
        if st.button("📊 View Current Model Results"):
            st.switch_page("pages/results_dashboard.py")
    else:
        if st.button("🧠 Train a Model First"):
            st.switch_page("pages/model_training.py")