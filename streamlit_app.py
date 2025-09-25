"""
Analytics Toolkit - Streamlit Web Interface
===========================================

A comprehensive web-based interface for the Analytics Toolkit,
providing interactive machine learning capabilities through a user-friendly UI.

Author: Analytics Team
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure Streamlit page
st.set_page_config(
    page_title="Analytics Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cce-dev-account/analytics_toolkit',
        'Report a bug': "https://github.com/cce-dev-account/analytics_toolkit/issues",
        'About': "Analytics Toolkit - Advanced ML with PyTorch Statistical Models"
    }
)

# Import pages
from ui.pages import (
    home,
    data_upload,
    preprocessing,
    feature_engineering,
    model_training,
    results_dashboard,
    model_comparison
)

def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.title("🧬 Analytics Toolkit")
    st.sidebar.markdown("---")

    # Navigation menu
    pages = {
        "🏠 Home": home,
        "📊 Data Upload": data_upload,
        "🔧 Preprocessing": preprocessing,
        "🔬 Feature Engineering": feature_engineering,
        "🧠 Model Training": model_training,
        "📈 Results Dashboard": results_dashboard,
        "⚖️ Model Comparison": model_comparison,
    }

    # Page selection
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=0
    )

    # Display selected page
    pages[selected_page].show()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Info")

    # Check module availability
    modules_status = check_module_availability()
    for module, available in modules_status.items():
        emoji = "✅" if available else "❌"
        st.sidebar.markdown(f"{emoji} {module}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Analytics Toolkit**  \n"
        "*Advanced ML with Statistical Rigor*  \n"
        "🤖 Powered by PyTorch & Streamlit"
    )


def check_module_availability():
    """Check which Analytics Toolkit modules are available."""
    modules = {}

    try:
        import analytics_toolkit.preprocessing
        modules["Preprocessing"] = True
    except ImportError:
        modules["Preprocessing"] = False

    try:
        import analytics_toolkit.pytorch_regression
        modules["PyTorch Regression"] = True
    except ImportError:
        modules["PyTorch Regression"] = False

    try:
        import analytics_toolkit.feature_engineering
        modules["Feature Engineering"] = True
    except ImportError:
        modules["Feature Engineering"] = False

    try:
        import analytics_toolkit.automl
        modules["AutoML"] = True
    except ImportError:
        modules["AutoML"] = False

    try:
        import analytics_toolkit.visualization
        modules["Visualization"] = True
    except ImportError:
        modules["Visualization"] = False

    return modules


if __name__ == "__main__":
    main()# Test modification for Claude Code Index demo
# Phase 2 test comment
