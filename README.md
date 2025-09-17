# Analytics Toolkit

A comprehensive Python toolkit for data analytics and machine learning with PyTorch support.

## Features

- **🔧 Advanced Preprocessing**: Intelligent data cleaning, encoding, and scaling
- **🔬 Feature Engineering**: Comprehensive transformations, selection, and interaction detection
- **🧠 PyTorch Statistical Models**: Linear & logistic regression with full statistical inference
- **🤖 AutoML Pipeline**: Automated model selection and hyperparameter optimization
- **📊 Rich Visualizations**: Interactive plots, model diagnostics, and performance metrics
- **🌐 Streamlit Web Interface**: Complete web-based UI for all capabilities
- **📈 Model Analysis**: Statistical inference, confidence intervals, and model comparison
- **🏦 Scorecard Integration**: Financial modeling and credit scoring capabilities
- **⚡ Production Ready**: GPU acceleration, model persistence, and robust error handling

## Installation

1. Clone the repository
2. Install Poetry if not already installed
3. Install dependencies:

```bash
poetry install --with dev
```

## Project Structure

```
analytics_toolkit/
├── src/analytics_toolkit/           # Main package
│   ├── __init__.py                 # Package initialization
│   ├── utils.py                    # Data utilities
│   ├── preprocessing.py            # Data preprocessing
│   ├── models.py                   # Basic ML models and PyTorch utilities
│   ├── scorecard_integration.py    # Financial scorecard modeling
│   ├── pytorch_regression/         # Advanced PyTorch statistical models
│   │   ├── linear.py              # Linear regression with inference
│   │   ├── logistic.py            # Logistic regression with inference
│   │   ├── advanced.py            # Polynomial, regularization, robust models
│   │   ├── stats.py               # Statistical inference utilities
│   │   └── transforms.py          # Advanced transformations
│   ├── feature_engineering/        # Comprehensive feature engineering
│   │   ├── encoding.py            # Categorical encoding strategies
│   │   ├── selection.py           # Feature selection methods
│   │   ├── transformers.py        # Data transformations
│   │   ├── interactions.py        # Feature interactions
│   │   └── temporal.py            # Time-based features
│   ├── automl/                     # Automated ML pipeline
│   │   ├── model_selection.py     # Automated model selection
│   │   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   │   ├── pipeline_builder.py    # ML pipeline construction
│   │   └── experiment_tracking.py # Experiment management
│   └── visualization/              # Rich plotting and analysis
│       ├── model_evaluation.py    # Model diagnostic plots
│       ├── statistical.py         # Statistical visualizations
│       ├── interactive.py         # Interactive dashboards
│       ├── data_profiling.py      # Data exploration plots
│       └── themes.py              # Consistent styling
├── ui/                             # Streamlit web interface
├── tests/                          # Comprehensive test suite
├── examples/                       # Usage examples and demos
├── docs/                           # Documentation
└── streamlit_app.py               # Web application entry point
```

## Quick Start

### 🌐 Web Interface (Recommended)
```bash
# Start the Streamlit web application
poetry run streamlit run streamlit_app.py
```

### 📊 Python API Usage

```python
import analytics_toolkit as at
import pandas as pd

# Load and explore data
data = at.load_data('your_data.csv')
description = at.describe_data(data)

# Advanced preprocessing with feature engineering
from analytics_toolkit.feature_engineering import AdvancedScaler, TargetEncoder
from analytics_toolkit.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
encoder = TargetEncoder()
scaler = AdvancedScaler(method='robust')

# PyTorch statistical regression with full inference
from analytics_toolkit.pytorch_regression import LinearRegression

model = LinearRegression(fit_intercept=True, compute_stats=True)
model.fit(X, y)

# Get comprehensive results with p-values, confidence intervals
summary = model.summary()
print(summary)

# AutoML pipeline for automated model selection
from analytics_toolkit.automl import AutoMLPipeline

automl = AutoMLPipeline(task_type='regression', time_limit=300)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

# Rich visualizations
from analytics_toolkit.visualization import ModelEvaluationPlotter

plotter = ModelEvaluationPlotter()
plotter.plot_regression_diagnostics(model, X_test, y_test)
```

## 🚀 Key Capabilities

### 🧠 PyTorch Statistical Regression
- **Full Statistical Inference**: Standard errors, t-statistics, p-values, confidence intervals
- **Advanced Models**: Linear, logistic, polynomial, regularized (L1/L2/ElasticNet), robust regression
- **GPU Acceleration**: CUDA support for large datasets
- **Sklearn Compatible**: Drop-in replacement with enhanced statistics

### 🔬 Feature Engineering
- **Smart Transformations**: Log, Box-Cox, outlier handling, advanced scaling
- **Categorical Encoding**: Target encoding, frequency encoding, Bayesian methods
- **Feature Selection**: Correlation, mutual information, importance-based selection
- **Interaction Detection**: Automated feature interaction discovery
- **Temporal Features**: Date/time extraction, seasonality detection, lag features

### 🤖 AutoML Pipeline
- **Automated Model Selection**: Compare multiple algorithms automatically
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Pipeline Construction**: End-to-end ML pipeline building
- **Experiment Tracking**: Comprehensive experiment management
- **Time-Bounded Training**: Configurable training time limits

### 📊 Rich Visualizations
- **Model Diagnostics**: Residual plots, Q-Q plots, leverage analysis
- **Performance Metrics**: ROC curves, precision-recall, feature importance
- **Data Profiling**: Distribution analysis, correlation heatmaps, missing value patterns
- **Interactive Dashboards**: Streamlit-powered web interface

### 🏦 Financial Modeling
- **Scorecard Integration**: Credit scoring and risk modeling
- **Weighted Optimization**: Financial constraint handling
- **Regulatory Compliance**: Statistical validation for financial models

## Development

### Quick Start

Set up your development environment:

```bash
# Linux/macOS
./scripts/dev.sh dev-setup

# Windows PowerShell
.\scripts\dev.ps1 dev-setup

# Windows Command Prompt
scripts\dev.bat dev-setup

# Or using Make (if available)
make dev-setup
```

### Development Commands

We provide cross-platform development scripts for common tasks:

| Command | Description |
|---------|-------------|
| `dev-setup` | Complete development environment setup |
| `install` | Install dependencies |
| `test` | Run tests with coverage |
| `lint` | Run all linting tools (black, ruff, mypy) |
| `format` | Format code with black and ruff |
| `clean` | Clean up cache and build files |
| `jupyter` | Start Jupyter Lab |
| `build` | Build package |
| `security` | Run security scans |

### Platform-Specific Usage

**Linux/macOS:**
```bash
./scripts/dev.sh [command]
make [command]  # if make is available
```

**Windows:**
```powershell
# PowerShell (recommended)
.\scripts\dev.ps1 [command]

# Command Prompt
scripts\dev.bat [command]
```

### Manual Commands

If you prefer using Poetry directly:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest --cov=src/analytics_toolkit

# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy src/
```

## Dependencies

- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- And more (see pyproject.toml)

## 🚀 Releases

This project uses automated releases triggered by version tags. See [Release Documentation](docs/RELEASE.md) for complete setup instructions.

### Quick Release

```bash
# Using release script (recommended)
./scripts/release.sh release 1.0.0

# Using make commands
make release VERSION=1.0.0

# Manual tag creation
git tag v1.0.0
git push origin v1.0.0
```

### Release Types

- **Stable**: `v1.0.0` → Publishes to PyPI
- **Pre-release**: `v1.0.0-alpha.1` → Publishes to Test PyPI
- **Hotfix**: `v1.0.1` → Critical bug fixes

For detailed release instructions and troubleshooting, see:
- [Release Documentation](docs/RELEASE.md)
- [Release Checklist](RELEASE_CHECKLIST.md)