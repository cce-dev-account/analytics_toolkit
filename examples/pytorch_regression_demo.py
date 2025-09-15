"""
Demonstration of PyTorch Statistical Regression Module.

This example shows how to use the PyTorch-based regression module with
comprehensive statistical inference capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import our PyTorch regression module
from analytics_toolkit.pytorch_regression import LinearRegression, LogisticRegression


def linear_regression_example():
    """Demonstrate linear regression with statistical inference."""
    print("=" * 60)
    print("LINEAR REGRESSION EXAMPLE")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)

    # Create feature names
    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    # Add a categorical variable
    df["category"] = np.random.choice(["A", "B", "C"], size=len(df))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    # Fit model
    print("Fitting linear regression model...")
    model = LinearRegression(penalty="l2", alpha=0.01)
    model.fit(X_train, y_train)

    # Display statistical summary
    print("\nStatistical Summary:")
    print(model.summary())

    # Make predictions and evaluate
    r2_score = model.score(X_test, y_test)
    print(f"\nTest R² Score: {r2_score:.4f}")

    # Confidence intervals
    conf_int = model.conf_int()
    print("\nConfidence Intervals (95%):")
    print(conf_int)

    # Prediction intervals
    print("\nPrediction intervals for first 5 test samples:")
    y_pred_interval, lower, upper = model.predict_interval(X_test[:5])
    for i in range(5):
        print(
            f"Sample {i+1}: {y_pred_interval[i]:.3f} [{lower[i]:.3f}, {upper[i]:.3f}]"
        )

    return model, X_test, y_test


def logistic_regression_example():
    """Demonstrate logistic regression with statistical inference."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION EXAMPLE")
    print("=" * 60)

    # Generate synthetic classification data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000, n_features=6, n_redundant=0, n_informative=4, random_state=42
    )

    # Create DataFrame with mixed data types
    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    # Add categorical variable
    df["region"] = np.random.choice(["North", "South", "East", "West"], size=len(df))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    # Fit model
    print("Fitting logistic regression model...")
    model = LogisticRegression(penalty="l2", alpha=0.1, max_iter=1000)
    model.fit(X_train, y_train)

    # Display statistical summary
    print("\nStatistical Summary:")
    print(model.summary())

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)

    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Show some probability predictions
    print("\nProbability predictions for first 5 test samples:")
    for i in range(5):
        print(
            f"Sample {i+1}: Class={y_pred[i]}, "
            f"P(class=0)={y_proba[i,0]:.3f}, P(class=1)={y_proba[i,1]:.3f}"
        )

    # Decision function
    decision_scores = model.decision_function(X_test[:5])
    print("\nDecision function (log-odds) for first 5 samples:")
    for i, score in enumerate(decision_scores):
        print(f"Sample {i+1}: {score:.3f}")

    return model, X_test, y_test


def performance_comparison():
    """Compare performance with scikit-learn."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON WITH SCIKIT-LEARN")
    print("=" * 60)

    import time

    from sklearn.linear_model import LinearRegression as SklearnLinear
    from sklearn.linear_model import LogisticRegression as SklearnLogistic

    # Generate larger dataset for timing
    X, y = make_regression(n_samples=10000, n_features=20, random_state=42)

    # Linear regression timing
    print("\nLinear Regression Timing:")

    # Our implementation
    start_time = time.time()
    our_model = LinearRegression()
    our_model.fit(X, y)
    our_pred = our_model.predict(X)
    our_time = time.time() - start_time

    # Scikit-learn
    start_time = time.time()
    sklearn_model = SklearnLinear()
    sklearn_model.fit(X, y)
    sklearn_pred = sklearn_model.predict(X)
    sklearn_time = time.time() - start_time

    print(f"Our implementation: {our_time:.4f} seconds")
    print(f"Scikit-learn:       {sklearn_time:.4f} seconds")
    print(f"Speedup factor:     {sklearn_time/our_time:.2f}x")

    # Check prediction accuracy
    correlation = np.corrcoef(our_pred, sklearn_pred)[0, 1]
    print(f"Prediction correlation: {correlation:.6f}")

    # Classification data for logistic regression
    X_clf, y_clf = make_classification(n_samples=5000, n_features=15, random_state=42)

    print("\nLogistic Regression Timing:")

    # Our implementation
    start_time = time.time()
    our_log_model = LogisticRegression(max_iter=1000)
    our_log_model.fit(X_clf, y_clf)
    our_log_pred = our_log_model.predict_proba(X_clf)
    our_log_time = time.time() - start_time

    # Scikit-learn
    start_time = time.time()
    sklearn_log_model = SklearnLogistic(max_iter=1000)
    sklearn_log_model.fit(X_clf, y_clf)
    sklearn_log_pred = sklearn_log_model.predict_proba(X_clf)
    sklearn_log_time = time.time() - start_time

    print(f"Our implementation: {our_log_time:.4f} seconds")
    print(f"Scikit-learn:       {sklearn_log_time:.4f} seconds")
    print(f"Speedup factor:     {sklearn_log_time/our_log_time:.2f}x")

    # Check probability prediction accuracy
    prob_diff = np.mean(np.abs(our_log_pred - sklearn_log_pred))
    print(f"Mean probability difference: {prob_diff:.6f}")


def statistical_features_demo():
    """Demonstrate advanced statistical features."""
    print("\n" + "=" * 60)
    print("ADVANCED STATISTICAL FEATURES")
    print("=" * 60)

    # Generate data with some multicollinearity
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = X1 + 0.5 * X2 + 0.1 * np.random.randn(n_samples)  # Collinear

    X = np.column_stack([X1, X2, X3])
    y = 2 * X1 - 1.5 * X2 + 0.5 * X3 + 0.1 * np.random.randn(n_samples)

    # Calculate VIF
    from analytics_toolkit.pytorch_regression.utils import calculate_vif

    print("Variance Inflation Factors:")
    vif_results = calculate_vif(X, ["X1", "X2", "X3"])
    print(vif_results)

    # Fit model and examine residuals
    model = LinearRegression()
    model.fit(X, y)

    print(f"\nModel R²: {model.r_squared_:.4f}")
    print(f"Adjusted R²: {model.adj_r_squared_:.4f}")

    # Residual analysis
    residuals = model.get_residuals(X, y, residual_type="standardized")
    print("\nStandardized Residuals Summary:")
    print(f"Mean: {np.mean(residuals):.6f}")
    print(f"Std:  {np.std(residuals):.6f}")
    print(f"Min:  {np.min(residuals):.3f}")
    print(f"Max:  {np.max(residuals):.3f}")

    # Check for outliers (|residual| > 2)
    outliers = np.where(np.abs(residuals) > 2)[0]
    print(f"Potential outliers (|residual| > 2): {len(outliers)} samples")

    return model


def regularization_demo():
    """Demonstrate regularization effects."""
    print("\n" + "=" * 60)
    print("REGULARIZATION COMPARISON")
    print("=" * 60)

    # Generate data with many features
    X, y = make_regression(n_samples=100, n_features=50, noise=0.1, random_state=42)

    models = {
        "No Regularization": LinearRegression(penalty="none"),
        "L1 (Lasso)": LinearRegression(penalty="l1", alpha=0.1),
        "L2 (Ridge)": LinearRegression(penalty="l2", alpha=0.1),
    }

    results = {}

    for name, model in models.items():
        model.fit(X, y)
        r2 = model.score(X, y)
        n_nonzero = np.sum(np.abs(model.coef_.cpu().numpy()) > 1e-6)
        coef_norm = np.linalg.norm(model.coef_.cpu().numpy())

        results[name] = {
            "R²": r2,
            "Non-zero coeffs": n_nonzero,
            "Coeff norm": coef_norm,
        }

    print("\nRegularization Effects:")
    print(f"{'Method':<20} {'R²':<8} {'Non-zero':<10} {'||β||':<10}")
    print("-" * 50)
    for name, stats in results.items():
        print(
            f"{name:<20} {stats['R²']:<8.4f} {stats['Non-zero']:<10} {stats['Coeff norm']:<10.3f}"
        )


def main():
    """Run all examples."""
    print("PyTorch Statistical Regression Module Demo")
    print("=========================================")

    try:
        # Linear regression example
        linear_model, X_test_lin, y_test_lin = linear_regression_example()

        # Logistic regression example
        logistic_model, X_test_log, y_test_log = logistic_regression_example()

        # Performance comparison
        performance_comparison()

        # Statistical features
        statistical_features_demo()

        # Regularization demo
        regularization_demo()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
