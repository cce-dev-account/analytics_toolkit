"""
Advanced Regression Features Demonstration

This example showcases the new advanced regression capabilities in Analytics Toolkit:
- Regularization paths with automatic alpha selection
- Polynomial regression with automatic degree selection
- Robust regression for outlier handling
- Non-linear transformations (B-splines, RBF, Fourier)
- Advanced feature engineering pipelines
"""

import warnings

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Import Analytics Toolkit components
from analytics_toolkit.pytorch_regression import (
    BSplineTransformer,
    FourierTransformer,
    LinearRegression,
    PolynomialRegression,
    RadialBasisTransformer,
    RegularizationPath,
    RobustRegression,
)


def demo_regularization_path():
    """Demonstrate automatic regularization parameter selection."""
    print("=" * 60)
    print("🎯 REGULARIZATION PATH DEMONSTRATION")
    print("=" * 60)

    # Generate data with noise
    X, y = make_regression(n_samples=200, n_features=15, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # L2 Regularization Path
    print("\n🔍 Computing L2 Regularization Path...")
    l2_path = RegularizationPath(penalty="l2", n_alphas=20, cv=5, random_state=42)
    l2_path.fit(X_train, y_train)

    print(f"✅ Optimal L2 alpha: {l2_path.alpha_optimal_:.6f}")
    print(f"📊 Best CV score: {np.max(l2_path.cv_scores_):.4f}")

    # Test best model
    y_pred = l2_path.best_model_.predict(X_test)
    test_score = l2_path.best_model_.score(X_test, y_test)
    print(f"🎯 Test R² score: {test_score:.4f}")

    # L1 Regularization Path (Lasso)
    print("\n🔍 Computing L1 Regularization Path...")
    l1_path = RegularizationPath(penalty="l1", n_alphas=10, cv=3, random_state=42)
    l1_path.fit(X_train, y_train)

    print(f"✅ Optimal L1 alpha: {l1_path.alpha_optimal_:.6f}")
    print(f"📊 Best CV score: {np.max(l1_path.cv_scores_):.4f}")

    # Check sparsity
    n_nonzero = np.sum(np.abs(l1_path.best_model_.coef_.detach().cpu().numpy()) > 1e-5)
    print(f"🎯 Non-zero coefficients: {n_nonzero}/{X.shape[1]} (sparsity achieved!)")

    return l2_path, l1_path


def demo_polynomial_regression():
    """Demonstrate polynomial regression with automatic degree selection."""
    print("\n\n" + "=" * 60)
    print("📐 POLYNOMIAL REGRESSION DEMONSTRATION")
    print("=" * 60)

    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-2, 2, 150).reshape(-1, 1)
    y = (
        0.5 * X.ravel() ** 3
        + 0.3 * X.ravel() ** 2
        - 0.2 * X.ravel()
        + 0.1 * np.random.randn(150)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Dataset: Non-linear function with cubic, quadratic, and linear terms")

    # Automatic degree selection
    print("\n🔍 Finding optimal polynomial degree...")
    poly_auto = PolynomialRegression(
        max_degree=6, cv=5, penalty="l2", alpha=0.01, random_state=42
    )
    poly_auto.fit(X_train, y_train)

    print(f"✅ Optimal degree: {poly_auto.degree_}")
    print(
        f"📊 CV scores by degree: {[f'{score:.4f}' for score in poly_auto.cv_scores_]}"
    )

    # Test performance
    test_score = poly_auto.score(X_test, y_test)
    print(f"🎯 Test R² score: {test_score:.4f}")

    # Compare with fixed degrees
    degrees = [1, 2, 3, 5]
    print("\n📊 Comparison with fixed degrees:")
    for degree in degrees:
        poly_fixed = PolynomialRegression(degree=degree, penalty="l2", alpha=0.01)
        poly_fixed.fit(X_train, y_train)
        score = poly_fixed.score(X_test, y_test)
        print(f"   Degree {degree}: R² = {score:.4f}")

    return poly_auto


def demo_robust_regression():
    """Demonstrate robust regression for outlier handling."""
    print("\n\n" + "=" * 60)
    print("🛡️ ROBUST REGRESSION DEMONSTRATION")
    print("=" * 60)

    # Generate data with outliers
    np.random.seed(42)
    X, y = make_regression(n_samples=150, n_features=3, noise=0.1, random_state=42)

    # Add outliers
    n_outliers = 15
    outlier_indices = np.random.choice(150, n_outliers, replace=False)
    y[outlier_indices] += np.random.normal(0, 5, n_outliers)  # Strong outliers

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset: {X.shape[0]} samples with {n_outliers} outliers")

    # Regular linear regression
    print("\n🔍 Standard Linear Regression:")
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_score = ols.score(X_test, y_test)
    print(f"   R² score: {ols_score:.4f}")

    # Robust regression
    print("\n🛡️ Huber Robust Regression:")
    huber = RobustRegression(method="huber", epsilon=1.35, max_iter=100)
    huber.fit(X_train, y_train)
    huber_score = huber.score(X_test, y_test)
    print(f"   R² score: {huber_score:.4f}")

    improvement = ((huber_score - ols_score) / abs(ols_score)) * 100
    if improvement > 0:
        print(f"✅ Robust regression improved performance by {improvement:.1f}%")
    else:
        print("📊 Both methods performed similarly (robust method is more stable)")

    return huber, ols


def demo_nonlinear_transformations():
    """Demonstrate non-linear transformations."""
    print("\n\n" + "=" * 60)
    print("🌊 NON-LINEAR TRANSFORMATIONS DEMONSTRATION")
    print("=" * 60)

    # Generate complex non-linear data
    np.random.seed(42)
    X = np.linspace(0, 4 * np.pi, 200).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.5 * np.sin(3 * X.ravel()) + 0.1 * np.random.randn(200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Dataset: Sinusoidal function with harmonics")

    # B-Spline transformation
    print("\n🎯 B-Spline Transformation:")
    bspline_pipeline = Pipeline(
        [
            ("bspline", BSplineTransformer(n_knots=10, degree=3)),
            ("linear", LinearRegression()),
        ]
    )
    bspline_pipeline.fit(X_train, y_train)
    bspline_score = bspline_pipeline.score(X_test, y_test)
    print(f"   R² score: {bspline_score:.4f}")

    # RBF transformation
    print("\n🎯 Radial Basis Function Transformation:")
    rbf_pipeline = Pipeline(
        [
            ("rbf", RadialBasisTransformer(n_centers=15, kernel="gaussian")),
            ("linear", LinearRegression()),
        ]
    )
    rbf_pipeline.fit(X_train, y_train)
    rbf_score = rbf_pipeline.score(X_test, y_test)
    print(f"   R² score: {rbf_score:.4f}")

    # Fourier transformation
    print("\n🎯 Fourier Transformation:")
    fourier_pipeline = Pipeline(
        [
            ("fourier", FourierTransformer(n_frequencies=8)),
            ("linear", LinearRegression()),
        ]
    )
    fourier_pipeline.fit(X_train, y_train)
    fourier_score = fourier_pipeline.score(X_test, y_test)
    print(f"   R² score: {fourier_score:.4f}")

    # Standard polynomial for comparison
    print("\n📊 Standard Polynomial (degree 6) for comparison:")
    poly_pipeline = Pipeline(
        [
            ("poly", PolynomialRegression(degree=6, penalty="l2", alpha=0.01)),
        ]
    )
    poly_pipeline.fit(X_train, y_train)
    poly_score = poly_pipeline.score(X_test, y_test)
    print(f"   R² score: {poly_score:.4f}")

    # Find best method
    methods = {
        "B-Splines": bspline_score,
        "RBF": rbf_score,
        "Fourier": fourier_score,
        "Polynomial": poly_score,
    }
    best_method = max(methods, key=methods.get)
    print(
        f"\n✅ Best method for this data: {best_method} (R² = {methods[best_method]:.4f})"
    )

    return bspline_pipeline, rbf_pipeline, fourier_pipeline


def demo_advanced_pipeline():
    """Demonstrate combining multiple advanced techniques."""
    print("\n\n" + "=" * 60)
    print("🚀 ADVANCED PIPELINE DEMONSTRATION")
    print("=" * 60)

    # Generate challenging dataset
    X, y = make_regression(n_samples=300, n_features=8, noise=0.2, random_state=42)

    # Add outliers
    outlier_indices = np.random.choice(300, 20, replace=False)
    y[outlier_indices] += np.random.normal(0, 3, 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, with outliers")

    # Advanced pipeline: RBF + Regularization + Robustness
    print("\n🎯 Advanced Pipeline: RBF → Regularization Path → Robust Regression")

    # Step 1: Non-linear transformation
    rbf_transformer = RadialBasisTransformer(n_centers=20, kernel="gaussian")
    X_train_transformed = rbf_transformer.fit_transform(X_train)
    X_test_transformed = rbf_transformer.transform(X_test)

    print(f"   After RBF: {X_train_transformed.shape[1]} features")

    # Step 2: Find optimal regularization
    reg_path = RegularizationPath(penalty="l2", n_alphas=10, cv=3)
    reg_path.fit(X_train_transformed, y_train)
    optimal_alpha = reg_path.alpha_optimal_

    print(f"   Optimal regularization: α = {optimal_alpha:.6f}")

    # Step 3: Robust regression with optimal regularization
    robust_model = RobustRegression(method="huber", alpha=optimal_alpha, max_iter=50)
    robust_model.fit(X_train_transformed, y_train)

    # Evaluate
    y_pred = robust_model.predict(X_test_transformed)
    final_score = robust_model.score(X_test_transformed, y_test)

    print(f"   Final R² score: {final_score:.4f}")

    # Compare with simple linear regression
    print("\n📊 Comparison with standard linear regression:")
    simple_model = LinearRegression()
    simple_model.fit(X_train, y_train)
    simple_score = simple_model.score(X_test, y_test)
    print(f"   Standard Linear R²: {simple_score:.4f}")

    improvement = ((final_score - simple_score) / abs(simple_score)) * 100
    if improvement > 0:
        print(f"✅ Advanced pipeline improved performance by {improvement:.1f}%")

    return robust_model, simple_model


def main():
    """Run all demonstrations."""
    print("*** ANALYTICS TOOLKIT - ADVANCED REGRESSION FEATURES ***")
    print("🚀 Comprehensive Demonstration")
    print()

    try:
        # Run all demonstrations
        l2_path, l1_path = demo_regularization_path()
        poly_model = demo_polynomial_regression()
        huber_model, ols_model = demo_robust_regression()
        bspline, rbf, fourier = demo_nonlinear_transformations()
        advanced_model, simple_model = demo_advanced_pipeline()

        # Final summary
        print("\n\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("🎯 Successfully demonstrated:")
        print("   • Regularization Paths (L1/L2) with automatic alpha selection")
        print("   • Polynomial Regression with automatic degree selection")
        print("   • Robust Regression (Huber) for outlier handling")
        print("   • Non-linear Transformations (B-Splines, RBF, Fourier)")
        print("   • Advanced Pipeline combining multiple techniques")
        print()
        print("🚀 The Analytics Toolkit now provides state-of-the-art")
        print("   regression capabilities with automatic hyperparameter")
        print("   selection and robust statistical inference!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("   Please check that all dependencies are installed correctly.")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All demonstrations completed successfully!")
    else:
        print("\n⚠️  Some demonstrations failed. Check the error messages above.")
