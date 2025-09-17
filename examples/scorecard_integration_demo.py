"""
Scorecard Integration Demo

This script demonstrates how to use the ScorecardIntegrator to combine
multiple scorecards using optimized weighted sums.

The demo shows:
1. Basic scorecard integration with log-loss optimization
2. Integration with AUC optimization
3. Using weight bounds and fixed weights
4. Getting optimal weights and combined scorecard function
"""

import numpy as np
import pandas as pd
from analytics_toolkit import ScorecardIntegrator
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, roc_auc_score


def create_sample_scorecards():
    """Create sample scorecard data for demonstration."""
    print("Creating sample scorecard data...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Create three different scorecards with varying predictive power
    scorecard_1 = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.2, n_samples)
    scorecard_2 = -0.3 * X[:, 2] + 0.8 * X[:, 3] + np.random.normal(0, 0.15, n_samples)
    scorecard_3 = 0.4 * X[:, 4] - 0.6 * X[:, 5] + np.random.normal(0, 0.25, n_samples)

    # Create DataFrame
    scores_df = pd.DataFrame(
        {
            "credit_score": scorecard_1,
            "behavioral_score": scorecard_2,
            "application_score": scorecard_3,
        }
    )

    return scores_df, y


def demo_basic_integration():
    """Demonstrate basic scorecard integration."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Scorecard Integration with Log-Loss")
    print("=" * 60)

    scores_df, target = create_sample_scorecards()

    # Initialize integrator
    integrator = ScorecardIntegrator(
        scorecard_columns=["credit_score", "behavioral_score", "application_score"],
        objective="log_loss",
    )

    # Fit the integrator
    print("Fitting scorecard integrator...")
    integrator.fit(scores_df, target)

    # Get optimal weights
    weights = integrator.get_weights()
    print("\nOptimal weights:")
    for scorecard, weight in weights.items():
        print(f"  {scorecard}: {weight:.4f}")

    # Make predictions
    predictions = integrator.predict(scores_df)

    # Convert to probabilities for evaluation
    probabilities = 1 / (1 + np.exp(-predictions))

    # Evaluate
    log_loss_score = log_loss(target, probabilities)
    auc_score = roc_auc_score(target, predictions)

    print("\nPerformance:")
    print(f"  Log Loss: {log_loss_score:.4f}")
    print(f"  AUC: {auc_score:.4f}")

    return integrator, scores_df, target


def demo_auc_optimization():
    """Demonstrate AUC-based optimization."""
    print("\n" + "=" * 60)
    print("DEMO 2: Scorecard Integration with AUC Optimization")
    print("=" * 60)

    scores_df, target = create_sample_scorecards()

    # Initialize integrator with AUC objective
    integrator = ScorecardIntegrator(
        scorecard_columns=["credit_score", "behavioral_score", "application_score"],
        objective="auc",
    )

    # Fit the integrator
    print("Fitting scorecard integrator with AUC objective...")
    integrator.fit(scores_df, target)

    # Get optimal weights
    weights = integrator.get_weights()
    print("\nOptimal weights (AUC optimization):")
    for scorecard, weight in weights.items():
        print(f"  {scorecard}: {weight:.4f}")

    # Make predictions and evaluate
    predictions = integrator.predict(scores_df)
    auc_score = roc_auc_score(target, predictions)

    print(f"\nAUC Score: {auc_score:.4f}")


def demo_constrained_optimization():
    """Demonstrate optimization with constraints."""
    print("\n" + "=" * 60)
    print("DEMO 3: Constrained Scorecard Integration")
    print("=" * 60)

    scores_df, target = create_sample_scorecards()

    # Initialize integrator with constraints
    integrator = ScorecardIntegrator(
        scorecard_columns=["credit_score", "behavioral_score", "application_score"],
        objective="log_loss",
        weight_bounds={
            "credit_score": (0.3, 0.6),  # Credit score must be 30-60% of weight
            "behavioral_score": (0.2, 0.5),  # Behavioral score must be 20-50% of weight
        },
        fixed_weights={
            # No fixed weights in this example
        },
    )

    # Fit the integrator
    print("Fitting with weight constraints:")
    print("  credit_score: 30-60% weight")
    print("  behavioral_score: 20-50% weight")

    integrator.fit(scores_df, target)

    # Display results
    summary = integrator.summary()
    print("\nOptimization Summary:")
    print(summary.to_string(index=False))

    # Get combined scorecard function
    combined_scorecard = integrator.get_combined_scorecard()

    # Test the combined function
    test_predictions = combined_scorecard(scores_df[:10])
    print(f"\nFirst 10 combined scores: {test_predictions}")


def demo_fixed_weights():
    """Demonstrate integration with some fixed weights."""
    print("\n" + "=" * 60)
    print("DEMO 4: Integration with Fixed Weights")
    print("=" * 60)

    scores_df, target = create_sample_scorecards()

    # Initialize integrator with one fixed weight
    integrator = ScorecardIntegrator(
        scorecard_columns=["credit_score", "behavioral_score", "application_score"],
        objective="log_loss",
        fixed_weights={"credit_score": 0.5},  # Fix credit score at 50% weight
    )

    # Fit the integrator
    print("Fitting with credit_score fixed at 50% weight...")
    integrator.fit(scores_df, target)

    # Display results
    weights = integrator.get_weights()
    print("\nFinal weights:")
    for scorecard, weight in weights.items():
        status = " (fixed)" if scorecard in integrator.fixed_weights else " (optimized)"
        print(f"  {scorecard}: {weight:.4f}{status}")


def compare_individual_vs_combined():
    """Compare individual scorecards vs combined scorecard performance."""
    print("\n" + "=" * 60)
    print("DEMO 5: Individual vs Combined Performance Comparison")
    print("=" * 60)

    scores_df, target = create_sample_scorecards()

    # Evaluate individual scorecards
    print("Individual scorecard performance:")
    for scorecard in scores_df.columns:
        auc = roc_auc_score(target, scores_df[scorecard])
        print(f"  {scorecard}: AUC = {auc:.4f}")

    # Fit combined scorecard
    integrator = ScorecardIntegrator(
        scorecard_columns=list(scores_df.columns), objective="auc"
    )
    integrator.fit(scores_df, target)

    # Evaluate combined scorecard
    combined_predictions = integrator.predict(scores_df)
    combined_auc = roc_auc_score(target, combined_predictions)

    print("\nCombined scorecard performance:")
    print(f"  Combined AUC = {combined_auc:.4f}")

    # Show improvement
    best_individual = max(
        [roc_auc_score(target, scores_df[col]) for col in scores_df.columns]
    )
    improvement = combined_auc - best_individual
    print(f"  Improvement over best individual: +{improvement:.4f}")


def main():
    """Run all demos."""
    print("Analytics Toolkit - Scorecard Integration Demo")
    print("=" * 60)

    try:
        # Run demos
        demo_basic_integration()
        demo_auc_optimization()
        demo_constrained_optimization()
        demo_fixed_weights()
        compare_individual_vs_combined()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running demo: {e}")
        raise


if __name__ == "__main__":
    main()
