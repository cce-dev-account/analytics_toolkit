"""
Examples demonstrating the feature engineering module capabilities.
"""

import numpy as np
import pandas as pd
from analytics_toolkit.feature_engineering import (
    BinningTransformer,
    DateTimeFeatures,
    FeatureSelector,
    InteractionDetector,
    InteractionGenerator,
    LagFeatures,
    LogTransformer,
    MutualInfoSelector,
    OutlierCapTransformer,
    RollingFeatures,
    TargetEncoder,
)
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def advanced_transformers_example():
    """Demonstrate advanced transformers."""
    print("=== Advanced Transformers Example ===")

    # Create sample data with outliers and skewed distribution
    np.random.seed(42)
    X = np.random.exponential(2, size=(1000, 3))
    X[:50, 0] = 100  # Add outliers

    print(f"Original data shape: {X.shape}")
    print(f"Original data stats:\n{pd.DataFrame(X).describe()}")

    # Apply log transformation to handle skewness
    log_transformer = LogTransformer(method="log1p")
    X_log = log_transformer.fit_transform(X)
    print("\nAfter log transformation - skewness reduced")

    # Cap outliers using IQR method
    outlier_capper = OutlierCapTransformer(method="iqr")
    X_capped = outlier_capper.fit_transform(X)
    print(f"Outliers capped - max values: {np.max(X_capped, axis=0)}")

    # Bin continuous features
    binner = BinningTransformer(strategy="quantile", n_bins=5)
    X_binned = binner.fit_transform(X)
    print(
        f"Binned features - unique values per column: {[len(np.unique(X_binned[:, i])) for i in range(X_binned.shape[1])]}"
    )


def feature_selection_example():
    """Demonstrate feature selection methods."""
    print("\n=== Feature Selection Example ===")

    # Create classification dataset with noise features
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    print(f"Original dataset: {X.shape[1]} features")

    # Comprehensive feature selection
    selector = FeatureSelector(
        methods=["variance", "correlation", "mutual_info"],
        variance_threshold=0.01,
        correlation_threshold=0.95,
        mutual_info_k=10,
    )
    X_selected = selector.fit_transform(X, y)
    print(f"After comprehensive selection: {X_selected.shape[1]} features")
    print(f"Selection history: {selector.selection_history_}")

    # Mutual information selection only
    mi_selector = MutualInfoSelector(k=8)
    X_mi_selected = mi_selector.fit_transform(X, y)
    print(f"Mutual info selection: {X_mi_selected.shape[1]} features")
    print(f"Top MI scores: {mi_selector.scores_[mi_selector.selected_indices_]}")


def categorical_encoding_example():
    """Demonstrate advanced categorical encoding."""
    print("\n=== Categorical Encoding Example ===")

    # Create sample categorical data
    np.random.seed(42)
    categories = ["A", "B", "C", "D", "E"]
    X_cat = np.random.choice(categories, size=(1000, 2))

    # Create target with some correlation to categories
    y_reg = (
        (X_cat[:, 0] == "A") * 5
        + (X_cat[:, 0] == "B") * 3
        + (X_cat[:, 1] == "C") * 2
        + np.random.normal(0, 0.5, 1000)
    )

    print(f"Categorical data shape: {X_cat.shape}")
    print(f"Categories: {categories}")

    # Target encoding with cross-validation
    target_encoder = TargetEncoder(cv=5, smooth=1.0)
    X_encoded = target_encoder.fit_transform(X_cat, y_reg)

    print(f"Target encoded shape: {X_encoded.shape}")
    print(f"Sample encoded values:\n{X_encoded[:10]}")

    # Compare with original means
    df_temp = pd.DataFrame({"cat": X_cat[:, 0], "target": y_reg})
    category_means = df_temp.groupby("cat")["target"].mean()
    print(f"True category means:\n{category_means}")


def interaction_detection_example():
    """Demonstrate interaction detection and generation."""
    print("\n=== Interaction Detection Example ===")

    # Create data with known interactions
    np.random.seed(42)
    X = np.random.randn(500, 4)
    # Add clear interaction: y depends on X0 * X1
    y = 2 * X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 500)

    print(f"Dataset with interactions: {X.shape}")

    # Detect interactions
    detector = InteractionDetector(method="tree_based", max_interactions=5)
    detector.fit(X, y)
    interactions = detector.get_interactions()

    print(f"Detected {len(interactions)} interactions:")
    for i, interaction in enumerate(interactions[:3]):  # Show top 3
        print(
            f"  {i+1}. Features {interaction['features']}: strength={interaction['strength']:.4f}"
        )

    # Generate interaction features
    generator = InteractionGenerator(
        interactions=None,  # Generate all pairwise
        interaction_types=["multiply", "add"],
        max_interactions=10,
    )
    X_with_interactions = generator.fit_transform(X)

    print(f"Original features: {X.shape[1]}")
    print(f"With interactions: {X_with_interactions.shape[1]}")


def temporal_features_example():
    """Demonstrate temporal feature engineering."""
    print("\n=== Temporal Features Example ===")

    # Create time series data
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(365) * 0.5)  # Random walk

    df = pd.DataFrame({"date": dates, "value": values})

    print(f"Time series data: {len(df)} observations")

    # Extract datetime features
    dt_extractor = DateTimeFeatures(
        features=["year", "month", "day", "dayofweek", "is_weekend"],
        cyclical_encoding=True,
    )
    df_with_dt = dt_extractor.fit_transform(df[["date"]])
    print(f"DateTime features extracted: {df_with_dt.shape[1]} features")
    print(f"Feature names: {list(df_with_dt.columns)}")

    # Create lag features
    lag_creator = LagFeatures(
        lags=[1, 7, 30], columns=[0]
    )  # Use column index for numpy array
    values_array = df[["value"]].values
    values_with_lags = lag_creator.fit_transform(values_array)

    print(f"With lag features: {values_with_lags.shape[1]} features")

    # Create rolling features
    rolling_creator = RollingFeatures(
        windows=[7, 30], statistics=["mean", "std"], columns=[0]
    )
    values_with_rolling = rolling_creator.fit_transform(values_array)
    print(f"With rolling features: {values_with_rolling.shape[1]} features")


def end_to_end_pipeline_example():
    """Demonstrate complete feature engineering pipeline."""
    print("\n=== End-to-End Pipeline Example ===")

    # Create mixed dataset
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )

    # Add categorical features
    np.random.seed(42)
    cat_features = np.random.choice(["A", "B", "C"], size=(1000, 2))

    print(
        f"Starting with {X.shape[1]} numerical and {cat_features.shape[1]} categorical features"
    )

    # Step 1: Encode categorical features
    target_encoder = TargetEncoder(cv=3)
    cat_encoded = target_encoder.fit_transform(cat_features, y)

    # Combine features
    X_combined = np.column_stack([X, cat_encoded])
    print(f"After encoding: {X_combined.shape[1]} features")

    # Step 2: Feature selection
    selector = FeatureSelector(methods=["variance", "mutual_info"], mutual_info_k=8)
    X_selected = selector.fit_transform(X_combined, y)
    print(f"After selection: {X_selected.shape[1]} features")

    # Step 3: Generate interactions for top features
    generator = InteractionGenerator(interaction_types=["multiply"], max_interactions=5)
    X_final = generator.fit_transform(X_selected)
    print(f"Final feature set: {X_final.shape[1]} features")

    # Evaluate with classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print("\nModel Performance:")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")


if __name__ == "__main__":
    print("Feature Engineering Examples")
    print("=" * 50)

    advanced_transformers_example()
    feature_selection_example()
    categorical_encoding_example()
    interaction_detection_example()
    temporal_features_example()
    end_to_end_pipeline_example()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
