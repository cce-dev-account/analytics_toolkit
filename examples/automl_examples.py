"""
Comprehensive AutoML examples demonstrating end-to-end machine learning automation.
"""

import numpy as np
import pandas as pd
from analytics_toolkit.automl import (
    AutoMLPipeline,
    AutoModelSelector,
    EnsembleBuilder,
    ExperimentTracker,
    HyperparameterOptimizer,
    ModelRegistry,
    OptimizationConfig,
    PipelineConfig,
)
from sklearn.datasets import (
    fetch_california_housing,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def basic_automl_example():
    """Demonstrate basic AutoML pipeline usage."""
    print("=== Basic AutoML Example ===")

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Convert to DataFrame with meaningful names
    feature_names = [f"feature_{i:02d}" for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)

    print(f"Dataset shape: {X_df.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    # Create AutoML pipeline with custom configuration
    config = PipelineConfig(
        feature_selection=True,
        generate_interactions=True,
        categorical_encoding="auto",
        cv_folds=5,
        random_state=42,
    )

    automl = AutoMLPipeline(config=config, verbose=True)

    # Fit the pipeline
    print("\n🚀 Starting AutoML pipeline...")
    automl.fit(X_df, y)

    # Make predictions
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n📈 Final Results:")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get feature importance
    importance_df = automl.get_feature_importance()
    if importance_df is not None:
        print("\n🔍 Top 5 Most Important Features:")
        print(importance_df.head().to_string(index=False))

    # Get pipeline summary
    summary = automl.summary()
    print("\n📊 Pipeline Summary:")
    print(f"Task Type: {summary['task_type']}")
    print(f"CV Score: {summary['performance_metrics']['cv_mean']:.4f}")

    return automl


def hyperparameter_optimization_example():
    """Demonstrate hyperparameter optimization with Optuna."""
    print("\n=== Hyperparameter Optimization Example ===")

    # Generate regression data
    X, y = make_regression(n_samples=500, n_features=15, noise=0.1, random_state=42)

    print(f"Regression dataset shape: {X.shape}")

    # Set up optimization configuration
    config = OptimizationConfig(
        n_trials=20,  # Reduced for demo
        timeout=60,  # 1 minute timeout
        cv_folds=3,
        direction="maximize",
        random_state=42,
    )

    optimizer = HyperparameterOptimizer(config=config, verbose=True)

    # Optimize Random Forest
    from sklearn.ensemble import RandomForestRegressor

    rf_model = RandomForestRegressor(random_state=42)

    print("\n🔧 Optimizing Random Forest hyperparameters...")
    results = optimizer.optimize(rf_model, X, y)

    print("\n✅ Optimization Results:")
    print(f"Best Score (R²): {results['best_score']:.4f}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Total Trials: {results['n_trials']}")
    print(f"Optimization Time: {results['optimization_time']:.2f} seconds")

    return results


def model_selection_and_comparison_example():
    """Demonstrate automated model selection and comparison."""
    print("\n=== Model Selection and Comparison Example ===")

    # Use California housing dataset for regression
    try:
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        X_df = pd.DataFrame(X, columns=housing.feature_names)

        print(f"California Housing dataset: {X.shape}")
        print(f"Features: {list(housing.feature_names)}")

        # Sample data for faster demo
        X_sample = X_df.sample(n=1000, random_state=42)
        y_sample = y[X_sample.index]

    except Exception:
        # Fallback to synthetic data
        print("Using synthetic regression data...")
        X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        X_sample, y_sample = X_df, y

    # Automated model selection
    selector = AutoModelSelector(
        task_type="regression", cv_folds=3, scoring="r2", random_state=42, verbose=True
    )

    print("\n🤖 Running automated model selection...")
    selector.fit(X_sample, y_sample)

    # Display results
    results_df = selector.get_results_dataframe()
    print("\n📊 Model Comparison Results:")
    print(results_df.round(4).to_string(index=False))

    print(f"\n🏆 Best Model: {selector.results_[0].model_name}")
    print(f"Best Score (R²): {selector.results_[0].mean_score:.4f}")

    return selector


def ensemble_building_example():
    """Demonstrate automated ensemble building."""
    print("\n=== Ensemble Building Example ===")

    # Generate classification data
    X, y = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    print(f"Classification dataset: {X.shape}")

    # Build ensemble
    ensemble_builder = EnsembleBuilder(
        ensemble_methods=["voting"],  # Start with voting ensemble
        n_base_models=3,
        random_state=42,
        verbose=True,
    )

    print("\n🎭 Building ensemble models...")
    ensemble_builder.build_ensemble(X, y)

    # Test ensemble predictions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get predictions from ensemble
    ensemble_pred = ensemble_builder.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    print("\n🏆 Ensemble Performance:")
    print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")

    # Compare with individual base models
    print("\n📊 Base Models Used:")
    for name, model in ensemble_builder.base_models_:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  {name}: {acc:.4f}")

    return ensemble_builder


def experiment_tracking_example():
    """Demonstrate experiment tracking and model registry."""
    print("\n=== Experiment Tracking Example ===")

    # Initialize experiment tracking
    tracker = ExperimentTracker(
        tracking_uri="./demo_experiments", experiment_name="automl_demo", auto_log=True
    )

    # Create model registry
    registry = ModelRegistry(registry_path="./demo_model_registry")

    # Generate data
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Run multiple experiments
    models_to_test = [
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]


    run_results = []

    for model_name, model in models_to_test:
        print(f"\n🧪 Running experiment: {model_name}")

        # Start tracking run
        run_id = tracker.start_run(run_name=f"{model_name}_experiment")

        # Log model parameters
        params = model.get_params()
        tracker.log_params(params)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        # Log metrics
        tracker.log_metrics(
            {
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
            }
        )

        # Log trained model
        tracker.log_model(model, artifact_path="trained_model")

        # Register best models in registry
        if test_score > 0.8:  # Arbitrary threshold
            version = registry.register_model(
                model=model,
                name=f"demo_{model_name.lower().replace(' ', '_')}",
                description=f"Demo {model_name} model",
                run_id=run_id,
            )
            print(f"  📦 Registered model version {version}")

        # End run
        tracker.end_run()

        run_results.append(
            {"model": model_name, "run_id": run_id, "test_accuracy": test_score}
        )

        print(f"  ✅ {model_name} - Test Accuracy: {test_score:.4f}")

    # Find best run
    best_run = tracker.get_best_run("test_accuracy", experiment_name="automl_demo")
    if best_run:
        print(f"\n🏆 Best Run: {best_run.model_name}")
        print(f"Best Test Accuracy: {best_run.metrics['test_accuracy']:.4f}")
        print(f"Run ID: {best_run.run_id}")

    # List registered models
    registered_models = registry.list_models()
    print(f"\n📦 Registered Models ({len(registered_models)}):")
    for model_info in registered_models:
        print(f"  - {model_info['name']}: {model_info['description']}")

    return tracker, registry


def end_to_end_automl_workflow():
    """Demonstrate complete end-to-end AutoML workflow."""
    print("\n=== Complete End-to-End AutoML Workflow ===")

    # 1. Data preparation
    print("\n1️⃣ Data Preparation")
    X, y = make_classification(
        n_samples=1200,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42,
    )

    # Add some categorical features
    np.random.seed(42)
    categorical_features = pd.DataFrame(
        {
            "category_A": np.random.choice(["cat1", "cat2", "cat3", "cat4"], 1200),
            "category_B": np.random.choice(["typeX", "typeY"], 1200),
        }
    )

    numerical_features = pd.DataFrame(
        X, columns=[f"num_feature_{i:02d}" for i in range(15)]
    )

    # Combine features
    X_combined = pd.concat([numerical_features, categorical_features], axis=1)

    print(f"Final dataset shape: {X_combined.shape}")
    print(f"Numerical features: {len(numerical_features.columns)}")
    print(f"Categorical features: {len(categorical_features.columns)}")

    # 2. Initialize experiment tracking
    print("\n2️⃣ Initialize Experiment Tracking")
    tracker = ExperimentTracker(
        tracking_uri="./complete_automl_demo",
        experiment_name="end_to_end_workflow",
        auto_log=True,
    )

    run_id = tracker.start_run(run_name="complete_automl_pipeline")

    # 3. AutoML Pipeline
    print("\n3️⃣ AutoML Pipeline Configuration")
    config = PipelineConfig(
        handle_missing=True,
        feature_selection=True,
        feature_selection_k=0.8,
        generate_interactions=True,
        max_interactions=15,
        categorical_encoding="auto",
        scaling=True,
        handle_outliers=True,
        cv_folds=5,
        random_state=42,
    )

    # Log configuration
    config_dict = {
        "feature_selection": config.feature_selection,
        "feature_selection_k": config.feature_selection_k,
        "generate_interactions": config.generate_interactions,
        "max_interactions": config.max_interactions,
        "categorical_encoding": config.categorical_encoding,
        "cv_folds": config.cv_folds,
    }
    tracker.log_params(config_dict)

    # 4. Train pipeline
    print("\n4️⃣ Training AutoML Pipeline")
    automl = AutoMLPipeline(config=config, verbose=True)
    automl.fit(X_combined, y)

    # 5. Model evaluation
    print("\n5️⃣ Model Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get predictions and probabilities
    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test)

    # Calculate metrics
    from sklearn.metrics import roc_auc_score

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba[:, 1])

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")

    # Log metrics
    tracker.log_metrics(
        {
            "final_test_accuracy": accuracy,
            "final_auc_score": auc_score,
            "cv_score": automl.performance_metrics_["cv_mean"],
        }
    )

    # 6. Feature analysis
    print("\n6️⃣ Feature Analysis")
    importance_df = automl.get_feature_importance()
    if importance_df is not None:
        print("Top 10 Most Important Features:")
        top_features = importance_df.head(10)
        print(top_features.to_string(index=False))

        # Log top features as parameters
        for i, row in top_features.iterrows():
            tracker.log_param(f"top_feature_{i+1}", row["feature"])

    # 7. Model comparison (optional)
    print("\n7️⃣ Model Comparison")
    selector = AutoModelSelector(
        include_models=["Random Forest", "Gradient Boosting", "Logistic Regression"],
        cv_folds=3,
        verbose=False,
    )
    selector.fit(X_train, y_train)

    comparison_df = selector.get_results_dataframe()
    print("Model Comparison Results:")
    print(comparison_df.round(4))

    # 8. Save results
    print("\n8️⃣ Saving Results")

    # Save the AutoML pipeline
    pipeline_path = "./complete_automl_pipeline.pkl"
    automl.save_pipeline(pipeline_path)
    tracker.log_artifact(pipeline_path, "automl_pipeline.pkl")

    # Log the trained pipeline as a model
    tracker.log_model(automl.pipeline, "complete_pipeline")

    # End experiment
    tracker.end_run()

    print("\n✅ Complete workflow finished!")
    print("📊 Final Results Summary:")
    print(f"  - Test Accuracy: {accuracy:.4f}")
    print(f"  - AUC Score: {auc_score:.4f}")
    print(f"  - CV Score: {automl.performance_metrics_['cv_mean']:.4f}")
    print(f"  - Run ID: {run_id}")

    return {
        "automl_pipeline": automl,
        "test_accuracy": accuracy,
        "auc_score": auc_score,
        "run_id": run_id,
        "tracker": tracker,
    }


if __name__ == "__main__":
    print("AutoML Examples - Comprehensive Machine Learning Automation")
    print("=" * 70)

    try:
        # Run examples
        automl_result = basic_automl_example()
        optimization_result = hyperparameter_optimization_example()
        selection_result = model_selection_and_comparison_example()
        ensemble_result = ensemble_building_example()
        tracking_result = experiment_tracking_example()
        workflow_result = end_to_end_automl_workflow()

        print("\n" + "=" * 70)
        print("All AutoML examples completed successfully!")
        print("\nKey Takeaways:")
        print("- AutoML pipelines can handle mixed data types automatically")
        print("- Hyperparameter optimization significantly improves performance")
        print("- Model selection helps identify the best algorithm for your data")
        print("- Ensemble methods often outperform individual models")
        print("- Experiment tracking is crucial for reproducible ML workflows")
        print("- End-to-end automation streamlines the entire ML process")

    except ImportError as e:
        print(f"\n⚠️ Some examples require additional dependencies: {e}")
        print("Install with: pip install optuna xgboost lightgbm")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
# Testing Phase 2 incremental updates
