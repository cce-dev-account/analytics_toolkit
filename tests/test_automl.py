"""
Comprehensive tests for AutoML module.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analytics_toolkit.automl.experiment_tracking import (
    ExperimentTracker,
    ModelRegistry,
    RunMetrics,
)
from analytics_toolkit.automl.hyperparameter_tuning import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptunaOptimizer,
)
from analytics_toolkit.automl.model_selection import (
    AutoModelSelector,
    EnsembleBuilder,
    ModelComparison,
)
from analytics_toolkit.automl.pipeline_builder import (
    AutoMLPipeline,
    DataTypeInference,
    PipelineConfig,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


class TestPipelineBuilder:
    def test_data_type_inference(self):
        """Test DataTypeInference with mixed data types."""
        # Create mixed dataset
        df = pd.DataFrame(
            {
                "numerical": [1.0, 2.0, 3.0, 4.0, 5.0],
                "categorical": ["A", "B", "A", "C", "B"],
                "datetime": pd.date_range("2023-01-01", periods=5),
                "high_cardinality": ["cat1", "cat2", "cat3", "cat4", "cat5"],
            }
        )

        analyzer = DataTypeInference()
        analysis = analyzer.analyze_dataset(df)

        assert analysis["n_samples"] == 5
        assert analysis["n_features"] == 4
        assert "numerical" in analysis["numerical_columns"]
        assert "categorical" in analysis["categorical_columns"]
        assert "datetime" in analysis["datetime_columns"]

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()

        assert config.handle_missing == True
        assert config.feature_selection == True
        assert config.scaling == True
        assert config.cv_folds == 5
        assert config.random_state == 42

    def test_automl_pipeline_classification(self):
        """Test AutoMLPipeline with classification data."""
        # Generate classification dataset
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Create and fit pipeline
        config = PipelineConfig(cv_folds=3)  # Use fewer folds for speed
        pipeline = AutoMLPipeline(config=config, verbose=False)
        pipeline.fit(X_df, y)

        assert pipeline.task_type_ == "classification"
        assert pipeline.pipeline is not None
        assert pipeline.performance_metrics_ is not None

        # Test predictions
        predictions = pipeline.predict(X_df[:10])
        assert len(predictions) == 10

        # Test probability predictions
        probabilities = pipeline.predict_proba(X_df[:10])
        assert probabilities.shape == (10, 2)

    def test_automl_pipeline_regression(self):
        """Test AutoMLPipeline with regression data."""
        # Generate regression dataset
        X, y = make_regression(n_samples=100, n_features=8, noise=0.1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])

        # Create and fit pipeline
        config = PipelineConfig(cv_folds=3)
        pipeline = AutoMLPipeline(config=config, verbose=False)
        pipeline.fit(X_df, y)

        assert pipeline.task_type_ == "regression"
        assert pipeline.pipeline is not None

        # Test predictions
        predictions = pipeline.predict(X_df[:10])
        assert len(predictions) == 10

    def test_pipeline_with_categorical_data(self):
        """Test pipeline with mixed numerical and categorical features."""
        # Create mixed dataset
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "num1": np.random.randn(n_samples),
                "num2": np.random.randn(n_samples),
                "cat1": np.random.choice(["A", "B", "C"], n_samples),
                "cat2": np.random.choice(["X", "Y"], n_samples),
            }
        )

        y = np.random.choice([0, 1], n_samples)

        config = PipelineConfig(categorical_encoding="target", cv_folds=3)
        pipeline = AutoMLPipeline(config=config, verbose=False)
        pipeline.fit(df, y)

        assert pipeline.task_type_ == "classification"
        predictions = pipeline.predict(df[:10])
        assert len(predictions) == 10

    def test_pipeline_save_load(self):
        """Test pipeline serialization."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_df = pd.DataFrame(X)

        pipeline = AutoMLPipeline(verbose=False)
        pipeline.fit(X_df, y)

        # Save pipeline
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pipeline.save_pipeline(tmp.name)

            # Load pipeline
            loaded_pipeline = AutoMLPipeline.load_pipeline(tmp.name)

            # Test loaded pipeline
            original_pred = pipeline.predict(X_df[:10])
            loaded_pred = loaded_pipeline.predict(X_df[:10])

            np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])

        pipeline = AutoMLPipeline(verbose=False)
        pipeline.fit(X_df, y)

        importance_df = pipeline.get_feature_importance()
        if importance_df is not None:
            assert "feature" in importance_df.columns
            assert "importance" in importance_df.columns
            assert len(importance_df) > 0

    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_df = pd.DataFrame(X)

        pipeline = AutoMLPipeline(verbose=False)
        pipeline.fit(X_df, y)

        summary = pipeline.summary()
        assert "task_type" in summary
        assert "performance_metrics" in summary
        assert "config" in summary


class TestHyperparameterTuning:
    def test_optimization_config(self):
        """Test OptimizationConfig default values."""
        config = OptimizationConfig()

        assert config.n_trials == 100
        assert config.cv_folds == 5
        assert config.direction == "maximize"
        assert config.random_state == 42

    def test_hyperparameter_optimizer_basic(self):
        """Test basic hyperparameter optimization."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Use fast config for testing
        config = OptimizationConfig(n_trials=5, cv_folds=3, timeout=30)
        optimizer = HyperparameterOptimizer(config=config, verbose=False)

        model = RandomForestClassifier(random_state=42)
        results = optimizer.optimize(model, X, y)

        assert "best_params" in results
        assert "best_score" in results
        assert isinstance(results["best_score"], float)

    def test_custom_search_space(self):
        """Test optimization with custom search space."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        config = OptimizationConfig(n_trials=3, cv_folds=3)
        optimizer = HyperparameterOptimizer(config=config, verbose=False)

        model = RandomForestClassifier(random_state=42)
        search_space = {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10]}

        results = optimizer.optimize(model, X, y, search_space=search_space)

        assert "best_params" in results
        assert "n_estimators" in results["best_params"]
        assert "max_depth" in results["best_params"]

    def test_regression_optimization(self):
        """Test optimization with regression."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)

        config = OptimizationConfig(n_trials=3, cv_folds=3)
        optimizer = HyperparameterOptimizer(config=config, verbose=False)

        model = RandomForestRegressor(random_state=42)
        results = optimizer.optimize(model, X, y)

        assert "best_params" in results
        assert "best_score" in results

    @pytest.mark.skipif(
        True,  # Skip by default since optuna might not be installed
        reason="Optuna not available or test too slow",
    )
    def test_optuna_optimizer(self):
        """Test OptunaOptimizer (only run if Optuna is available)."""
        try:
            from analytics_toolkit.automl.hyperparameter_tuning import OPTUNA_AVAILABLE

            if not OPTUNA_AVAILABLE:
                pytest.skip("Optuna not available")

            X, y = make_classification(n_samples=100, n_features=5, random_state=42)

            config = OptimizationConfig(n_trials=5, cv_folds=3)
            optimizer = OptunaOptimizer(config=config, verbose=False)

            model = RandomForestClassifier(random_state=42)
            results = optimizer.optimize(model, X, y)

            assert "study" in results
            assert results["study"] is not None

        except ImportError:
            pytest.skip("Optuna not available")


class TestModelSelection:
    def test_auto_model_selector_classification(self):
        """Test AutoModelSelector with classification."""
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)

        selector = AutoModelSelector(
            include_models=["Random Forest", "Logistic Regression"],
            cv_folds=3,
            verbose=False,
        )
        selector.fit(X, y)

        assert len(selector.results_) == 2
        assert selector.best_model_ is not None
        assert selector.best_score_ is not None
        assert selector.task_type_ == "classification"

    def test_auto_model_selector_regression(self):
        """Test AutoModelSelector with regression."""
        X, y = make_regression(n_samples=100, n_features=8, random_state=42)

        selector = AutoModelSelector(
            include_models=["Random Forest", "Ridge"], cv_folds=3, verbose=False
        )
        selector.fit(X, y)

        assert len(selector.results_) == 2
        assert selector.task_type_ == "regression"

    def test_model_selector_exclude_models(self):
        """Test model exclusion."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        selector = AutoModelSelector(
            exclude_models=["SVM", "Neural Network"], cv_folds=3, verbose=False
        )
        selector.fit(X, y)

        # Check that excluded models are not in results
        model_names = [result.model_name for result in selector.results_]
        assert "SVM" not in model_names
        assert "Neural Network" not in model_names

    def test_results_dataframe(self):
        """Test results DataFrame generation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        selector = AutoModelSelector(
            include_models=["Random Forest", "Logistic Regression"],
            cv_folds=3,
            verbose=False,
        )
        selector.fit(X, y)

        df = selector.get_results_dataframe()
        assert not df.empty
        assert "model" in df.columns
        assert "mean_score" in df.columns
        assert len(df) == 2

    def test_model_comparison(self):
        """Test ModelComparison class."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        models = {
            "rf": RandomForestClassifier(n_estimators=10, random_state=42),
            "lr": LogisticRegression(random_state=42, max_iter=1000),
        }

        comparator = ModelComparison(cv_folds=3, random_state=42)
        comparison_df = comparator.compare_models(models, X, y)

        assert not comparison_df.empty
        assert "model_1" in comparison_df.columns
        assert "model_2" in comparison_df.columns
        assert "p_value" in comparison_df.columns

    def test_ensemble_builder(self):
        """Test EnsembleBuilder."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        base_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000),
        ]

        builder = EnsembleBuilder(
            ensemble_methods=["voting"], verbose=False, random_state=42
        )
        builder.build_ensemble(X, y, base_models=base_models)

        assert builder.best_ensemble_ is not None

        # Test predictions
        predictions = builder.predict(X[:10])
        assert len(predictions) == 10


class TestExperimentTracking:
    def test_run_metrics_creation(self):
        """Test RunMetrics dataclass."""
        metrics = RunMetrics(
            run_id="test_run_1",
            experiment_name="test_exp",
            model_name="RandomForest",
            parameters={"n_estimators": 100},
            metrics={"accuracy": 0.85},
        )

        assert metrics.run_id == "test_run_1"
        assert metrics.parameters["n_estimators"] == 100
        assert metrics.metrics["accuracy"] == 0.85

        # Test serialization
        data_dict = metrics.to_dict()
        assert "run_id" in data_dict
        assert "start_time" in data_dict

        # Test deserialization
        restored_metrics = RunMetrics.from_dict(data_dict)
        assert restored_metrics.run_id == metrics.run_id

    @pytest.mark.skipif(os.name == 'nt', reason="Database file locking issues on Windows")
    def test_experiment_tracker_basic(self):
        """Test basic ExperimentTracker functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(
                tracking_uri=tmp_dir, experiment_name="test_experiment", auto_log=False
            )

            # Start run
            run_id = tracker.start_run(run_name="test_run")
            assert tracker.current_run is not None
            assert tracker.current_run.run_id == run_id

            # Log parameters and metrics
            tracker.log_param("n_estimators", 100)
            tracker.log_metric("accuracy", 0.85)

            # End run
            tracker.end_run()
            assert tracker.current_run is None

            # Retrieve run
            run = tracker.get_run(run_id)
            assert run is not None
            assert run.parameters["n_estimators"] == 100
            assert run.metrics["accuracy"] == 0.85

    @pytest.mark.skipif(os.name == 'nt', reason="Database file locking issues on Windows")
    def test_experiment_tracker_model_logging(self):
        """Test model logging."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(tracking_uri=tmp_dir)

            run_id = tracker.start_run()
            tracker.log_model(model, "test_model")
            tracker.end_run()

            # Check that model was saved
            run_dir = Path(tmp_dir) / run_id
            model_path = run_dir / "test_model" / "model.pkl"
            assert model_path.exists()

    @pytest.mark.skipif(os.name == 'nt', reason="Database file locking issues on Windows")
    def test_experiment_search_runs(self):
        """Test run searching functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(
                tracking_uri=tmp_dir, experiment_name="search_test"
            )

            # Create multiple runs
            for i in range(3):
                run_id = tracker.start_run(run_name=f"run_{i}")
                tracker.log_metric("accuracy", 0.8 + i * 0.05)
                tracker.end_run()

            # Search runs
            runs = tracker.search_runs(experiment_name="search_test")
            assert len(runs) == 3

            # Get best run
            best_run = tracker.get_best_run("accuracy", experiment_name="search_test")
            assert best_run is not None
            assert best_run.metrics["accuracy"] == 0.9  # Highest accuracy

    @pytest.mark.skipif(os.name == 'nt', reason="Database file locking issues on Windows")
    def test_experiment_compare_runs(self):
        """Test run comparison."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(tracking_uri=tmp_dir)

            run_ids = []
            for i in range(2):
                run_id = tracker.start_run()
                tracker.log_param("model", f"model_{i}")
                tracker.log_metric("accuracy", 0.8 + i * 0.1)
                tracker.end_run()
                run_ids.append(run_id)

            # Compare runs
            comparison_df = tracker.compare_runs(run_ids)
            assert not comparison_df.empty
            assert len(comparison_df) == 2
            assert "param_model" in comparison_df.columns
            assert "metric_accuracy" in comparison_df.columns

    @pytest.mark.skipif(os.name == 'nt', reason="Database file locking issues on Windows")
    def test_model_registry(self):
        """Test ModelRegistry functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = ModelRegistry(registry_path=tmp_dir)

            # Create and train model
            X, y = make_classification(n_samples=50, n_features=5, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Register model
            version = registry.register_model(
                model=model, name="test_model", description="Test model for unit tests"
            )
            assert version == 1

            # List models
            models = registry.list_models()
            assert len(models) == 1
            assert models[0]["name"] == "test_model"

            # Get model versions
            versions = registry.get_model_versions("test_model")
            assert len(versions) == 1
            assert versions[0]["version"] == 1

            # Load model
            loaded_model = registry.get_model("test_model", version=1)
            assert loaded_model is not None

            # Test predictions match
            original_pred = model.predict(X[:5])
            loaded_pred = loaded_model.predict(X[:5])
            np.testing.assert_array_equal(original_pred, loaded_pred)

    @pytest.mark.skipif(os.name == 'nt', reason="Database file locking issues on Windows")
    def test_nested_runs(self):
        """Test nested run functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = ExperimentTracker(tracking_uri=tmp_dir)

            # Start parent run
            parent_run_id = tracker.start_run(run_name="parent_run")
            tracker.log_param("parent_param", "parent_value")

            # Start nested run
            nested_run_id = tracker.start_run(run_name="nested_run", nested=True)
            tracker.log_param("nested_param", "nested_value")

            # End nested run
            tracker.end_run()

            # Should be back to parent run
            assert tracker.current_run.run_id == parent_run_id

            # End parent run
            tracker.end_run()
            assert tracker.current_run is None

            # Verify both runs exist
            parent_run = tracker.get_run(parent_run_id)
            nested_run = tracker.get_run(nested_run_id)

            assert parent_run is not None
            assert nested_run is not None
            assert parent_run.parameters["parent_param"] == "parent_value"
            assert nested_run.parameters["nested_param"] == "nested_value"
