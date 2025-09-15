"""
Intelligent hyperparameter optimization using Optuna and other optimization algorithms.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Try to import Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "Optuna not available. Hyperparameter optimization will use grid search fallback."
    )


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # Optimization settings
    n_trials: int = 100
    timeout: Optional[int] = 300  # 5 minutes
    direction: str = "maximize"  # 'maximize' or 'minimize'

    # Cross-validation settings
    cv_folds: int = 5
    scoring: Optional[str] = None  # Auto-detect based on task

    # Pruning settings
    enable_pruning: bool = True
    pruner_patience: int = 3

    # Parallel settings
    n_jobs: int = -1

    # Random state
    random_state: int = 42

    # Study settings
    study_name: Optional[str] = None
    storage: Optional[str] = None  # For distributed optimization

    # Search space customization
    custom_search_space: Optional[dict[str, Any]] = None


class HyperparameterOptimizer:
    """
    Base class for hyperparameter optimization with multiple backend support.
    """

    def __init__(
        self, config: Optional[OptimizationConfig] = None, verbose: bool = True
    ):
        self.config = config or OptimizationConfig()
        self.verbose = verbose
        self.study = None
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []

    def optimize(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        search_space: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters for the given estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Sklearn estimator to optimize
        X : DataFrame or array-like
            Training features
        y : Series or array-like
            Training target
        search_space : dict, optional
            Custom search space definition

        Returns
        -------
        dict
            Best parameters and optimization results
        """
        if self.verbose:
            print("🔍 Starting hyperparameter optimization...")

        # Prepare data
        X = self._prepare_features(X)
        y = self._prepare_target(y)

        # Determine task type and scoring
        task_type = self._determine_task_type(y)
        scoring = self._get_scoring_metric(task_type)

        # Get search space
        if search_space is None:
            search_space = self._get_default_search_space(estimator, task_type)
        elif self.config.custom_search_space:
            search_space.update(self.config.custom_search_space)

        if self.verbose:
            print(f"📊 Task type: {task_type}")
            print(f"🎯 Scoring metric: {scoring}")
            print(f"🔧 Search space: {list(search_space.keys())}")

        # Run optimization
        if OPTUNA_AVAILABLE:
            results = self._optimize_with_optuna(
                estimator, X, y, search_space, scoring, task_type
            )
        else:
            results = self._optimize_with_grid_search(
                estimator, X, y, search_space, scoring
            )

        if self.verbose:
            print("✅ Optimization completed!")
            print(f"📈 Best score: {results['best_score']:.4f}")
            print(f"⚙️ Best parameters: {results['best_params']}")

        return results

    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prepare feature matrix."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X

    def _prepare_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Prepare target vector."""
        if isinstance(y, pd.Series):
            return y.values
        return y

    def _determine_task_type(self, y: np.ndarray) -> str:
        """Determine if task is classification or regression."""
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            return "classification"
        return "regression"

    def _get_scoring_metric(self, task_type: str) -> str:
        """Get appropriate scoring metric."""
        if self.config.scoring:
            return self.config.scoring

        if task_type == "classification":
            return "accuracy"
        else:
            return "r2"

    def _get_default_search_space(
        self, estimator: BaseEstimator, task_type: str
    ) -> dict[str, Any]:
        """Get default search space for common estimators."""
        estimator_name = estimator.__class__.__name__

        # Random Forest
        if "RandomForest" in estimator_name:
            return {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }

        # Logistic Regression
        elif estimator_name == "LogisticRegression":
            return {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000, 2000, 5000],
            }

        # Ridge Regression
        elif estimator_name == "Ridge":
            return {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
            }

        # SVM
        elif estimator_name in ["SVC", "SVR"]:
            return {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "kernel": ["rbf", "linear", "poly"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            }

        # KNN
        elif "KNeighbors" in estimator_name:
            return {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }

        # Decision Tree
        elif "DecisionTree" in estimator_name:
            return {
                "max_depth": [None, 5, 10, 15, 20, 25],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"]
                if task_type == "classification"
                else ["squared_error", "absolute_error"],
            }

        # Default fallback
        return {}

    def _optimize_with_optuna(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        search_space: dict[str, Any],
        scoring: str,
        task_type: str,
    ) -> dict[str, Any]:
        """Optimize using Optuna."""

        # Create study
        direction = (
            "maximize" if scoring in ["accuracy", "f1", "r2", "roc_auc"] else "minimize"
        )

        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner()
            if self.config.enable_pruning
            else optuna.pruners.NopPruner(),
            study_name=self.config.study_name,
            storage=self.config.storage,
        )

        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in search_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numerical parameter
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(
                                param_name, min(param_values), max(param_values)
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name, min(param_values), max(param_values)
                            )
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_values
                        )

            # Create estimator with suggested parameters
            estimator_with_params = estimator.set_params(**params)

            # Cross-validation
            if task_type == "classification":
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )
            else:
                cv = KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )

            scores = cross_val_score(
                estimator_with_params, X, y, cv=cv, scoring=scoring, n_jobs=1
            )

            # Report intermediate values for pruning
            if self.config.enable_pruning:
                for step, score in enumerate(scores):
                    trial.report(score, step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            return scores.mean()

        # Optimize
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=1,
        )  # Set to 1 to avoid nested parallelism

        # Extract results
        self.study = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        # Create optimization history
        trials_df = study.trials_dataframe()
        if not trials_df.empty:
            self.optimization_history_ = trials_df.to_dict("records")

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "n_trials": len(study.trials),
            "optimization_time": time.time() - start_time,
            "study": study,
        }

    def _optimize_with_grid_search(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        search_space: dict[str, Any],
        scoring: str,
    ) -> dict[str, Any]:
        """Fallback optimization using GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        if self.verbose:
            print(
                "⚠️ Using GridSearchCV fallback (install optuna for better optimization)"
            )

        # Limit search space for grid search
        limited_space = {}
        for param, values in search_space.items():
            if isinstance(values, list) and len(values) > 5:
                # Limit to 5 values max for grid search
                limited_space[param] = values[:: len(values) // 4][:5]
            else:
                limited_space[param] = values

        grid_search = GridSearchCV(
            estimator,
            limited_space,
            cv=self.config.cv_folds,
            scoring=scoring,
            n_jobs=self.config.n_jobs,
            verbose=1 if self.verbose else 0,
        )

        start_time = time.time()
        grid_search.fit(X, y)

        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "n_trials": len(grid_search.cv_results_["params"]),
            "optimization_time": time.time() - start_time,
            "grid_search": grid_search,
        }

    def plot_optimization_history(self):
        """Plot optimization history (requires Optuna)."""
        if not OPTUNA_AVAILABLE or self.study is None:
            print(
                "⚠️ Optimization history plotting requires Optuna and a completed study"
            )
            return

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(
                self.study, ax=ax1
            )
            ax1.set_title("Optimization History")

            # Plot parameter importances
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title("Parameter Importances")

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("⚠️ Matplotlib not available for plotting")


class OptunaOptimizer(HyperparameterOptimizer):
    """
    Specialized Optuna optimizer with advanced features.
    """

    def __init__(
        self, config: Optional[OptimizationConfig] = None, verbose: bool = True
    ):
        super().__init__(config, verbose)

        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for OptunaOptimizer. Install with: pip install optuna"
            )

    def multi_objective_optimize(
        self,
        estimators: list[BaseEstimator],
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        objectives: list[str] = None,
    ) -> dict[str, Any]:
        """
        Multi-objective optimization across multiple models and objectives.

        Parameters
        ----------
        estimators : list of BaseEstimator
            List of estimators to optimize
        X, y : array-like
            Training data
        objectives : list of str
            List of objectives to optimize (e.g., ['accuracy', 'f1'])
        """
        if objectives is None:
            objectives = ["accuracy", "precision"]

        if self.verbose:
            print(
                f"🎯 Multi-objective optimization with {len(estimators)} models and {len(objectives)} objectives"
            )

        # Create multi-objective study
        study = optuna.create_study(
            directions=["maximize"] * len(objectives),
            sampler=TPESampler(seed=self.config.random_state),
        )

        def objective(trial):
            # Select model
            model_idx = trial.suggest_int("model_idx", 0, len(estimators) - 1)
            estimator = estimators[model_idx]

            # Get search space for selected model
            task_type = self._determine_task_type(self._prepare_target(y))
            search_space = self._get_default_search_space(estimator, task_type)

            # Sample parameters
            params = {}
            for param_name, param_values in search_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(
                                f"{param_name}", min(param_values), max(param_values)
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                f"{param_name}", min(param_values), max(param_values)
                            )
                    else:
                        params[param_name] = trial.suggest_categorical(
                            f"{param_name}", param_values
                        )

            # Fit model and evaluate
            estimator_with_params = estimator.set_params(**params)

            # Calculate multiple objectives
            results = []
            for objective_name in objectives:
                scores = cross_val_score(
                    estimator_with_params,
                    self._prepare_features(X),
                    self._prepare_target(y),
                    cv=self.config.cv_folds,
                    scoring=objective_name,
                    n_jobs=1,
                )
                results.append(scores.mean())

            return results

        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials)

        # Get Pareto front
        pareto_front = study.best_trials

        return {
            "pareto_front": pareto_front,
            "n_trials": len(study.trials),
            "study": study,
        }

    def automated_feature_selection_optimization(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> dict[str, Any]:
        """
        Optimize both hyperparameters and feature selection simultaneously.
        """
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        from sklearn.pipeline import Pipeline

        X_prepared = self._prepare_features(X)
        y_prepared = self._prepare_target(y)
        task_type = self._determine_task_type(y_prepared)

        def objective(trial):
            # Feature selection parameters
            k_features = trial.suggest_int(
                "k_features", 1, min(50, X_prepared.shape[1])
            )

            # Model hyperparameters
            model_params = {}
            search_space = self._get_default_search_space(estimator, task_type)

            for param_name, param_values in search_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            model_params[f"model__{param_name}"] = trial.suggest_int(
                                f"model__{param_name}",
                                min(param_values),
                                max(param_values),
                            )
                        else:
                            model_params[f"model__{param_name}"] = trial.suggest_float(
                                f"model__{param_name}",
                                min(param_values),
                                max(param_values),
                            )
                    else:
                        model_params[
                            f"model__{param_name}"
                        ] = trial.suggest_categorical(
                            f"model__{param_name}", param_values
                        )

            # Create pipeline
            score_func = f_classif if task_type == "classification" else f_regression
            pipeline = Pipeline(
                [
                    (
                        "feature_selection",
                        SelectKBest(score_func=score_func, k=k_features),
                    ),
                    ("model", estimator),
                ]
            )

            pipeline.set_params(**model_params)

            # Cross-validation
            scoring = self._get_scoring_metric(task_type)
            scores = cross_val_score(
                pipeline,
                X_prepared,
                y_prepared,
                cv=self.config.cv_folds,
                scoring=scoring,
                n_jobs=1,
            )

            return scores.mean()

        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }
