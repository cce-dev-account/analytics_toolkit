"""
Automated model selection, comparison, and ensemble building.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Try to import additional models
try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@dataclass
class ModelComparisonResult:
    """Results from model comparison."""

    model_name: str
    estimator: BaseEstimator
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    fit_time: float
    score_time: float
    best_params: Optional[dict[str, Any]] = None


class AutoModelSelector:
    """
    Automated model selection with intelligent algorithm recommendation.
    """

    def __init__(
        self,
        task_type: str = "auto",
        include_models: Optional[list[str]] = None,
        exclude_models: Optional[list[str]] = None,
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize AutoModelSelector.

        Parameters
        ----------
        task_type : str, default='auto'
            'classification', 'regression', or 'auto' for automatic detection
        include_models : list, optional
            Specific models to include
        exclude_models : list, optional
            Models to exclude from selection
        cv_folds : int, default=5
            Number of cross-validation folds
        scoring : str, optional
            Scoring metric (auto-detected if None)
        random_state : int, default=42
            Random state for reproducibility
        verbose : bool, default=True
            Enable verbose output
        """
        self.task_type = task_type
        self.include_models = include_models
        self.exclude_models = exclude_models or []
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

        self.results_ = []
        self.best_model_ = None
        self.best_score_ = None
        self.task_type_ = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "AutoModelSelector":
        """
        Fit and compare multiple models.

        Parameters
        ----------
        X : DataFrame or array-like
            Training features
        y : Series or array-like
            Training target

        Returns
        -------
        self
        """
        if self.verbose:
            print("🤖 Starting automated model selection...")

        # Prepare data
        X = self._prepare_features(X)
        y = self._prepare_target(y)

        # Determine task type
        if self.task_type == "auto":
            self.task_type_ = self._determine_task_type(y)
        else:
            self.task_type_ = self.task_type

        # Get scoring metric
        scoring = self.scoring or self._get_default_scoring(self.task_type_)

        if self.verbose:
            print(f"📊 Task type: {self.task_type_}")
            print(f"🎯 Scoring metric: {scoring}")
            print(f"🔢 Data shape: {X.shape}")

        # Get models to evaluate
        models = self._get_models()

        if self.verbose:
            print(f"🧪 Evaluating {len(models)} models...")

        # Evaluate models
        self.results_ = []
        for i, (name, model) in enumerate(models.items()):
            if self.verbose:
                print(f"[{i+1}/{len(models)}] Evaluating {name}...")

            try:
                result = self._evaluate_model(model, X, y, scoring, name)
                self.results_.append(result)

                if self.verbose:
                    print(f"  Score: {result.mean_score:.4f} ± {result.std_score:.4f}")

            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Failed: {str(e)}")
                continue

        # Sort results by mean score
        self.results_.sort(key=lambda x: x.mean_score, reverse=True)

        # Select best model
        if self.results_:
            self.best_model_ = self.results_[0].estimator
            self.best_score_ = self.results_[0].mean_score

            if self.verbose:
                print(f"\n🏆 Best model: {self.results_[0].model_name}")
                print(f"📈 Best score: {self.best_score_:.4f}")

        return self

    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prepare feature matrix."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)

    def _prepare_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Prepare target vector."""
        if isinstance(y, pd.Series):
            return y.values
        return np.array(y)

    def _determine_task_type(self, y: np.ndarray) -> str:
        """Determine task type from target variable."""
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            return "classification"
        return "regression"

    def _get_default_scoring(self, task_type: str) -> str:
        """Get default scoring metric for task type."""
        if task_type == "classification":
            return "accuracy"
        else:
            return "r2"

    def _get_models(self) -> dict[str, BaseEstimator]:
        """Get models to evaluate."""
        if self.task_type_ == "classification":
            models = self._get_classification_models()
        else:
            models = self._get_regression_models()

        # Filter models based on include/exclude lists
        if self.include_models:
            models = {k: v for k, v in models.items() if k in self.include_models}

        if self.exclude_models:
            models = {k: v for k, v in models.items() if k not in self.exclude_models}

        return models

    def _get_classification_models(self) -> dict[str, BaseEstimator]:
        """Get classification models."""
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            "SVM": SVC(random_state=self.random_state, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=self.random_state),
            "Neural Network": MLPClassifier(
                random_state=self.random_state, max_iter=500
            ),
        }

        # Add gradient boosting models if available
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                random_state=self.random_state, eval_metric="logloss", n_jobs=-1
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(
                random_state=self.random_state, verbose=-1, n_jobs=-1
            )

        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(
                random_state=self.random_state, verbose=False
            )

        return models

    def _get_regression_models(self) -> dict[str, BaseEstimator]:
        """Get regression models."""
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            ),
            "Ridge": Ridge(random_state=self.random_state),
            "Lasso": Lasso(random_state=self.random_state, max_iter=1000),
            "ElasticNet": ElasticNet(random_state=self.random_state, max_iter=1000),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
            "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
            "AdaBoost": AdaBoostRegressor(random_state=self.random_state),
            "Neural Network": MLPRegressor(
                random_state=self.random_state, max_iter=500
            ),
        }

        # Add gradient boosting models if available
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(random_state=self.random_state, n_jobs=-1)

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(
                random_state=self.random_state, verbose=-1, n_jobs=-1
            )

        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostRegressor(
                random_state=self.random_state, verbose=False
            )

        return models

    def _evaluate_model(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str,
        name: str,
    ) -> ModelComparisonResult:
        """Evaluate a single model using cross-validation."""

        # Set up cross-validation
        if self.task_type_ == "classification":
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        else:
            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

        # Perform cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # Avoid nested parallelism
            return_train_score=False,
        )

        return ModelComparisonResult(
            model_name=name,
            estimator=clone(model),
            cv_scores=cv_results["test_score"],
            mean_score=cv_results["test_score"].mean(),
            std_score=cv_results["test_score"].std(),
            fit_time=cv_results["fit_time"].mean(),
            score_time=cv_results["score_time"].mean(),
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a DataFrame."""
        if not self.results_:
            return pd.DataFrame()

        data = []
        for result in self.results_:
            data.append(
                {
                    "model": result.model_name,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "fit_time": result.fit_time,
                    "score_time": result.score_time,
                }
            )

        return pd.DataFrame(data)

    def plot_results(self):
        """Plot model comparison results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not self.results_:
                print("No results to plot. Run fit() first.")
                return

            # Prepare data
            df = self.get_results_dataframe()

            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Score comparison
            sns.barplot(data=df, x="mean_score", y="model", ax=ax1)
            ax1.set_title("Model Performance Comparison")
            ax1.set_xlabel("Mean CV Score")

            # Score with error bars
            ax2.barh(df["model"], df["mean_score"], xerr=df["std_score"])
            ax2.set_xlabel("CV Score ± Std")
            ax2.set_title("Model Performance with Standard Deviation")

            # Fit time comparison
            sns.barplot(data=df, x="fit_time", y="model", ax=ax3)
            ax3.set_title("Model Training Time")
            ax3.set_xlabel("Fit Time (seconds)")

            # Score vs Time scatter
            ax4.scatter(df["fit_time"], df["mean_score"], s=100)
            for i, model in enumerate(df["model"]):
                ax4.annotate(
                    model,
                    (df.iloc[i]["fit_time"], df.iloc[i]["mean_score"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
            ax4.set_xlabel("Fit Time (seconds)")
            ax4.set_ylabel("Mean CV Score")
            ax4.set_title("Performance vs Training Time")

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")


class ModelComparison:
    """
    Comprehensive model comparison with statistical significance testing.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def compare_models(
        self,
        models: dict[str, BaseEstimator],
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        scoring: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Compare models with statistical significance testing.

        Parameters
        ----------
        models : dict
            Dictionary of {name: model} pairs
        X, y : array-like
            Training data
        scoring : str
            Scoring metric

        Returns
        -------
        DataFrame
            Comparison results with statistical tests
        """
        from scipy import stats

        # Prepare data
        X = np.array(X)
        y = np.array(y)

        # Determine task type
        task_type = self._determine_task_type(y)

        # Set up cross-validation
        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        else:
            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

        # Collect results
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            results[name] = scores

        # Create comparison DataFrame
        comparison_data = []
        model_names = list(results.keys())

        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names[i + 1 :], i + 1):
                scores1 = results[name1]
                scores2 = results[name2]

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(scores1, scores2)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std

                comparison_data.append(
                    {
                        "model_1": name1,
                        "model_2": name2,
                        "mean_diff": np.mean(scores1) - np.mean(scores2),
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "cohens_d": cohens_d,
                        "effect_size": self._interpret_effect_size(abs(cohens_d)),
                    }
                )

        return pd.DataFrame(comparison_data)

    def _determine_task_type(self, y: np.ndarray) -> str:
        """Determine task type."""
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            return "classification"
        return "regression"

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"


class EnsembleBuilder:
    """
    Automated ensemble model builder using voting, stacking, and blending.
    """

    def __init__(
        self,
        ensemble_methods: list[str] = None,
        n_base_models: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize EnsembleBuilder.

        Parameters
        ----------
        ensemble_methods : list, optional
            Ensemble methods to try ['voting', 'stacking', 'blending']
        n_base_models : int, default=5
            Number of base models to use in ensemble
        random_state : int, default=42
            Random state for reproducibility
        verbose : bool, default=True
            Enable verbose output
        """
        self.ensemble_methods = ensemble_methods or ["voting"]
        self.n_base_models = n_base_models
        self.random_state = random_state
        self.verbose = verbose

        self.base_models_ = []
        self.ensemble_models_ = {}
        self.best_ensemble_ = None

    def build_ensemble(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        base_models: Optional[list[BaseEstimator]] = None,
    ) -> "EnsembleBuilder":
        """
        Build ensemble models from base models.

        Parameters
        ----------
        X, y : array-like
            Training data
        base_models : list, optional
            Base models to use (auto-selected if None)

        Returns
        -------
        self
        """
        if self.verbose:
            print("🎭 Building ensemble models...")

        # Prepare data
        X = np.array(X)
        y = np.array(y)

        # Determine task type
        task_type = self._determine_task_type(y)

        # Get base models
        if base_models is None:
            selector = AutoModelSelector(
                task_type=task_type, verbose=False, random_state=self.random_state
            )
            selector.fit(X, y)

            # Select top N models
            self.base_models_ = [
                (result.model_name, result.estimator)
                for result in selector.results_[: self.n_base_models]
            ]
        else:
            self.base_models_ = [
                (f"model_{i}", model) for i, model in enumerate(base_models)
            ]

        if self.verbose:
            print(f"🔧 Using {len(self.base_models_)} base models")

        # Build ensemble models
        self.ensemble_models_ = {}

        for method in self.ensemble_methods:
            if self.verbose:
                print(f"📊 Building {method} ensemble...")

            ensemble = self._build_ensemble_method(method, task_type)
            if ensemble is not None:
                self.ensemble_models_[method] = ensemble

        # Evaluate ensemble models
        best_score = -np.inf
        scoring = "accuracy" if task_type == "classification" else "r2"

        for method, ensemble in self.ensemble_models_.items():
            scores = cross_val_score(ensemble, X, y, cv=5, scoring=scoring, n_jobs=-1)
            mean_score = scores.mean()

            if self.verbose:
                print(f"  {method}: {mean_score:.4f} ± {scores.std():.4f}")

            if mean_score > best_score:
                best_score = mean_score
                self.best_ensemble_ = ensemble

        if self.verbose and self.best_ensemble_:
            print(f"🏆 Best ensemble score: {best_score:.4f}")

        # Fit the best ensemble on the full training data
        if self.best_ensemble_ is not None:
            self.best_ensemble_.fit(X, y)
            if self.verbose:
                print("✅ Best ensemble fitted on full training data")

        return self

    def _determine_task_type(self, y: np.ndarray) -> str:
        """Determine task type."""
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.all(y == y.astype(int)):
            return "classification"
        return "regression"

    def _build_ensemble_method(
        self, method: str, task_type: str
    ) -> Optional[BaseEstimator]:
        """Build ensemble using specified method."""
        try:
            if method == "voting":
                return self._build_voting_ensemble(task_type)
            elif method == "stacking":
                return self._build_stacking_ensemble(task_type)
            elif method == "blending":
                return self._build_blending_ensemble(task_type)
            else:
                warnings.warn(f"Unknown ensemble method: {method}")
                return None

        except Exception as e:
            if self.verbose:
                print(f"  ❌ Failed to build {method} ensemble: {e}")
            return None

    def _build_voting_ensemble(self, task_type: str) -> BaseEstimator:
        """Build voting ensemble."""
        estimators = [(name, clone(model)) for name, model in self.base_models_]

        if task_type == "classification":
            return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        else:
            return VotingRegressor(estimators=estimators, n_jobs=-1)

    def _build_stacking_ensemble(self, task_type: str) -> BaseEstimator:
        """Build stacking ensemble."""
        try:
            from sklearn.ensemble import StackingClassifier, StackingRegressor
        except ImportError:
            warnings.warn("Stacking requires scikit-learn >= 0.22")
            return None

        estimators = [(name, clone(model)) for name, model in self.base_models_]

        if task_type == "classification":
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=self.random_state),
                cv=5,
                n_jobs=-1,
            )
        else:
            return StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(random_state=self.random_state),
                cv=5,
                n_jobs=-1,
            )

    def _build_blending_ensemble(self, task_type: str) -> BaseEstimator:
        """Build blending ensemble (simplified stacking without cross-validation)."""
        # This would require a more complex implementation
        # For now, return None and log warning
        warnings.warn("Blending ensemble not yet implemented")
        return None

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using best ensemble."""
        if self.best_ensemble_ is None:
            raise ValueError("No ensemble built. Run build_ensemble() first.")

        X = np.array(X)
        return self.best_ensemble_.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities using best ensemble (classification only)."""
        if self.best_ensemble_ is None:
            raise ValueError("No ensemble built. Run build_ensemble() first.")

        if not hasattr(self.best_ensemble_, "predict_proba"):
            raise ValueError("Best ensemble doesn't support probability prediction")

        X = np.array(X)
        return self.best_ensemble_.predict_proba(X)
