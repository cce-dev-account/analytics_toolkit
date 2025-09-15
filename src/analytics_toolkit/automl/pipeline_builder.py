"""
Automated ML Pipeline Builder with intelligent preprocessing and feature engineering.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Import our feature engineering components
try:
    from ..feature_engineering import (
        BinningTransformer,
        DateTimeFeatures,
        FeatureSelector,
        InteractionDetector,
        LogTransformer,
        OutlierCapTransformer,
        TargetEncoder,
    )
except ImportError:
    # Fallback if feature engineering not available
    warnings.warn(
        "Feature engineering module not available. Some features will be limited."
    )


@dataclass
class PipelineConfig:
    """Configuration for AutoML pipeline building."""

    # Data preprocessing
    handle_missing: bool = True
    missing_strategy: str = "auto"  # 'auto', 'median', 'mode', 'drop'

    # Feature engineering
    feature_selection: bool = True
    feature_selection_k: Union[int, float] = 0.8
    generate_interactions: bool = True
    max_interactions: int = 20

    # Categorical encoding
    categorical_encoding: str = "auto"  # 'auto', 'target', 'onehot', 'ordinal'

    # Numerical preprocessing
    scaling: bool = True
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust'
    handle_outliers: bool = True
    outlier_method: str = "iqr"

    # Feature transformations
    apply_log_transform: bool = True
    create_polynomial_features: bool = False
    polynomial_degree: int = 2

    # Model settings
    task_type: str = "auto"  # 'auto', 'classification', 'regression'

    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Performance thresholds
    min_feature_importance: float = 0.001
    correlation_threshold: float = 0.95


class DataTypeInference:
    """Intelligent data type inference and preprocessing recommendations."""

    def __init__(self):
        self.column_types = {}
        self.recommendations = {}

    def analyze_dataset(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> dict[str, Any]:
        """Analyze dataset and provide preprocessing recommendations."""
        analysis = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "column_types": {},
            "categorical_columns": [],
            "numerical_columns": [],
            "datetime_columns": [],
            "recommendations": {},
        }

        for column in df.columns:
            if column == target_column:
                continue

            col_type = self._infer_column_type(df[column])
            analysis["column_types"][column] = col_type

            if col_type == "categorical":
                analysis["categorical_columns"].append(column)
            elif col_type == "numerical":
                analysis["numerical_columns"].append(column)
            elif col_type == "datetime":
                analysis["datetime_columns"].append(column)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(
            df, analysis, target_column
        )

        return analysis

    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the type of a pandas Series."""
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Try to convert to datetime
        if series.dtype == "object":
            sample_values = series.dropna().head(100)
            try:
                pd.to_datetime(sample_values, infer_datetime_format=True)
                return "datetime"
            except:
                pass

        # Check for numerical
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = len(series.unique()) / len(series.dropna())
            if unique_ratio < 0.05 and len(series.unique()) < 20:
                return "categorical"
            return "numerical"

        # Check for categorical
        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            unique_values = len(series.unique())
            if unique_values < 50 or unique_values / len(series) < 0.5:
                return "categorical"
            return "text"

        return "other"

    def _generate_recommendations(
        self, df: pd.DataFrame, analysis: dict, target_column: Optional[str]
    ) -> dict[str, Any]:
        """Generate preprocessing recommendations based on data analysis."""
        recommendations = {
            "preprocessing_steps": [],
            "feature_engineering": [],
            "warnings": [],
        }

        # Missing value recommendations
        missing_cols = [
            col for col, count in analysis["missing_values"].items() if count > 0
        ]
        if missing_cols:
            recommendations["preprocessing_steps"].append("handle_missing_values")

        # High cardinality categorical warning
        for col in analysis["categorical_columns"]:
            unique_count = df[col].nunique()
            if unique_count > 100:
                recommendations["warnings"].append(
                    f"Column '{col}' has high cardinality ({unique_count} unique values). "
                    "Consider grouping rare categories."
                )

        # Recommend feature engineering
        if len(analysis["numerical_columns"]) >= 2:
            recommendations["feature_engineering"].append("generate_interactions")

        if analysis["datetime_columns"]:
            recommendations["feature_engineering"].append("extract_datetime_features")

        # Skewness detection for numerical columns
        for col in analysis["numerical_columns"]:
            if df[col].skew() > 2:
                recommendations["feature_engineering"].append(
                    f"apply_log_transform_to_{col}"
                )

        return recommendations


class AutoMLPipeline:
    """
    Automated ML Pipeline builder that intelligently preprocesses data and builds ML models.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = True):
        self.config = config or PipelineConfig()
        self.verbose = verbose
        self.pipeline = None
        self.feature_names_ = None
        self.target_encoder_ = None
        self.data_analysis_ = None
        self.performance_metrics_ = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        target_column: Optional[str] = None,
    ) -> "AutoMLPipeline":
        """
        Automatically build and fit ML pipeline.

        Parameters
        ----------
        X : DataFrame or array-like
            Input features
        y : Series or array-like
            Target variable
        target_column : str, optional
            Name of target column (for DataFrame input)
        """
        if self.verbose:
            print("🤖 AutoML Pipeline Builder Starting...")

        # Convert inputs to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Analyze dataset
        analyzer = DataTypeInference()
        self.data_analysis_ = analyzer.analyze_dataset(X, target_column)

        if self.verbose:
            print("📊 Dataset Analysis:")
            print(f"  - Samples: {self.data_analysis_['n_samples']:,}")
            print(f"  - Features: {self.data_analysis_['n_features']}")
            print(f"  - Categorical: {len(self.data_analysis_['categorical_columns'])}")
            print(f"  - Numerical: {len(self.data_analysis_['numerical_columns'])}")
            print(f"  - DateTime: {len(self.data_analysis_['datetime_columns'])}")

        # Determine task type
        y_array = np.array(y)
        if self.config.task_type == "auto":
            unique_targets = len(np.unique(y_array))
            if unique_targets <= 20 and np.all(y_array == y_array.astype(int)):
                self.task_type_ = "classification"
            else:
                self.task_type_ = "regression"
        else:
            self.task_type_ = self.config.task_type

        if self.verbose:
            print(f"🎯 Task Type: {self.task_type_}")

        # Build pipeline
        self.pipeline = self._build_pipeline(X, y)

        # Fit pipeline
        if self.verbose:
            print("🔧 Fitting pipeline...")

        self.pipeline.fit(X, y)

        # Evaluate pipeline
        self._evaluate_pipeline(X, y)

        if self.verbose:
            print("✅ AutoML Pipeline completed successfully!")

        return self

    def _build_pipeline(self, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
        """Build the ML pipeline based on data analysis and config."""
        steps = []

        # Step 1: Handle missing values
        if self.config.handle_missing:
            steps.append(("missing_handler", self._create_missing_handler(X)))

        # Step 2: Feature engineering
        if self.data_analysis_["datetime_columns"]:
            steps.append(("datetime_features", self._create_datetime_transformer()))

        # Step 3: Preprocessing (scaling, encoding)
        steps.append(("preprocessor", self._create_preprocessor(X, y)))

        # Step 4: Feature selection and interaction detection
        if self.config.feature_selection or self.config.generate_interactions:
            steps.append(
                ("feature_engineering", self._create_feature_engineering_step(X, y))
            )

        # Step 5: Final feature selection
        if self.config.feature_selection:
            try:
                steps.append(
                    (
                        "feature_selector",
                        FeatureSelector(
                            methods=["variance", "mutual_info"],
                            mutual_info_k=self.config.feature_selection_k,
                        ),
                    )
                )
            except NameError:
                if self.verbose:
                    print("⚠️ Feature selection not available, skipping...")

        # Step 6: Model
        steps.append(("model", self._create_default_model()))

        return Pipeline(steps)

    def _create_missing_handler(self, X: pd.DataFrame) -> BaseEstimator:
        """Create missing value handler."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        numeric_features = self.data_analysis_["numerical_columns"]
        categorical_features = self.data_analysis_["categorical_columns"]

        transformers = []

        if numeric_features:
            if self.config.missing_strategy == "auto":
                strategy = "median"
            else:
                strategy = self.config.missing_strategy
            transformers.append(
                ("num_imputer", SimpleImputer(strategy=strategy), numeric_features)
            )

        if categorical_features:
            transformers.append(
                (
                    "cat_imputer",
                    SimpleImputer(strategy="most_frequent"),
                    categorical_features,
                )
            )

        if not transformers:
            # If no missing values, return passthrough
            from sklearn.preprocessing import FunctionTransformer

            return FunctionTransformer()

        return ColumnTransformer(transformers, remainder="passthrough")

    def _create_datetime_transformer(self) -> BaseEstimator:
        """Create datetime feature extractor."""
        try:
            return DateTimeFeatures(
                features=["year", "month", "day", "dayofweek", "is_weekend"],
                cyclical_encoding=True,
            )
        except NameError:
            if self.verbose:
                print("⚠️ DateTime features not available, using basic transformer...")
            return "passthrough"

    def _create_preprocessor(self, X: pd.DataFrame, y: np.ndarray) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric_features = self.data_analysis_["numerical_columns"]
        categorical_features = self.data_analysis_["categorical_columns"]

        transformers = []

        # Numerical preprocessing
        if numeric_features:
            numeric_pipeline = []

            # Outlier handling
            if self.config.handle_outliers:
                try:
                    numeric_pipeline.append(
                        (
                            "outlier_cap",
                            OutlierCapTransformer(method=self.config.outlier_method),
                        )
                    )
                except NameError:
                    pass

            # Log transformation
            if self.config.apply_log_transform:
                try:
                    numeric_pipeline.append(("log_transform", LogTransformer()))
                except NameError:
                    pass

            # Scaling
            if self.config.scaling:
                if self.config.scaling_method == "standard":
                    numeric_pipeline.append(("scaler", StandardScaler()))
                # Add other scaling methods as needed

            if numeric_pipeline:
                from sklearn.pipeline import Pipeline as SklearnPipeline

                numeric_transformer = SklearnPipeline(numeric_pipeline)
            else:
                numeric_transformer = "passthrough"

            transformers.append(("num", numeric_transformer, numeric_features))

        # Categorical preprocessing
        if categorical_features:
            if self.config.categorical_encoding == "auto":
                # Use target encoding for high cardinality, onehot for low
                high_cardinality_cols = []
                low_cardinality_cols = []

                for col in categorical_features:
                    if X[col].nunique() > 10:
                        high_cardinality_cols.append(col)
                    else:
                        low_cardinality_cols.append(col)

                if high_cardinality_cols:
                    try:
                        transformers.append(
                            ("cat_target", TargetEncoder(), high_cardinality_cols)
                        )
                    except NameError:
                        transformers.append(
                            (
                                "cat_high",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                                high_cardinality_cols,
                            )
                        )

                if low_cardinality_cols:
                    transformers.append(
                        (
                            "cat_onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            low_cardinality_cols,
                        )
                    )

            elif self.config.categorical_encoding == "onehot":
                transformers.append(
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_features,
                    )
                )

            elif self.config.categorical_encoding == "target":
                try:
                    transformers.append(("cat", TargetEncoder(), categorical_features))
                except NameError:
                    transformers.append(
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            categorical_features,
                        )
                    )

        return ColumnTransformer(transformers, remainder="passthrough")

    def _create_feature_engineering_step(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> BaseEstimator:
        """Create feature engineering step."""
        if self.config.generate_interactions:
            try:
                return InteractionDetector(
                    method="tree_based", max_interactions=self.config.max_interactions
                )
            except NameError:
                if self.verbose:
                    print("⚠️ Interaction detection not available, skipping...")
                return "passthrough"
        return "passthrough"

    def _create_default_model(self) -> BaseEstimator:
        """Create default model based on task type."""
        if self.task_type_ == "classification":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state, n_jobs=-1
            )
        else:
            return RandomForestRegressor(
                n_estimators=100, random_state=self.config.random_state, n_jobs=-1
            )

    def _evaluate_pipeline(self, X: pd.DataFrame, y: np.ndarray):
        """Evaluate the pipeline performance."""
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.task_type_ == "classification" else None,
        )

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.pipeline,
            X_train,
            y_train,
            cv=self.config.cv_folds,
            scoring="accuracy" if self.task_type_ == "classification" else "r2",
            n_jobs=-1,
        )

        # Test set evaluation
        y_pred = self.pipeline.predict(X_test)

        if self.task_type_ == "classification":
            test_score = accuracy_score(y_test, y_pred)
            self.performance_metrics_ = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": test_score,
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }
        else:
            test_score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            self.performance_metrics_ = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_r2": test_score,
                "test_mse": mse,
                "test_rmse": np.sqrt(mse),
            }

        if self.verbose:
            print("📈 Performance Metrics:")
            print(f"  - CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            if self.task_type_ == "classification":
                print(f"  - Test Accuracy: {test_score:.4f}")
            else:
                print(f"  - Test R²: {test_score:.4f}")
                print(f"  - Test RMSE: {np.sqrt(mse):.4f}")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        return self.pipeline.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task_type_ != "classification":
            raise ValueError("predict_proba only available for classification tasks")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        return self.pipeline.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the fitted model."""
        if self.pipeline is None:
            return None

        model = self.pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            # Get feature names after all transformations
            feature_names = self._get_feature_names()
            return pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

        return None

    def _get_feature_names(self) -> list[str]:
        """Extract feature names after all transformations."""
        try:
            # This is complex with ColumnTransformer - simplified version
            if hasattr(self.pipeline, "feature_names_in_"):
                return list(self.pipeline.feature_names_in_)
            else:
                # Fallback to generic names
                model = self.pipeline.named_steps["model"]
                n_features = model.n_features_in_
                return [f"feature_{i}" for i in range(n_features)]
        except:
            return ["unknown_features"]

    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline to disk."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        joblib.dump(
            {
                "pipeline": self.pipeline,
                "config": self.config,
                "task_type": self.task_type_,
                "data_analysis": self.data_analysis_,
                "performance_metrics": self.performance_metrics_,
            },
            filepath,
        )

        if self.verbose:
            print(f"💾 Pipeline saved to {filepath}")

    @classmethod
    def load_pipeline(cls, filepath: str) -> "AutoMLPipeline":
        """Load a saved pipeline from disk."""
        data = joblib.load(filepath)

        instance = cls(config=data["config"], verbose=False)
        instance.pipeline = data["pipeline"]
        instance.task_type_ = data["task_type"]
        instance.data_analysis_ = data["data_analysis"]
        instance.performance_metrics_ = data["performance_metrics"]

        return instance

    def summary(self) -> dict[str, Any]:
        """Get a summary of the pipeline and its performance."""
        return {
            "task_type": self.task_type_,
            "data_analysis": self.data_analysis_,
            "performance_metrics": self.performance_metrics_,
            "pipeline_steps": list(self.pipeline.named_steps.keys())
            if self.pipeline
            else None,
            "config": self.config,
        }
