"""
Model evaluation and performance visualization tools.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve, validation_curve

from .themes import PlotTheme, apply_theme


class ModelEvaluationPlots:
    """Main class for model evaluation visualizations."""

    def __init__(self, theme: PlotTheme | str = "default"):
        """Initialize with theme."""
        self.theme = apply_theme(theme)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
        normalize: bool = False,
        figsize: tuple[int, int] = (8, 6),
        title: str = "Confusion Matrix",
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Class names for labeling
        normalize : bool
            Whether to normalize the confusion matrix
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            xticklabels=class_names or True,
            yticklabels=class_names or True,
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        plt.tight_layout()
        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        class_names: list[str] | None = None,
        figsize: tuple[int, int] = (10, 8),
        title: str = "ROC Curves",
    ) -> plt.Figure:
        """
        Plot ROC curves for binary or multiclass classification.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_scores : array-like
            Predicted probabilities or decision scores
        class_names : list, optional
            Class names
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Handle binary classification
        if y_scores.ndim == 1 or y_scores.shape[1] == 2:
            if y_scores.ndim == 2:
                y_scores = y_scores[:, 1]  # Use positive class probabilities

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                color=self.theme.colors["primary"],
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )

        else:
            # Handle multiclass classification
            from itertools import cycle

            from sklearn.preprocessing import label_binarize

            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)

            colors = cycle(self.theme.get_color_palette(len(classes)))

            for i, (class_idx, color) in enumerate(zip(classes, colors, strict=False)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)

                class_name = class_names[i] if class_names else f"Class {class_idx}"
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    lw=2,
                    label=f"{class_name} (AUC = {roc_auc:.2f})",
                )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.8)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        class_names: list[str] | None = None,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Precision-Recall Curves",
    ) -> plt.Figure:
        """
        Plot precision-recall curves.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_scores : array-like
            Predicted probabilities
        class_names : list, optional
            Class names
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Handle binary classification
        if y_scores.ndim == 1 or y_scores.shape[1] == 2:
            if y_scores.ndim == 2:
                y_scores = y_scores[:, 1]

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)

            ax.plot(
                recall,
                precision,
                color=self.theme.colors["primary"],
                lw=2,
                label=f"PR curve (AP = {avg_precision:.2f})",
            )

            # Add baseline
            baseline = np.sum(y_true) / len(y_true)
            ax.axhline(
                y=baseline,
                color="gray",
                linestyle="--",
                alpha=0.8,
                label=f"Baseline (AP = {baseline:.2f})",
            )

        else:
            # Handle multiclass classification
            from itertools import cycle

            from sklearn.preprocessing import label_binarize

            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)

            colors = cycle(self.theme.get_color_palette(len(classes)))

            for i, (class_idx, color) in enumerate(zip(classes, colors, strict=False)):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_scores[:, i]
                )
                avg_precision = average_precision_score(
                    y_true_bin[:, i], y_scores[:, i]
                )

                class_name = class_names[i] if class_names else f"Class {class_idx}"
                ax.plot(
                    recall,
                    precision,
                    color=color,
                    lw=2,
                    label=f"{class_name} (AP = {avg_precision:.2f})",
                )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_learning_curve(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        train_sizes: np.ndarray | None = None,
        scoring: str = "accuracy",
        figsize: tuple[int, int] = (10, 6),
        title: str = "Learning Curve",
    ) -> plt.Figure:
        """
        Plot learning curve showing training and validation scores.

        Parameters
        ----------
        estimator : sklearn estimator
            Estimator to evaluate
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        cv : int
            Cross-validation folds
        train_sizes : array-like, optional
            Training set sizes to use
        scoring : str
            Scoring metric
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot training scores
        ax.plot(
            train_sizes_abs,
            train_mean,
            "o-",
            color=self.theme.colors["primary"],
            label="Training score",
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color=self.theme.colors["primary"],
        )

        # Plot validation scores
        ax.plot(
            train_sizes_abs,
            val_mean,
            "o-",
            color=self.theme.colors["secondary"],
            label="Validation score",
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color=self.theme.colors["secondary"],
        )

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(f"Score ({scoring})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_validation_curve(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot validation curve for hyperparameter tuning.

        Parameters
        ----------
        estimator : sklearn estimator
            Estimator to evaluate
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        param_name : str
            Parameter name to vary
        param_range : array-like
            Parameter values to test
        cv : int
            Cross-validation folds
        scoring : str
            Scoring metric
        figsize : tuple
            Figure size
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        train_scores, val_scores = validation_curve(
            estimator,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot training scores
        ax.plot(
            param_range,
            train_mean,
            "o-",
            color=self.theme.colors["primary"],
            label="Training score",
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            param_range,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color=self.theme.colors["primary"],
        )

        # Plot validation scores
        ax.plot(
            param_range,
            val_mean,
            "o-",
            color=self.theme.colors["secondary"],
            label="Validation score",
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            param_range,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color=self.theme.colors["secondary"],
        )

        ax.set_xlabel(param_name)
        ax.set_ylabel(f"Score ({scoring})")
        ax.set_title(title or f"Validation Curve ({param_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Handle log scale for certain parameters
        if any(
            keyword in param_name.lower()
            for keyword in ["c", "alpha", "gamma", "learning_rate"]
        ):
            ax.set_xscale("log")

        plt.tight_layout()
        return fig


class ClassificationPlots(ModelEvaluationPlots):
    """Specialized plots for classification tasks."""

    def plot_classification_report_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
        figsize: tuple[int, int] = (10, 6),
        title: str = "Classification Report",
    ) -> plt.Figure:
        """
        Plot classification report as heatmap.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Class names
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get classification report as dict
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        # Convert to DataFrame for heatmap
        df_report = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'support' row

        sns.heatmap(
            df_report.iloc[:, :-1],
            annot=True,
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Score"},
        )

        ax.set_title(title)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Classes")

        plt.tight_layout()
        return fig

    def plot_class_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray | None = None,
        class_names: list[str] | None = None,
        figsize: tuple[int, int] = (12, 5),
        title: str = "Class Distribution",
    ) -> plt.Figure:
        """
        Plot class distribution comparison.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like, optional
            Predicted labels
        class_names : list, optional
            Class names
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        if y_pred is not None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            axes = [ax]

        # True distribution
        true_counts = pd.Series(y_true).value_counts().sort_index()
        if class_names:
            true_counts.index = [class_names[i] for i in true_counts.index]

        axes[0].bar(
            range(len(true_counts)),
            true_counts.values,
            color=self.theme.colors["primary"],
            alpha=0.7,
        )
        axes[0].set_xticks(range(len(true_counts)))
        axes[0].set_xticklabels(true_counts.index, rotation=45)
        axes[0].set_title("True Distribution")
        axes[0].set_ylabel("Count")

        # Predicted distribution
        if y_pred is not None:
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            if class_names:
                pred_counts = pred_counts.reindex(range(len(class_names)), fill_value=0)
                pred_counts.index = [class_names[i] for i in pred_counts.index]

            axes[1].bar(
                range(len(pred_counts)),
                pred_counts.values,
                color=self.theme.colors["secondary"],
                alpha=0.7,
            )
            axes[1].set_xticks(range(len(pred_counts)))
            axes[1].set_xticklabels(pred_counts.index, rotation=45)
            axes[1].set_title("Predicted Distribution")
            axes[1].set_ylabel("Count")

        fig.suptitle(title)
        plt.tight_layout()
        return fig


class RegressionPlots(ModelEvaluationPlots):
    """Specialized plots for regression tasks."""

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: tuple[int, int] = (8, 8),
        title: str = "Predictions vs Actual",
    ) -> plt.Figure:
        """
        Plot predicted vs actual values.

        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, color=self.theme.colors["primary"])

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "--",
            color=self.theme.colors["warning"],
            linewidth=2,
            label="Perfect Prediction",
        )

        # Calculate R²
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true, y_pred)

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{title}\n$R^2$ = {r2:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: tuple[int, int] = (12, 5),
        title: str = "Residual Analysis",
    ) -> plt.Figure:
        """
        Plot residual analysis.

        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title)

        residuals = y_true - y_pred

        # Residuals vs Predicted
        axes[0].scatter(
            y_pred, residuals, alpha=0.6, color=self.theme.colors["primary"]
        )
        axes[0].axhline(
            y=0, color=self.theme.colors["warning"], linestyle="--", linewidth=2
        )
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1].hist(
            residuals, bins=30, alpha=0.7, color=self.theme.colors["secondary"]
        )
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residuals Distribution")
        axes[1].axvline(
            x=0, color=self.theme.colors["warning"], linestyle="--", linewidth=2
        )

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        importance_values: np.ndarray,
        feature_names: list[str],
        top_n: int = 20,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Feature Importance",
    ) -> plt.Figure:
        """
        Plot feature importance.

        Parameters
        ----------
        importance_values : array-like
            Feature importance values
        feature_names : list
            Feature names
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create DataFrame and sort by importance
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance_values})
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )

        # Horizontal bar plot
        ax.barh(
            range(len(importance_df)),
            importance_df["importance"],
            color=self.theme.colors["primary"],
            alpha=0.7,
        )
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df["feature"])
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig


# Convenience functions
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs
) -> plt.Figure:
    """Convenience function for confusion matrix plot."""
    plotter = ClassificationPlots()
    return plotter.plot_confusion_matrix(y_true, y_pred, **kwargs)


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, **kwargs) -> plt.Figure:
    """Convenience function for ROC curve plot."""
    plotter = ClassificationPlots()
    return plotter.plot_roc_curve(y_true, y_scores, **kwargs)


def plot_learning_curve(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs
) -> plt.Figure:
    """Convenience function for learning curve plot."""
    plotter = ModelEvaluationPlots()
    return plotter.plot_learning_curve(estimator, X, y, **kwargs)
