"""
Automated data exploration and profiling visualizations.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .themes import PlotTheme, apply_theme


@dataclass
class ProfileReport:
    """Data structure for profile report results."""

    dataset_info: dict[str, Any]
    column_profiles: dict[str, dict[str, Any]]
    correlations: pd.DataFrame
    missing_data: dict[str, Any]
    duplicates: dict[str, Any]
    figures: dict[str, plt.Figure]


class DataProfiler:
    """Automated data profiling and exploration."""

    def __init__(self, theme: PlotTheme | str = "default"):
        """Initialize with theme."""
        self.theme = apply_theme(theme)

    def generate_profile(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        sample_size: int | None = None,
        include_plots: bool = True,
    ) -> ProfileReport:
        """
        Generate comprehensive data profile.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to profile
        target_column : str, optional
            Target variable for supervised learning context
        sample_size : int, optional
            Sample size for large datasets
        include_plots : bool
            Whether to generate visualizations

        Returns
        -------
        ProfileReport
            Comprehensive profiling results
        """
        # Sample data if too large
        if sample_size and len(data) > sample_size:
            data_sample = data.sample(n=sample_size, random_state=42)
            warnings.warn(
                f"Large dataset detected. Using sample of {sample_size} rows."
            )
        else:
            data_sample = data.copy()

        # Generate profile components
        dataset_info = self._analyze_dataset(data_sample)
        column_profiles = self._profile_columns(data_sample)
        correlations = self._compute_correlations(data_sample)
        missing_data = self._analyze_missing_data(data_sample)
        duplicates = self._analyze_duplicates(data_sample)

        figures = {}
        if include_plots:
            figures = self._generate_visualizations(
                data_sample, target_column, column_profiles, correlations, missing_data
            )

        return ProfileReport(
            dataset_info=dataset_info,
            column_profiles=column_profiles,
            correlations=correlations,
            missing_data=missing_data,
            duplicates=duplicates,
            figures=figures,
        )

    def _analyze_dataset(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze overall dataset characteristics."""
        return {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(
                data.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime_columns": len(data.select_dtypes(include=["datetime64"]).columns),
            "data_types": data.dtypes.to_dict(),
        }

    def _profile_columns(self, data: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Profile individual columns."""
        profiles = {}

        for column in data.columns:
            col_data = data[column]
            profile = {
                "dtype": str(col_data.dtype),
                "non_null_count": col_data.notna().sum(),
                "null_count": col_data.isna().sum(),
                "null_percentage": (col_data.isna().sum() / len(col_data)) * 100,
                "unique_count": col_data.nunique(),
                "unique_percentage": (col_data.nunique() / len(col_data)) * 100,
            }

            if np.issubdtype(col_data.dtype, np.number):
                # Numeric column statistics
                profile.update(
                    {
                        "mean": col_data.mean(),
                        "median": col_data.median(),
                        "std": col_data.std(),
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "q25": col_data.quantile(0.25),
                        "q75": col_data.quantile(0.75),
                        "skewness": stats.skew(col_data.dropna()),
                        "kurtosis": stats.kurtosis(col_data.dropna()),
                        "zeros_count": (col_data == 0).sum(),
                        "negative_count": (col_data < 0).sum(),
                    }
                )

                # Detect potential outliers
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[
                    (col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)
                ]
                profile["outliers_count"] = len(outliers)

            elif col_data.dtype == "object" or pd.api.types.is_categorical_dtype(
                col_data
            ):
                # Categorical column statistics
                value_counts = col_data.value_counts()
                profile.update(
                    {
                        "most_frequent": (
                            value_counts.index[0] if len(value_counts) > 0 else None
                        ),
                        "most_frequent_count": (
                            value_counts.iloc[0] if len(value_counts) > 0 else 0
                        ),
                        "least_frequent": (
                            value_counts.index[-1] if len(value_counts) > 0 else None
                        ),
                        "least_frequent_count": (
                            value_counts.iloc[-1] if len(value_counts) > 0 else 0
                        ),
                        "top_categories": value_counts.head(10).to_dict(),
                    }
                )

                # String length analysis for object columns
                if col_data.dtype == "object":
                    str_lengths = col_data.astype(str).str.len()
                    profile.update(
                        {
                            "avg_string_length": str_lengths.mean(),
                            "min_string_length": str_lengths.min(),
                            "max_string_length": str_lengths.max(),
                        }
                    )

            profiles[column] = profile

        return profiles

    def _compute_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) < 2:
            return pd.DataFrame()

        return numeric_data.corr()

    def _analyze_missing_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100

        # Find patterns of missing data
        missing_matrix = data.isnull()
        missing_patterns = missing_matrix.value_counts()

        return {
            "total_missing": missing_counts.sum(),
            "columns_with_missing": (missing_counts > 0).sum(),
            "missing_by_column": missing_counts.to_dict(),
            "missing_percentages": missing_percentages.to_dict(),
            "missing_patterns": missing_patterns.head(10).to_dict(),
        }

    def _analyze_duplicates(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze duplicate rows."""
        duplicated = data.duplicated()

        return {
            "total_duplicates": duplicated.sum(),
            "duplicate_percentage": (duplicated.sum() / len(data)) * 100,
            "unique_rows": (~duplicated).sum(),
        }

    def _generate_visualizations(
        self,
        data: pd.DataFrame,
        target_column: str | None,
        column_profiles: dict,
        correlations: pd.DataFrame,
        missing_data: dict,
    ) -> dict[str, plt.Figure]:
        """Generate profile visualizations."""
        figures = {}

        # Dataset overview
        figures["overview"] = self._plot_dataset_overview(data, column_profiles)

        # Missing data visualization
        if missing_data["total_missing"] > 0:
            figures["missing_data"] = self._plot_missing_data(data)

        # Correlation matrix
        if not correlations.empty and len(correlations.columns) > 1:
            figures["correlations"] = self._plot_correlation_matrix(correlations)

        # Numeric distributions
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            figures["numeric_distributions"] = self._plot_numeric_distributions(
                data, numeric_cols
            )

        # Categorical distributions
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            figures["categorical_distributions"] = self._plot_categorical_distributions(
                data, categorical_cols
            )

        # Target analysis if provided
        if target_column and target_column in data.columns:
            figures["target_analysis"] = self._plot_target_analysis(data, target_column)

        return figures

    def _plot_dataset_overview(self, data: pd.DataFrame, profiles: dict) -> plt.Figure:
        """Plot dataset overview."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Dataset Overview", fontsize=16)

        # Data types distribution
        dtypes = data.dtypes.value_counts()
        axes[0, 0].pie(
            dtypes.values,
            labels=dtypes.index,
            autopct="%1.1f%%",
            colors=self.theme.get_color_palette(len(dtypes)),
        )
        axes[0, 0].set_title("Data Types Distribution")

        # Missing data by column
        missing_data = data.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0][:10]  # Top 10
        if not missing_data.empty:
            axes[0, 1].bar(
                range(len(missing_data)),
                missing_data.values,
                color=self.theme.colors["warning"],
            )
            axes[0, 1].set_xticks(range(len(missing_data)))
            axes[0, 1].set_xticklabels(missing_data.index, rotation=45, ha="right")
            axes[0, 1].set_title("Missing Values by Column")
            axes[0, 1].set_ylabel("Missing Count")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No Missing Data",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
                fontsize=12,
            )
            axes[0, 1].set_title("Missing Values by Column")

        # Unique values distribution
        unique_counts = [profiles[col].get("unique_count", 0) for col in data.columns]
        axes[1, 0].hist(
            unique_counts, bins=20, alpha=0.7, color=self.theme.colors["primary"]
        )
        axes[1, 0].set_title("Distribution of Unique Values")
        axes[1, 0].set_xlabel("Unique Values Count")
        axes[1, 0].set_ylabel("Frequency")

        # Memory usage by column
        memory_usage = data.memory_usage(deep=True).drop("Index")[:10]  # Top 10
        axes[1, 1].barh(
            range(len(memory_usage)),
            memory_usage.values,
            color=self.theme.colors["secondary"],
        )
        axes[1, 1].set_yticks(range(len(memory_usage)))
        axes[1, 1].set_yticklabels(memory_usage.index)
        axes[1, 1].set_title("Memory Usage by Column (Top 10)")
        axes[1, 1].set_xlabel("Memory Usage (bytes)")

        plt.tight_layout()
        return fig

    def _plot_missing_data(self, data: pd.DataFrame) -> plt.Figure:
        """Plot missing data patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Missing Data Analysis", fontsize=16)

        # Missing data heatmap
        missing_matrix = data.isnull()

        # Sample if too many rows
        if len(missing_matrix) > 1000:
            missing_sample = missing_matrix.sample(n=1000, random_state=42)
        else:
            missing_sample = missing_matrix

        sns.heatmap(
            missing_sample.T,
            cbar=True,
            ax=axes[0],
            cmap="viridis",
            yticklabels=True,
            xticklabels=False,
        )
        axes[0].set_title("Missing Data Pattern")
        axes[0].set_xlabel("Samples")

        # Missing data bar plot
        missing_counts = data.isnull().sum().sort_values(ascending=True)
        missing_counts = missing_counts[missing_counts > 0]

        if not missing_counts.empty:
            axes[1].barh(
                range(len(missing_counts)),
                missing_counts.values,
                color=self.theme.colors["warning"],
            )
            axes[1].set_yticks(range(len(missing_counts)))
            axes[1].set_yticklabels(missing_counts.index)
            axes[1].set_title("Missing Values Count")
            axes[1].set_xlabel("Missing Count")

        plt.tight_layout()
        return fig

    def _plot_correlation_matrix(self, correlations: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix."""
        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(correlations, dtype=bool))
        sns.heatmap(
            correlations,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title("Correlation Matrix", fontsize=14)
        plt.tight_layout()
        return fig

    def _plot_numeric_distributions(
        self, data: pd.DataFrame, numeric_cols: list[str]
    ) -> plt.Figure:
        """Plot distributions for numeric columns."""
        n_cols = min(len(numeric_cols), 6)  # Limit to avoid overcrowding
        cols_to_plot = numeric_cols[:n_cols]

        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        fig.suptitle("Numeric Distributions", fontsize=16)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(cols_to_plot):
            row, col_idx = divmod(i, 3)
            ax = axes[row, col_idx]

            data[col].hist(
                bins=30, alpha=0.7, ax=ax, color=self.theme.colors["primary"]
            )
            ax.set_title(f"{col}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        # Hide unused subplots
        for i in range(n_cols, n_rows * 3):
            row, col_idx = divmod(i, 3)
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        return fig

    def _plot_categorical_distributions(
        self, data: pd.DataFrame, categorical_cols: list[str]
    ) -> plt.Figure:
        """Plot distributions for categorical columns."""
        n_cols = min(len(categorical_cols), 6)
        cols_to_plot = categorical_cols[:n_cols]

        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        fig.suptitle("Categorical Distributions", fontsize=16)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(cols_to_plot):
            row, col_idx = divmod(i, 3)
            ax = axes[row, col_idx]

            value_counts = data[col].value_counts().head(10)
            value_counts.plot(kind="bar", ax=ax, color=self.theme.colors["secondary"])
            ax.set_title(f"{col} (Top 10)")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for i in range(n_cols, n_rows * 3):
            row, col_idx = divmod(i, 3)
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        return fig

    def _plot_target_analysis(
        self, data: pd.DataFrame, target_column: str
    ) -> plt.Figure:
        """Plot target variable analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Target Analysis: {target_column}", fontsize=16)

        target_data = data[target_column].dropna()

        if np.issubdtype(target_data.dtype, np.number):
            # Numeric target
            axes[0, 0].hist(
                target_data, bins=30, alpha=0.7, color=self.theme.colors["primary"]
            )
            axes[0, 0].set_title("Target Distribution")
            axes[0, 0].set_xlabel(target_column)
            axes[0, 0].set_ylabel("Frequency")

            # Box plot
            axes[0, 1].boxplot(
                target_data,
                patch_artist=True,
                boxprops=dict(facecolor=self.theme.colors["secondary"]),
            )
            axes[0, 1].set_title("Target Box Plot")
            axes[0, 1].set_ylabel(target_column)

            # Q-Q plot
            stats.probplot(target_data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Q-Q Plot")

            # Summary statistics
            axes[1, 1].axis("off")
            stats_text = f"""Target Statistics:
Count: {len(target_data):,}
Mean: {target_data.mean():.3f}
Median: {target_data.median():.3f}
Std: {target_data.std():.3f}
Min: {target_data.min():.3f}
Max: {target_data.max():.3f}
Skewness: {stats.skew(target_data):.3f}
Kurtosis: {stats.kurtosis(target_data):.3f}"""
            axes[1, 1].text(
                0.1,
                0.9,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
            )

        else:
            # Categorical target
            value_counts = target_data.value_counts()

            # Bar plot
            value_counts.plot(
                kind="bar", ax=axes[0, 0], color=self.theme.colors["primary"]
            )
            axes[0, 0].set_title("Target Value Counts")
            axes[0, 0].set_xlabel(target_column)
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # Pie chart
            axes[0, 1].pie(
                value_counts.values,
                labels=value_counts.index,
                autopct="%1.1f%%",
                colors=self.theme.get_color_palette(len(value_counts)),
            )
            axes[0, 1].set_title("Target Distribution")

            # Hide unused subplots for categorical
            axes[1, 0].set_visible(False)
            axes[1, 1].axis("off")

            stats_text = f"""Target Statistics:
Total Count: {len(target_data):,}
Unique Values: {target_data.nunique()}
Most Frequent: {value_counts.index[0]}
Most Frequent Count: {value_counts.iloc[0]:,}
Class Imbalance Ratio: {value_counts.max() / value_counts.min():.2f}"""
            axes[1, 1].text(
                0.1,
                0.9,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
            )

        plt.tight_layout()
        return fig


def generate_profile_report(
    data: pd.DataFrame,
    target_column: str | None = None,
    theme: PlotTheme | str = "default",
    **kwargs,
) -> ProfileReport:
    """
    Convenience function to generate profile report.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to profile
    target_column : str, optional
        Target variable column name
    theme : PlotTheme or str
        Theme for visualizations
    **kwargs
        Additional arguments passed to DataProfiler.generate_profile()

    Returns
    -------
    ProfileReport
        Comprehensive profiling results
    """
    profiler = DataProfiler(theme=theme)
    return profiler.generate_profile(data, target_column=target_column, **kwargs)
