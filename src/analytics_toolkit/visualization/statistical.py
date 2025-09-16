"""
Statistical visualization components for data analysis and exploration.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .themes import PlotTheme, apply_theme


class StatisticalPlots:
    """Main class for creating statistical visualizations."""

    def __init__(self, theme: PlotTheme | str = "default"):
        """Initialize with theme."""
        self.theme = apply_theme(theme)

    def plot_distribution(
        self,
        data: pd.Series | np.ndarray | list,
        column: str | None = None,
        bins: int = 30,
        kde: bool = True,
        normal_overlay: bool = False,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot distribution of a variable with optional overlays.

        Parameters
        ----------
        data : pd.Series, np.ndarray, or List
            Data to plot
        column : str, optional
            Column name for labeling
        bins : int
            Number of histogram bins
        kde : bool
            Whether to overlay kernel density estimation
        normal_overlay : bool
            Whether to overlay normal distribution
        figsize : tuple
            Figure size
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            title or f"Distribution Analysis: {column or 'Variable'}", fontsize=14
        )

        # Convert to numpy array
        if isinstance(data, pd.Series):
            values = data.dropna().values
            column = column or data.name
        else:
            values = np.array(data)
            values = values[~np.isnan(values)]

        # Histogram with KDE
        axes[0, 0].hist(
            values,
            bins=bins,
            alpha=0.7,
            density=True,
            color=self.theme.colors["primary"],
        )
        if kde:
            try:
                from scipy.stats import gaussian_kde

                kde_func = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                axes[0, 0].plot(
                    x_range,
                    kde_func(x_range),
                    color=self.theme.colors["accent"],
                    linewidth=2,
                )
            except ImportError:
                pass

        if normal_overlay:
            x_range = np.linspace(values.min(), values.max(), 200)
            normal_dist = stats.norm.pdf(x_range, values.mean(), values.std())
            axes[0, 0].plot(
                x_range,
                normal_dist,
                "--",
                color=self.theme.colors["warning"],
                linewidth=2,
            )

        axes[0, 0].set_title("Histogram with Density")
        axes[0, 0].set_xlabel(column or "Value")
        axes[0, 0].set_ylabel("Density")

        # Box plot
        axes[0, 1].boxplot(
            values,
            patch_artist=True,
            boxprops=dict(facecolor=self.theme.colors["secondary"]),
        )
        axes[0, 1].set_title("Box Plot")
        axes[0, 1].set_ylabel(column or "Value")

        # Q-Q plot
        stats.probplot(values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal)")
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        axes[1, 1].axis("off")
        stats_text = self._get_summary_statistics(values)
        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
        )

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = "pearson",
        figsize: tuple[int, int] = (12, 10),
        annot: bool = True,
        cmap: str = "RdBu_r",
        title: str | None = None,
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Parameters
        ----------
        data : pd.DataFrame
            Data to compute correlations
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        figsize : tuple
            Figure size
        annot : bool
            Whether to annotate cells with correlation values
        cmap : str
            Colormap name
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Compute correlation matrix
        corr_matrix = data.select_dtypes(include=[np.number]).corr(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            cmap=cmap,
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(title or f"{method.capitalize()} Correlation Matrix", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_pairwise_relationships(
        self,
        data: pd.DataFrame,
        variables: list[str] | None = None,
        hue: str | None = None,
        kind: str = "scatter",
        figsize: tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Plot pairwise relationships between variables.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        variables : list, optional
            List of variables to plot. If None, uses all numeric columns
        hue : str, optional
            Variable to use for color encoding
        kind : str
            Kind of plot ('scatter', 'reg', 'hist')
        figsize : tuple
            Figure size

        Returns
        -------
        plt.Figure
            The created figure
        """
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(variables) > 6:  # Limit to avoid overcrowded plots
                variables = variables[:6]

        plot_data = data[variables + ([hue] if hue else [])]

        try:
            g = sns.PairGrid(plot_data, hue=hue, height=2.5)

            if kind == "scatter":
                g.map_upper(plt.scatter, alpha=0.6)
                g.map_lower(sns.scatterplot, alpha=0.6)
                g.map_diag(plt.hist, alpha=0.7)
            elif kind == "reg":
                g.map_upper(sns.regplot, scatter_kws={"alpha": 0.6})
                g.map_lower(sns.regplot, scatter_kws={"alpha": 0.6})
                g.map_diag(plt.hist, alpha=0.7)
            else:  # hist
                g.map_upper(plt.scatter, alpha=0.6)
                g.map_lower(plt.scatter, alpha=0.6)
                g.map_diag(plt.hist, alpha=0.7)

            if hue:
                g.add_legend()

            plt.suptitle("Pairwise Relationships", y=1.02, fontsize=14)
            return g.fig

        except Exception as e:
            # Fallback to basic scatter plots
            warnings.warn(
                f"Could not create pairwise plot: {e}. Using basic scatter plots."
            )
            n_vars = len(variables)
            fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)

            if n_vars == 1:
                axes = np.array([[axes]])
            elif n_vars == 2:
                axes = axes.reshape(2, 2)

            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i == j:
                        axes[i, j].hist(
                            data[var1].dropna(),
                            alpha=0.7,
                            color=self.theme.colors["primary"],
                        )
                    else:
                        axes[i, j].scatter(
                            data[var2],
                            data[var1],
                            alpha=0.6,
                            color=self.theme.colors["primary"],
                        )

                    if i == n_vars - 1:
                        axes[i, j].set_xlabel(var2)
                    if j == 0:
                        axes[i, j].set_ylabel(var1)

            plt.suptitle("Pairwise Relationships", fontsize=14)
            plt.tight_layout()
            return fig

    def _get_summary_statistics(self, values: np.ndarray) -> str:
        """Get formatted summary statistics string."""
        stats_dict = {
            "Count": len(values),
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Std Dev": np.std(values),
            "Min": np.min(values),
            "Max": np.max(values),
            "Skewness": stats.skew(values),
            "Kurtosis": stats.kurtosis(values),
        }

        text_lines = ["Summary Statistics:"]
        for key, value in stats_dict.items():
            if key == "Count":
                text_lines.append(f"{key}: {value:,}")
            else:
                text_lines.append(f"{key}: {value:.3f}")

        return "\n".join(text_lines)


class DistributionPlots:
    """Specialized class for distribution analysis."""

    def __init__(self, theme: PlotTheme | str = "default"):
        self.theme = apply_theme(theme)

    def compare_distributions(
        self,
        data1: pd.Series | np.ndarray,
        data2: pd.Series | np.ndarray,
        labels: tuple[str, str] = ("Group 1", "Group 2"),
        figsize: tuple[int, int] = (12, 8),
        statistical_test: bool = True,
    ) -> plt.Figure:
        """
        Compare two distributions visually and statistically.

        Parameters
        ----------
        data1, data2 : array-like
            Data arrays to compare
        labels : tuple
            Labels for the two groups
        figsize : tuple
            Figure size
        statistical_test : bool
            Whether to perform statistical tests

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Distribution Comparison: {labels[0]} vs {labels[1]}", fontsize=14
        )

        # Clean data
        data1_clean = np.array(data1)[~np.isnan(np.array(data1))]
        data2_clean = np.array(data2)[~np.isnan(np.array(data2))]

        # Overlapping histograms
        axes[0, 0].hist(
            data1_clean,
            alpha=0.7,
            label=labels[0],
            bins=30,
            color=self.theme.colors["primary"],
        )
        axes[0, 0].hist(
            data2_clean,
            alpha=0.7,
            label=labels[1],
            bins=30,
            color=self.theme.colors["secondary"],
        )
        axes[0, 0].set_title("Overlapping Histograms")
        axes[0, 0].legend()

        # Box plots
        axes[0, 1].boxplot(
            [data1_clean, data2_clean],
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor=self.theme.colors["primary"]),
        )
        axes[0, 1].set_title("Box Plot Comparison")

        # Q-Q plots
        stats.probplot(data1_clean, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f"Q-Q Plot: {labels[0]}")

        # Statistical test results
        axes[1, 1].axis("off")
        if statistical_test:
            test_results = self._perform_distribution_tests(data1_clean, data2_clean)
            axes[1, 1].text(
                0.1,
                0.9,
                test_results,
                transform=axes[1, 1].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
            )

        plt.tight_layout()
        return fig

    def _perform_distribution_tests(self, data1: np.ndarray, data2: np.ndarray) -> str:
        """Perform statistical tests to compare distributions."""
        results = ["Statistical Tests:"]

        # Two-sample t-test
        t_stat, t_pvalue = stats.ttest_ind(data1, data2)
        results.append(f"T-test: t={t_stat:.3f}, p={t_pvalue:.3f}")

        # Mann-Whitney U test
        try:
            u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            results.append(f"Mann-Whitney U: U={u_stat:.3f}, p={u_pvalue:.3f}")
        except ValueError:
            results.append("Mann-Whitney U: Could not compute")

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
        results.append(f"K-S test: D={ks_stat:.3f}, p={ks_pvalue:.3f}")

        # Effect size (Cohen's d)
        cohens_d = (np.mean(data1) - np.mean(data2)) / np.sqrt(
            (
                (len(data1) - 1) * np.var(data1, ddof=1)
                + (len(data2) - 1) * np.var(data2, ddof=1)
            )
            / (len(data1) + len(data2) - 2)
        )
        results.append(f"Cohen's d: {cohens_d:.3f}")

        return "\n".join(results)


class CorrelationPlots:
    """Specialized class for correlation analysis."""

    def __init__(self, theme: PlotTheme | str = "default"):
        self.theme = apply_theme(theme)

    def plot_correlation_network(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5,
        figsize: tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Plot correlation network showing strong correlations.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        threshold : float
            Minimum correlation strength to display
        figsize : tuple
            Figure size

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Compute correlation matrix
        corr_matrix = data.select_dtypes(include=[np.number]).corr()

        # Find strong correlations
        strong_corr = []
        variables = corr_matrix.columns.tolist()

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j and abs(corr_matrix.loc[var1, var2]) >= threshold:
                    strong_corr.append(
                        {
                            "var1": var1,
                            "var2": var2,
                            "correlation": corr_matrix.loc[var1, var2],
                        }
                    )

        if not strong_corr:
            ax.text(
                0.5,
                0.5,
                f"No correlations above threshold {threshold}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Correlation Network")
            return fig

        # Simple network visualization
        # Position variables in a circle
        n_vars = len(variables)
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
        positions = {
            var: (np.cos(angle), np.sin(angle)) for var, angle in zip(variables, angles, strict=False)
        }

        # Draw connections
        for corr in strong_corr:
            var1, var2 = corr["var1"], corr["var2"]
            x1, y1 = positions[var1]
            x2, y2 = positions[var2]

            color = (
                self.theme.colors["primary"]
                if corr["correlation"] > 0
                else self.theme.colors["warning"]
            )
            alpha = min(abs(corr["correlation"]), 1.0)
            width = abs(corr["correlation"]) * 3

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                alpha=alpha,
                linewidth=width,
                solid_capstyle="round",
            )

        # Draw variable points
        for var, (x, y) in positions.items():
            ax.scatter(
                x, y, s=200, color=self.theme.colors["secondary"], alpha=0.7, zorder=5
            )
            ax.annotate(
                var,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                ha="left",
            )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_title(f"Correlation Network (threshold: {threshold})")
        ax.axis("off")

        return fig


def plot_distribution(data: pd.Series | np.ndarray | list, **kwargs) -> plt.Figure:
    """Convenience function for plotting distributions."""
    plotter = StatisticalPlots()
    return plotter.plot_distribution(data, **kwargs)


def plot_correlation_matrix(data: pd.DataFrame, **kwargs) -> plt.Figure:
    """Convenience function for plotting correlation matrices."""
    plotter = StatisticalPlots()
    return plotter.plot_correlation_matrix(data, **kwargs)


def plot_pairwise_relationships(data: pd.DataFrame, **kwargs) -> plt.Figure:
    """Convenience function for plotting pairwise relationships."""
    plotter = StatisticalPlots()
    return plotter.plot_pairwise_relationships(data, **kwargs)
