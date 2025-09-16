"""
Interactive plotting capabilities with Plotly and Bokeh backends.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from .themes import DefaultTheme, PlotTheme


class PlottingBackend(ABC):
    """Abstract base class for plotting backends."""

    @abstractmethod
    def scatter_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """Create scatter plot."""
        pass

    @abstractmethod
    def line_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """Create line plot."""
        pass

    @abstractmethod
    def bar_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """Create bar plot."""
        pass

    @abstractmethod
    def histogram(self, data: np.ndarray, **kwargs) -> Any:
        """Create histogram."""
        pass

    @abstractmethod
    def box_plot(self, data: np.ndarray, **kwargs) -> Any:
        """Create box plot."""
        pass


class PlotlyBackend(PlottingBackend):
    """Plotly backend for interactive plots."""

    def __init__(self):
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            self.go = go
            self.px = px
            self.available = True
        except ImportError:
            warnings.warn("Plotly not available. Install with: pip install plotly")
            self.available = False

    def scatter_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: np.ndarray | None = None,
        size: np.ndarray | None = None,
        hover_data: dict | None = None,
        title: str = "Scatter Plot",
        **kwargs,
    ) -> Any:
        """Create interactive scatter plot."""
        if not self.available:
            raise ImportError("Plotly not available")

        fig = self.go.Figure()

        scatter_kwargs = {
            "x": x,
            "y": y,
            "mode": "markers",
            "marker": dict(
                size=size if size is not None else 8,
                color=color if color is not None else "blue",
                showscale=color is not None,
                colorscale="Viridis" if color is not None else None,
            ),
            "hovertemplate": "<b>X</b>: %{x}<br><b>Y</b>: %{y}<extra></extra>",
        }

        if hover_data:
            customdata = np.column_stack([hover_data[key] for key in hover_data.keys()])
            scatter_kwargs["customdata"] = customdata
            hover_template = "<b>X</b>: %{x}<br><b>Y</b>: %{y}<br>"
            for i, key in enumerate(hover_data.keys()):
                hover_template += f"<b>{key}</b>: %{{customdata[{i}]}}<br>"
            scatter_kwargs["hovertemplate"] = hover_template + "<extra></extra>"

        fig.add_trace(self.go.Scatter(**scatter_kwargs))

        fig.update_layout(
            title=title, xaxis_title="X", yaxis_title="Y", hovermode="closest"
        )

        return fig

    def line_plot(
        self, x: np.ndarray, y: np.ndarray, title: str = "Line Plot", **kwargs
    ) -> Any:
        """Create interactive line plot."""
        if not self.available:
            raise ImportError("Plotly not available")

        fig = self.go.Figure()
        fig.add_trace(
            self.go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title=title, xaxis_title="X", yaxis_title="Y", hovermode="x unified"
        )

        return fig

    def bar_plot(
        self, x: np.ndarray, y: np.ndarray, title: str = "Bar Plot", **kwargs
    ) -> Any:
        """Create interactive bar plot."""
        if not self.available:
            raise ImportError("Plotly not available")

        fig = self.go.Figure()
        fig.add_trace(self.go.Bar(x=x, y=y, marker_color="blue"))

        fig.update_layout(title=title, xaxis_title="Category", yaxis_title="Value")

        return fig

    def histogram(
        self, data: np.ndarray, bins: int = 30, title: str = "Histogram", **kwargs
    ) -> Any:
        """Create interactive histogram."""
        if not self.available:
            raise ImportError("Plotly not available")

        fig = self.go.Figure()
        fig.add_trace(
            self.go.Histogram(x=data, nbinsx=bins, marker_color="blue", opacity=0.7)
        )

        fig.update_layout(title=title, xaxis_title="Value", yaxis_title="Frequency")

        return fig

    def box_plot(self, data: np.ndarray, title: str = "Box Plot", **kwargs) -> Any:
        """Create interactive box plot."""
        if not self.available:
            raise ImportError("Plotly not available")

        fig = self.go.Figure()
        fig.add_trace(self.go.Box(y=data, marker_color="blue", boxpoints="outliers"))

        fig.update_layout(title=title, yaxis_title="Value")

        return fig

    def correlation_heatmap(
        self, corr_matrix: pd.DataFrame, title: str = "Correlation Heatmap", **kwargs
    ) -> Any:
        """Create interactive correlation heatmap."""
        if not self.available:
            raise ImportError("Plotly not available")

        fig = self.go.Figure(
            data=self.go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(title=title, xaxis_title="Variables", yaxis_title="Variables")

        return fig


class BokehBackend(PlottingBackend):
    """Bokeh backend for interactive plots."""

    def __init__(self):
        try:
            from bokeh.io import output_notebook, push_notebook, show
            from bokeh.models import HoverTool
            from bokeh.plotting import figure

            self.figure = figure
            self.HoverTool = HoverTool
            self.show = show
            self.output_notebook = output_notebook
            self.push_notebook = push_notebook
            self.available = True
        except ImportError:
            warnings.warn("Bokeh not available. Install with: pip install bokeh")
            self.available = False

    def scatter_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: np.ndarray | None = None,
        size: np.ndarray | None = None,
        title: str = "Scatter Plot",
        **kwargs,
    ) -> Any:
        """Create interactive scatter plot."""
        if not self.available:
            raise ImportError("Bokeh not available")

        p = self.figure(
            title=title, x_axis_label="X", y_axis_label="Y", width=700, height=500
        )

        scatter_kwargs = {
            "x": x,
            "y": y,
            "size": size if size is not None else 8,
            "color": color if color is not None else "blue",
            "alpha": 0.7,
        }

        p.circle(**scatter_kwargs)

        # Add hover tool
        hover = self.HoverTool(tooltips=[("X", "@x"), ("Y", "@y")])
        p.add_tools(hover)

        return p

    def line_plot(
        self, x: np.ndarray, y: np.ndarray, title: str = "Line Plot", **kwargs
    ) -> Any:
        """Create interactive line plot."""
        if not self.available:
            raise ImportError("Bokeh not available")

        p = self.figure(
            title=title, x_axis_label="X", y_axis_label="Y", width=700, height=500
        )

        p.line(x, y, line_width=2, color="blue")
        p.circle(x, y, size=6, color="blue", alpha=0.8)

        return p

    def bar_plot(
        self, x: np.ndarray, y: np.ndarray, title: str = "Bar Plot", **kwargs
    ) -> Any:
        """Create interactive bar plot."""
        if not self.available:
            raise ImportError("Bokeh not available")

        p = self.figure(
            x_range=x,
            title=title,
            x_axis_label="Category",
            y_axis_label="Value",
            width=700,
            height=500,
        )

        p.vbar(x=x, top=y, width=0.8, color="blue", alpha=0.7)

        return p

    def histogram(
        self, data: np.ndarray, bins: int = 30, title: str = "Histogram", **kwargs
    ) -> Any:
        """Create interactive histogram."""
        if not self.available:
            raise ImportError("Bokeh not available")

        hist, edges = np.histogram(data, bins=bins)
        p = self.figure(
            title=title,
            x_axis_label="Value",
            y_axis_label="Frequency",
            width=700,
            height=500,
        )

        p.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="blue",
            line_color="white",
            alpha=0.7,
        )

        return p

    def box_plot(self, data: np.ndarray, title: str = "Box Plot", **kwargs) -> Any:
        """Create interactive box plot."""
        if not self.available:
            raise ImportError("Bokeh not available")

        # Calculate box plot statistics
        q1 = np.percentile(data, 25)
        q2 = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        p = self.figure(title=title, y_axis_label="Value", width=400, height=500)

        # Box
        p.quad(top=q3, bottom=q1, left=0.7, right=1.3, fill_color="blue", alpha=0.7)

        # Median line
        p.line([0.7, 1.3], [q2, q2], line_width=2, color="white")

        # Whiskers
        p.line([1, 1], [q3, upper], line_width=2, color="black")
        p.line([1, 1], [q1, lower], line_width=2, color="black")

        # Outliers
        outliers = data[(data > upper) | (data < lower)]
        if len(outliers) > 0:
            p.circle([1] * len(outliers), outliers, size=6, color="red", alpha=0.6)

        return p


class InteractivePlotter:
    """Main class for creating interactive visualizations."""

    def __init__(
        self, backend: str = "plotly", theme: PlotTheme | str = "default"
    ):
        """
        Initialize interactive plotter.

        Parameters
        ----------
        backend : str
            Plotting backend ('plotly' or 'bokeh')
        theme : PlotTheme or str
            Theme to use for styling
        """
        self.backend_name = backend
        self.theme = DefaultTheme() if isinstance(theme, str) else theme

        if backend == "plotly":
            self.backend = PlotlyBackend()
        elif backend == "bokeh":
            self.backend = BokehBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        size: str | None = None,
        title: str | None = None,
        **kwargs,
    ) -> Any:
        """
        Create interactive scatter plot.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        x, y : str
            Column names for x and y axes
        color : str, optional
            Column name for color encoding
        size : str, optional
            Column name for size encoding
        title : str, optional
            Plot title

        Returns
        -------
        Plot object (backend-specific)
        """
        x_data = data[x].values
        y_data = data[y].values
        color_data = data[color].values if color else None
        size_data = data[size].values if size else None

        title = title or f"{y} vs {x}"

        # Add hover data for additional columns
        hover_data = {}
        for col in data.columns:
            if col not in [x, y, color, size]:
                hover_data[col] = data[col].values

        return self.backend.scatter_plot(
            x_data,
            y_data,
            color=color_data,
            size=size_data,
            hover_data=hover_data,
            title=title,
            **kwargs,
        )

    def line_plot(
        self, data: pd.DataFrame, x: str, y: str, title: str | None = None, **kwargs
    ) -> Any:
        """Create interactive line plot."""
        x_data = data[x].values
        y_data = data[y].values
        title = title or f"{y} over {x}"

        return self.backend.line_plot(x_data, y_data, title=title, **kwargs)

    def bar_plot(
        self, data: pd.DataFrame, x: str, y: str, title: str | None = None, **kwargs
    ) -> Any:
        """Create interactive bar plot."""
        x_data = data[x].values
        y_data = data[y].values
        title = title or f"{y} by {x}"

        return self.backend.bar_plot(x_data, y_data, title=title, **kwargs)

    def histogram(
        self,
        data: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: str | None = None,
        **kwargs,
    ) -> Any:
        """Create interactive histogram."""
        column_data = data[column].dropna().values
        title = title or f"Distribution of {column}"

        return self.backend.histogram(column_data, bins=bins, title=title, **kwargs)

    def box_plot(
        self, data: pd.DataFrame, column: str, title: str | None = None, **kwargs
    ) -> Any:
        """Create interactive box plot."""
        column_data = data[column].dropna().values
        title = title or f"Box Plot of {column}"

        return self.backend.box_plot(column_data, title=title, **kwargs)

    def correlation_heatmap(
        self, data: pd.DataFrame, title: str | None = None, **kwargs
    ) -> Any:
        """Create interactive correlation heatmap."""
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        title = title or "Correlation Matrix"

        if hasattr(self.backend, "correlation_heatmap"):
            return self.backend.correlation_heatmap(corr_matrix, title=title, **kwargs)
        else:
            raise NotImplementedError(
                f"Correlation heatmap not implemented for {self.backend_name}"
            )

    def show(self, plot):
        """Display the plot."""
        if self.backend_name == "plotly":
            plot.show()
        elif self.backend_name == "bokeh":
            self.backend.show(plot)
