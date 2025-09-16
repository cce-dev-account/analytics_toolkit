"""
Theming system for consistent visualization styling across the analytics toolkit.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class PlotTheme(ABC):
    """Base class for plot themes."""

    name: str
    colors: dict[str, str]
    font_family: str
    font_size: int
    figure_facecolor: str
    axes_facecolor: str
    grid_color: str
    grid_alpha: float
    spine_color: str
    text_color: str

    @abstractmethod
    def apply_matplotlib_style(self) -> None:
        """Apply theme to matplotlib."""
        pass

    def get_color_palette(self, n_colors: int = 10) -> list:
        """Get color palette with n_colors."""
        base_colors = list(self.colors.values())
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]

        # Repeat colors if more needed
        palette = []
        for i in range(n_colors):
            palette.append(base_colors[i % len(base_colors)])
        return palette


class DefaultTheme(PlotTheme):
    """Default analytics toolkit theme."""

    def __init__(self):
        super().__init__(
            name="default",
            colors={
                "primary": "#2E86C1",
                "secondary": "#28B463",
                "accent": "#F39C12",
                "warning": "#E74C3C",
                "info": "#8E44AD",
                "success": "#27AE60",
                "neutral": "#566573",
            },
            font_family="Arial",
            font_size=10,
            figure_facecolor="white",
            axes_facecolor="white",
            grid_color="#E5E5E5",
            grid_alpha=0.6,
            spine_color="#CCCCCC",
            text_color="#2C3E50",
        )

    def apply_matplotlib_style(self) -> None:
        """Apply default theme to matplotlib."""
        plt.rcParams.update(
            {
                "figure.facecolor": self.figure_facecolor,
                "axes.facecolor": self.axes_facecolor,
                "axes.edgecolor": self.spine_color,
                "axes.linewidth": 0.8,
                "axes.grid": True,
                "axes.grid.axis": "both",
                "grid.color": self.grid_color,
                "grid.alpha": self.grid_alpha,
                "grid.linewidth": 0.5,
                "font.family": self.font_family,
                "font.size": self.font_size,
                "text.color": self.text_color,
                "axes.labelcolor": self.text_color,
                "xtick.color": self.text_color,
                "ytick.color": self.text_color,
                "axes.prop_cycle": plt.cycler("color", self.get_color_palette()),
            }
        )


class MinimalTheme(PlotTheme):
    """Minimal clean theme."""

    def __init__(self):
        super().__init__(
            name="minimal",
            colors={
                "primary": "#34495E",
                "secondary": "#7F8C8D",
                "accent": "#3498DB",
                "warning": "#E67E22",
                "info": "#9B59B6",
                "success": "#2ECC71",
                "neutral": "#95A5A6",
            },
            font_family="Helvetica",
            font_size=9,
            figure_facecolor="white",
            axes_facecolor="white",
            grid_color="#F8F9FA",
            grid_alpha=0.8,
            spine_color="#E9ECEF",
            text_color="#212529",
        )

    def apply_matplotlib_style(self) -> None:
        """Apply minimal theme to matplotlib."""
        plt.rcParams.update(
            {
                "figure.facecolor": self.figure_facecolor,
                "axes.facecolor": self.axes_facecolor,
                "axes.edgecolor": self.spine_color,
                "axes.linewidth": 0.5,
                "axes.grid": True,
                "grid.color": self.grid_color,
                "grid.alpha": self.grid_alpha,
                "grid.linewidth": 0.3,
                "font.family": self.font_family,
                "font.size": self.font_size,
                "text.color": self.text_color,
                "axes.labelcolor": self.text_color,
                "xtick.color": self.text_color,
                "ytick.color": self.text_color,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.prop_cycle": plt.cycler("color", self.get_color_palette()),
            }
        )


class DarkTheme(PlotTheme):
    """Dark theme for low-light environments."""

    def __init__(self):
        super().__init__(
            name="dark",
            colors={
                "primary": "#3498DB",
                "secondary": "#2ECC71",
                "accent": "#F39C12",
                "warning": "#E74C3C",
                "info": "#9B59B6",
                "success": "#27AE60",
                "neutral": "#BDC3C7",
            },
            font_family="Arial",
            font_size=10,
            figure_facecolor="#2C3E50",
            axes_facecolor="#34495E",
            grid_color="#566573",
            grid_alpha=0.3,
            spine_color="#7F8C8D",
            text_color="#ECF0F1",
        )

    def apply_matplotlib_style(self) -> None:
        """Apply dark theme to matplotlib."""
        plt.rcParams.update(
            {
                "figure.facecolor": self.figure_facecolor,
                "axes.facecolor": self.axes_facecolor,
                "axes.edgecolor": self.spine_color,
                "axes.linewidth": 0.8,
                "axes.grid": True,
                "grid.color": self.grid_color,
                "grid.alpha": self.grid_alpha,
                "grid.linewidth": 0.5,
                "font.family": self.font_family,
                "font.size": self.font_size,
                "text.color": self.text_color,
                "axes.labelcolor": self.text_color,
                "xtick.color": self.text_color,
                "ytick.color": self.text_color,
                "axes.prop_cycle": plt.cycler("color", self.get_color_palette()),
            }
        )


def apply_theme(theme: PlotTheme | str = "default") -> PlotTheme:
    """
    Apply a theme to matplotlib globally.

    Parameters
    ----------
    theme : PlotTheme or str
        Theme to apply. Can be PlotTheme instance or string name.

    Returns
    -------
    PlotTheme
        Applied theme instance.
    """
    if isinstance(theme, str):
        theme_map = {
            "default": DefaultTheme(),
            "minimal": MinimalTheme(),
            "dark": DarkTheme(),
        }
        if theme not in theme_map:
            raise ValueError(
                f"Unknown theme: {theme}. Available: {list(theme_map.keys())}"
            )
        theme = theme_map[theme]

    theme.apply_matplotlib_style()
    return theme


def get_theme_colors(
    theme: PlotTheme | str = "default", n_colors: int = 10
) -> list:
    """
    Get color palette from theme.

    Parameters
    ----------
    theme : PlotTheme or str
        Theme to get colors from.
    n_colors : int
        Number of colors to return.

    Returns
    -------
    list
        List of color strings.
    """
    if isinstance(theme, str):
        theme_map = {
            "default": DefaultTheme(),
            "minimal": MinimalTheme(),
            "dark": DarkTheme(),
        }
        theme = theme_map.get(theme, DefaultTheme())

    return theme.get_color_palette(n_colors)


def reset_theme() -> None:
    """Reset matplotlib to default settings."""
    plt.rcdefaults()
