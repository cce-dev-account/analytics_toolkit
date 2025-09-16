"""
Time-based feature engineering for temporal data analysis.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract comprehensive datetime features from timestamp columns.

    Parameters
    ----------
    features : list, default='all'
        Features to extract: 'all' or list of specific features
    cyclical_encoding : bool, default=True
        Whether to use cyclical encoding for periodic features
    drop_original : bool, default=True
        Whether to drop original datetime columns
    """

    def __init__(self, features="all", cyclical_encoding=True, drop_original=True):
        self.features = features
        self.cyclical_encoding = cyclical_encoding
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """Fit the datetime feature extractor."""
        if isinstance(X, pd.DataFrame):
            self.datetime_columns_ = X.select_dtypes(
                include=["datetime64"]
            ).columns.tolist()
        else:
            # Assume all columns are datetime if numpy array
            self.datetime_columns_ = list(range(X.shape[1]))

        if self.features == "all":
            self.features_ = [
                "year",
                "month",
                "day",
                "dayofweek",
                "dayofyear",
                "hour",
                "minute",
                "second",
                "quarter",
                "week",
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "is_year_start",
                "is_year_end",
            ]
        else:
            self.features_ = self.features

        return self

    def transform(self, X):
        """Transform by extracting datetime features."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            result = pd.DataFrame(X)

        new_features = {}

        for col in self.datetime_columns_:
            if isinstance(X, pd.DataFrame):
                dt_series = pd.to_datetime(result[col])
            else:
                dt_series = pd.to_datetime(result.iloc[:, col])

            col_name = col if isinstance(col, str) else f"datetime_{col}"

            # Extract features
            for feature in self.features_:
                feature_name = f"{col_name}_{feature}"

                if feature == "year":
                    new_features[feature_name] = dt_series.dt.year
                elif feature == "month":
                    values = dt_series.dt.month
                    if self.cyclical_encoding:
                        new_features[f"{feature_name}_sin"] = np.sin(
                            2 * np.pi * values / 12
                        )
                        new_features[f"{feature_name}_cos"] = np.cos(
                            2 * np.pi * values / 12
                        )
                    else:
                        new_features[feature_name] = values
                elif feature == "day":
                    values = dt_series.dt.day
                    if self.cyclical_encoding:
                        new_features[f"{feature_name}_sin"] = np.sin(
                            2 * np.pi * values / 31
                        )
                        new_features[f"{feature_name}_cos"] = np.cos(
                            2 * np.pi * values / 31
                        )
                    else:
                        new_features[feature_name] = values
                elif feature == "dayofweek":
                    values = dt_series.dt.dayofweek
                    if self.cyclical_encoding:
                        new_features[f"{feature_name}_sin"] = np.sin(
                            2 * np.pi * values / 7
                        )
                        new_features[f"{feature_name}_cos"] = np.cos(
                            2 * np.pi * values / 7
                        )
                    else:
                        new_features[feature_name] = values
                elif feature == "hour":
                    values = dt_series.dt.hour
                    if self.cyclical_encoding:
                        new_features[f"{feature_name}_sin"] = np.sin(
                            2 * np.pi * values / 24
                        )
                        new_features[f"{feature_name}_cos"] = np.cos(
                            2 * np.pi * values / 24
                        )
                    else:
                        new_features[feature_name] = values
                elif feature == "minute":
                    values = dt_series.dt.minute
                    if self.cyclical_encoding:
                        new_features[f"{feature_name}_sin"] = np.sin(
                            2 * np.pi * values / 60
                        )
                        new_features[f"{feature_name}_cos"] = np.cos(
                            2 * np.pi * values / 60
                        )
                    else:
                        new_features[feature_name] = values
                elif feature == "second":
                    new_features[feature_name] = dt_series.dt.second
                elif feature == "dayofyear":
                    values = dt_series.dt.dayofyear
                    if self.cyclical_encoding:
                        new_features[f"{feature_name}_sin"] = np.sin(
                            2 * np.pi * values / 366
                        )
                        new_features[f"{feature_name}_cos"] = np.cos(
                            2 * np.pi * values / 366
                        )
                    else:
                        new_features[feature_name] = values
                elif feature == "quarter":
                    new_features[feature_name] = dt_series.dt.quarter
                elif feature == "week":
                    new_features[feature_name] = dt_series.dt.isocalendar().week
                elif feature == "is_weekend":
                    new_features[feature_name] = (dt_series.dt.dayofweek >= 5).astype(
                        int
                    )
                elif feature == "is_month_start":
                    new_features[feature_name] = dt_series.dt.is_month_start.astype(int)
                elif feature == "is_month_end":
                    new_features[feature_name] = dt_series.dt.is_month_end.astype(int)
                elif feature == "is_quarter_start":
                    new_features[feature_name] = dt_series.dt.is_quarter_start.astype(
                        int
                    )
                elif feature == "is_quarter_end":
                    new_features[feature_name] = dt_series.dt.is_quarter_end.astype(int)
                elif feature == "is_year_start":
                    new_features[feature_name] = dt_series.dt.is_year_start.astype(int)
                elif feature == "is_year_end":
                    new_features[feature_name] = dt_series.dt.is_year_end.astype(int)

        # Add new features to result
        for feature_name, feature_values in new_features.items():
            result[feature_name] = feature_values

        # Drop original datetime columns if requested
        if self.drop_original:
            result = result.drop(columns=self.datetime_columns_)

        return result


class LagFeatures(BaseEstimator, TransformerMixin):
    """
    Create lag features for time series data.

    Parameters
    ----------
    lags : list or int, default=[1, 2, 3]
        Lag periods to create
    columns : list, default=None
        Columns to create lags for. If None, uses all numeric columns
    fill_value : float, default=0
        Value to fill initial NaN values
    """

    def __init__(self, lags=None, columns=None, fill_value=0):
        if lags is None:
            lags = [1, 2, 3]
        self.lags = lags if isinstance(lags, list) else [lags]
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """Fit the lag feature creator."""
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                self.columns_ = self.columns
        else:
            if self.columns is None:
                self.columns_ = list(range(X.shape[1]))
            else:
                self.columns_ = self.columns

        return self

    def transform(self, X):
        """Transform by creating lag features."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            result = pd.DataFrame(X)

        # Create lag features
        for col in self.columns_:
            for lag in self.lags:
                lag_col_name = f"{col}_lag_{lag}"
                if isinstance(X, pd.DataFrame):
                    result[lag_col_name] = (
                        result[col].shift(lag).fillna(self.fill_value)
                    )
                else:
                    # For numpy arrays converted to DataFrame, use column index
                    result[lag_col_name] = (
                        result[col].shift(lag).fillna(self.fill_value)
                    )

        return result


class RollingFeatures(BaseEstimator, TransformerMixin):
    """
    Create rolling window statistical features.

    Parameters
    ----------
    windows : list, default=[3, 7, 14]
        Window sizes for rolling calculations
    statistics : list, default=['mean', 'std', 'min', 'max']
        Statistics to calculate for each window
    columns : list, default=None
        Columns to create rolling features for
    min_periods : int, default=1
        Minimum number of observations required for calculation
    """

    def __init__(
        self,
        windows=None,
        statistics=None,
        columns=None,
        min_periods=1,
    ):
        if statistics is None:
            statistics = ["mean", "std", "min", "max"]
        if windows is None:
            windows = [3, 7, 14]
        self.windows = windows if isinstance(windows, list) else [windows]
        self.statistics = statistics
        self.columns = columns
        self.min_periods = min_periods

    def fit(self, X, y=None):
        """Fit the rolling feature creator."""
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                self.columns_ = self.columns
        else:
            if self.columns is None:
                self.columns_ = list(range(X.shape[1]))
            else:
                self.columns_ = self.columns

        return self

    def transform(self, X):
        """Transform by creating rolling features."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            result = pd.DataFrame(X)

        # Create rolling features
        for col in self.columns_:
            for window in self.windows:
                for stat in self.statistics:
                    feature_name = f"{col}_rolling_{window}_{stat}"

                    if isinstance(X, pd.DataFrame):
                        series = result[col]
                    else:
                        series = result.iloc[:, col]

                    rolling_obj = series.rolling(
                        window=window, min_periods=self.min_periods
                    )

                    if stat == "mean":
                        result[feature_name] = rolling_obj.mean()
                    elif stat == "std":
                        result[feature_name] = rolling_obj.std()
                    elif stat == "min":
                        result[feature_name] = rolling_obj.min()
                    elif stat == "max":
                        result[feature_name] = rolling_obj.max()
                    elif stat == "median":
                        result[feature_name] = rolling_obj.median()
                    elif stat == "sum":
                        result[feature_name] = rolling_obj.sum()
                    elif stat == "var":
                        result[feature_name] = rolling_obj.var()
                    elif stat == "skew":
                        result[feature_name] = rolling_obj.skew()
                    elif stat == "kurt":
                        result[feature_name] = rolling_obj.kurt()

        return result


class SeasonalDecomposition(BaseEstimator, TransformerMixin):
    """
    Decompose time series into trend, seasonal, and residual components.

    Parameters
    ----------
    period : int, default=None
        Seasonal period. If None, attempts to detect automatically
    model : str, default='additive'
        Type of decomposition: 'additive' or 'multiplicative'
    columns : list, default=None
        Columns to decompose
    """

    def __init__(self, period=None, model="additive", columns=None):
        self.period = period
        self.model = model
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the seasonal decomposition."""
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                self.columns_ = self.columns
        else:
            if self.columns is None:
                self.columns_ = list(range(X.shape[1]))
            else:
                self.columns_ = self.columns

        # Auto-detect period if not provided
        if self.period is None:
            # Simple heuristic: assume daily data and detect weekly/monthly patterns
            self.period_ = self._detect_period(X)
        else:
            self.period_ = self.period

        return self

    def _detect_period(self, X):
        """Detect seasonal period using autocorrelation."""
        try:
            from statsmodels.tsa.stattools import acf

            # Use first numeric column for period detection
            if isinstance(X, pd.DataFrame):
                series = X[self.columns_[0]].dropna()
            else:
                series = pd.Series(X[:, self.columns_[0]]).dropna()

            if len(series) < 50:
                return 7  # Default to weekly seasonality

            # Calculate autocorrelation
            max_lag = min(len(series) // 4, 100)
            autocorr = acf(series, nlags=max_lag, fft=True)

            # Find peaks in autocorrelation
            peaks = []
            for i in range(2, len(autocorr) - 1):
                if (
                    autocorr[i] > autocorr[i - 1]
                    and autocorr[i] > autocorr[i + 1]
                    and autocorr[i] > 0.1
                ):
                    peaks.append((i, autocorr[i]))

            if peaks:
                # Return the lag with highest significant autocorrelation
                best_period = max(peaks, key=lambda x: x[1])[0]
                return best_period
            else:
                return 7  # Default to weekly seasonality

        except ImportError:
            warnings.warn("statsmodels not available, using default period of 7")
            return 7
        except:
            return 7  # Fallback to weekly seasonality

    def transform(self, X):
        """Transform by decomposing into seasonal components."""
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            result = pd.DataFrame(X)

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            for col in self.columns_:
                if isinstance(X, pd.DataFrame):
                    series = result[col].dropna()
                else:
                    series = result.iloc[:, col].dropna()

                if len(series) >= 2 * self.period_:
                    try:
                        decomposition = seasonal_decompose(
                            series,
                            model=self.model,
                            period=self.period_,
                            extrapolate_trend="freq",
                        )

                        # Add decomposed components as new features
                        col_name = col if isinstance(col, str) else f"col_{col}"

                        # Align components with original index
                        trend = pd.Series(index=result.index, dtype=float)
                        seasonal = pd.Series(index=result.index, dtype=float)
                        resid = pd.Series(index=result.index, dtype=float)

                        trend.loc[decomposition.trend.index] = decomposition.trend
                        seasonal.loc[
                            decomposition.seasonal.index
                        ] = decomposition.seasonal
                        resid.loc[decomposition.resid.index] = decomposition.resid

                        result[f"{col_name}_trend"] = trend
                        result[f"{col_name}_seasonal"] = seasonal
                        result[f"{col_name}_residual"] = resid

                    except Exception as e:
                        warnings.warn(f"Could not decompose column {col}: {str(e)}")
                        continue
                else:
                    warnings.warn(
                        f"Column {col} has insufficient data for decomposition"
                    )

        except ImportError:
            warnings.warn("statsmodels not available, skipping seasonal decomposition")

        return result
