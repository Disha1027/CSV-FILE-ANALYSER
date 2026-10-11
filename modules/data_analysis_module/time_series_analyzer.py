import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class TimeSeriesAnalyzer:
    """
    A class for time-series exploratory data analysis.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the TimeSeriesAnalyzer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def get_datetime_columns(self):
        """
        Get columns that can be converted to datetime.
        
        Returns:
            list: List of column names that are or can be datetime
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None:
            return []
        
        datetime_cols = []
        
        # Already datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            datetime_cols.append(col)
        
        # Try to convert object columns to datetime
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col].head(100), errors='raise')
                datetime_cols.append(col)
            except:
                continue
        
        return datetime_cols
    
    def analyze_time_series(self, date_column, value_column):
        """
        Perform comprehensive time-series analysis.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            
        Returns:
            dict: Dictionary containing time-series analysis results
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                return {'error': 'Could not convert date column to datetime'}
        
        # Create a copy with date as index
        ts_df = df[[date_column, value_column]].copy()
        ts_df = ts_df.dropna()
        ts_df = ts_df.set_index(date_column)
        ts_df = ts_df.sort_index()
        
        if len(ts_df) == 0:
            return {'error': 'No valid data after cleaning'}
        
        analysis = {
            'date_column': date_column,
            'value_column': value_column,
            'start_date': ts_df.index.min(),
            'end_date': ts_df.index.max(),
            'duration_days': (ts_df.index.max() - ts_df.index.min()).days,
            'total_observations': len(ts_df),
            'duplicate_timestamps': ts_df.index.duplicated().sum(),
            'missing_timestamps': self._detect_missing_timestamps(ts_df)
        }
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': ts_df[value_column].mean(),
            'median': ts_df[value_column].median(),
            'std': ts_df[value_column].std(),
            'min': ts_df[value_column].min(),
            'max': ts_df[value_column].max(),
            'trend': self._calculate_trend(ts_df[value_column])
        }
        
        # Try to infer frequency
        try:
            analysis['inferred_frequency'] = pd.infer_freq(ts_df.index)
        except:
            analysis['inferred_frequency'] = None
        
        return analysis
    
    def _detect_missing_timestamps(self, ts_df):
        """Detect gaps in time series."""
        if len(ts_df) < 2:
            return {'gaps': []}
        
        # Calculate time differences
        time_diffs = ts_df.index.to_series().diff()
        
        # Find irregular gaps (more than 2x the median difference)
        median_diff = time_diffs.median()
        if median_diff is not None and median_diff > pd.Timedelta(0):
            threshold = median_diff * 2
            gaps = time_diffs[time_diffs > threshold]
            
            return {
                'count': len(gaps),
                'gaps': gaps.tolist()[:10],  # Limit to first 10
                'median_interval': median_diff
            }
        
        return {'gaps': []}
    
    def _calculate_trend(self, series):
        """Calculate linear trend using simple linear regression."""
        if len(series) < 2:
            return None
        
        x = np.arange(len(series))
        y = series.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
        }
    
    def create_time_series_plot(self, date_column, value_column):
        """
        Create a basic time-series line plot.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            
        Returns:
            matplotlib.figure.Figure: The time-series plot figure
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                return None
        
        # Clean data
        ts_df = df[[date_column, value_column]].dropna()
        ts_df = ts_df.sort_values(date_column)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(ts_df[date_column], ts_df[value_column], linewidth=1)
        ax.set_xlabel(date_column)
        ax.set_ylabel(value_column)
        ax.set_title(f'Time Series: {value_column} over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_rolling_statistics_plot(self, date_column, value_column, window=7):
        """
        Create a plot with rolling mean and standard deviation.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            window (int): Rolling window size
            
        Returns:
            matplotlib.figure.Figure: The rolling statistics plot figure
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                return None
        
        # Clean data
        ts_df = df[[date_column, value_column]].dropna()
        ts_df = ts_df.sort_values(date_column)
        ts_df = ts_df.set_index(date_column)
        
        # Calculate rolling statistics
        rolling_mean = ts_df[value_column].rolling(window=window).mean()
        rolling_std = ts_df[value_column].rolling(window=window).std()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(ts_df.index, ts_df[value_column], label='Original', alpha=0.5)
        ax.plot(rolling_mean.index, rolling_mean, label=f'Rolling Mean ({window})', linewidth=2)
        ax.fill_between(rolling_std.index, 
                       rolling_mean - 2*rolling_std, 
                       rolling_mean + 2*rolling_std, 
                       alpha=0.2, label=f'±2 Std Dev ({window})')
        ax.set_xlabel(date_column)
        ax.set_ylabel(value_column)
        ax.set_title(f'Rolling Statistics - {value_column}')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_seasonal_decomposition_plot(self, date_column, value_column, model='additive', period=None):
        """
        Create seasonal decomposition plot.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            model (str): 'additive' or 'multiplicative'
            period (int, optional): Period for seasonal decomposition
            
        Returns:
            matplotlib.figure.Figure: The seasonal decomposition figure
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                return None
        
        # Clean data
        ts_df = df[[date_column, value_column]].dropna()
        ts_df = ts_df.sort_values(date_column)
        ts_df = ts_df.set_index(date_column)
        
        # Ensure sufficient data
        if len(ts_df) < 2 * (period if period else 2):
            return None
        
        try:
            # Perform seasonal decomposition
            result = seasonal_decompose(ts_df[value_column], model=model, period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 10))
            
            result.observed.plot(ax=axes[0])
            axes[0].set_title('Observed')
            
            result.trend.plot(ax=axes[1])
            axes[1].set_title('Trend')
            
            result.seasonal.plot(ax=axes[2])
            axes[2].set_title('Seasonal')
            
            result.resid.plot(ax=axes[3])
            axes[3].set_title('Residual')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error performing seasonal decomposition: {e}")
            return None
    
    def create_acf_pacf_plots(self, date_column, value_column, lags=40):
        """
        Create ACF and PACF plots.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            lags (int): Number of lags to display
            
        Returns:
            matplotlib.figure.Figure: The ACF/PACF figure
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Clean data
        ts_df = df[[date_column, value_column]].dropna()
        ts_df = ts_df.sort_values(date_column)
        
        # Use the value column directly
        series = ts_df[value_column].values
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # ACF plot
        plot_acf(series, lags=min(lags, len(series)-1), ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(series, lags=min(lags, len(series)-1), ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    
    def create_lag_plot(self, value_column, lag=1):
        """
        Create a lag plot to visualize autocorrelation.
        
        Args:
            value_column (str): Numeric value column
            lag (int): Lag value
            
        Returns:
            matplotlib.figure.Figure: The lag plot figure
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or value_column not in df.columns:
            return None
        
        series = df[value_column].dropna()
        
        if len(series) <= lag:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        pd.plotting.lag_plot(series, lag=lag, ax=ax)
        ax.set_title(f'Lag Plot (lag={lag}) - {value_column}')
        
        return fig
    
    def resample_time_series(self, date_column, value_column, freq='D', agg_func='mean'):
        """
        Resample time series to a different frequency.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            freq (str): Resampling frequency ('D', 'W', 'M', 'Q', 'Y', etc.)
            agg_func (str): Aggregation function ('mean', 'sum', 'min', 'max', etc.)
            
        Returns:
            pandas.DataFrame: Resampled time series
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                return None
        
        # Clean data
        ts_df = df[[date_column, value_column]].dropna()
        ts_df = ts_df.sort_values(date_column)
        ts_df = ts_df.set_index(date_column)
        
        # Resample
        agg_map = {
            'mean': 'mean',
            'sum': 'sum',
            'min': 'min',
            'max': 'max',
            'median': 'median',
            'std': 'std',
            'count': 'count'
        }
        
        func = agg_map.get(agg_func, 'mean')
        
        try:
            resampled = ts_df[value_column].resample(freq).agg(func)
            return resampled.dropna()
        except Exception as e:
            st.error(f"Error resampling time series: {e}")
            return None
    
    def create_percentage_change_plot(self, date_column, value_column):
        """
        Create a plot showing percentage changes.
        
        Args:
            date_column (str): Datetime column
            value_column (str): Numeric value column
            
        Returns:
            matplotlib.figure.Figure: The percentage change plot figure
        """
        source_df = self.data_loader.get_data()
        df = source_df.copy() if source_df is not None else None
        
        if df is None or date_column not in df.columns or value_column not in df.columns:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except:
                return None
        
        # Clean data
        ts_df = df[[date_column, value_column]].dropna()
        ts_df = ts_df.sort_values(date_column)
        ts_df = ts_df.set_index(date_column)
        
        # Calculate percentage change
        pct_change = ts_df[value_column].pct_change() * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pct_change.index, pct_change, linewidth=1)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel(date_column)
        ax.set_ylabel('Percentage Change (%)')
        ax.set_title(f'Percentage Change - {value_column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
