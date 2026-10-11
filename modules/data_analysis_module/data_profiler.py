import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import streamlit as st


class DataProfiler:
    """
    A class for comprehensive dataset profiling and automatic analysis.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the DataProfiler class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def get_dataset_profile(self):
        """
        Generate a comprehensive dataset-level profile.
        
        Returns:
            dict: Dictionary containing dataset-level metrics
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        # Basic dimensions
        rows, cols = df.shape
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        memory_str = self._format_memory(memory_usage)
        
        # Column type counts
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        bool_cols = df.select_dtypes(include=['bool']).columns
        
        # Missing values
        total_cells = rows * cols
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Duplicate rows
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / rows) * 100 if rows > 0 else 0
        
        # Unique columns (columns with all unique values)
        unique_cols = [col for col in df.columns if df[col].nunique() == rows]
        
        # Constant columns (columns with only one unique value)
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        
        # Rows with missing values
        rows_with_missing = df.isnull().any(axis=1).sum()
        complete_rows = rows - rows_with_missing
        
        # Mixed type detection
        mixed_type_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                types = df[col].dropna().apply(type).nunique()
                if types > 1:
                    mixed_type_cols.append(col)
        
        # Identifier-like columns (high cardinality, likely IDs)
        identifier_cols = []
        for col in df.columns:
            if df[col].nunique() > rows * 0.9 and df[col].nunique() > 10:
                identifier_cols.append(col)
        
        return {
            'rows': rows,
            'columns': cols,
            'memory_usage': memory_str,
            'memory_bytes': memory_usage,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols),
            'datetime_count': len(datetime_cols),
            'boolean_count': len(bool_cols),
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': missing_pct,
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_pct,
            'unique_columns': unique_cols,
            'constant_columns': constant_cols,
            'mixed_type_columns': mixed_type_cols,
            'identifier_columns': identifier_cols,
            'rows_with_missing': rows_with_missing,
            'complete_rows': complete_rows,
            'numeric_columns': numeric_cols.tolist(),
            'categorical_columns': categorical_cols.tolist(),
            'datetime_columns': datetime_cols.tolist(),
            'boolean_columns': bool_cols.tolist()
        }
    
    def get_column_profile(self, column):
        """
        Generate a detailed profile for a specific column.
        
        Args:
            column (str): Column name to profile
            
        Returns:
            dict: Dictionary containing column-level metrics
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        col_data = df[column]
        dtype = str(col_data.dtype)
        
        profile = {
            'column': column,
            'dtype': dtype,
            'count': col_data.count(),
            'missing_count': col_data.isnull().sum(),
            'missing_percentage': (col_data.isnull().sum() / len(df)) * 100,
            'unique_count': col_data.nunique(),
            'unique_percentage': (col_data.nunique() / len(df)) * 100
        }
        
        # Type-specific metrics
        if pd.api.types.is_numeric_dtype(col_data):
            profile.update(self._get_numeric_profile(col_data))
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            profile.update(self._get_datetime_profile(col_data))
        elif pd.api.types.is_bool_dtype(col_data):
            profile.update(self._get_boolean_profile(col_data))
        else:
            # Treat as categorical/text
            profile.update(self._get_categorical_profile(col_data))
        
        return profile
    
    def _get_numeric_profile(self, series):
        """Get profile metrics for numeric columns."""
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            return {'type': 'numeric', 'error': 'No valid data'}
        
        profile = {
            'type': 'numeric',
            'mean': valid_data.mean(),
            'median': valid_data.median(),
            'mode': valid_data.mode()[0] if len(valid_data.mode()) > 0 else None,
            'std': valid_data.std(),
            'variance': valid_data.var(),
            'min': valid_data.min(),
            'max': valid_data.max(),
            'range': valid_data.max() - valid_data.min(),
            'q1': valid_data.quantile(0.25),
            'q2': valid_data.quantile(0.5),
            'q3': valid_data.quantile(0.75),
            'iqr': valid_data.quantile(0.75) - valid_data.quantile(0.25),
            'skewness': valid_data.skew(),
            'kurtosis': valid_data.kurtosis(),
            'zero_count': (valid_data == 0).sum(),
            'negative_count': (valid_data < 0).sum(),
            'percentiles': {
                '1%': valid_data.quantile(0.01),
                '5%': valid_data.quantile(0.05),
                '10%': valid_data.quantile(0.10),
                '90%': valid_data.quantile(0.90),
                '95%': valid_data.quantile(0.95),
                '99%': valid_data.quantile(0.99)
            }
        }
        
        return profile
    
    def _get_categorical_profile(self, series):
        """Get profile metrics for categorical/text columns."""
        valid_data = series.dropna().astype(str)
        
        if len(valid_data) == 0:
            return {'type': 'categorical', 'error': 'No valid data'}
        
        value_counts = valid_data.value_counts()
        
        # Check for empty strings and whitespace
        empty_strings = (valid_data == '').sum()
        whitespace_only = valid_data.str.strip().eq('').sum()
        
        # Check for inconsistent capitalization
        unique_normalized = valid_data.str.lower().nunique()
        original_unique = valid_data.nunique()
        potential_duplicates = original_unique - unique_normalized
        
        # Most and least frequent
        most_frequent = value_counts.index[0] if len(value_counts) > 0 else None
        least_frequent = value_counts.index[-1] if len(value_counts) > 0 else None
        
        # String length statistics
        str_lengths = valid_data.str.len()
        
        # Cardinality ratio
        cardinality_ratio = valid_data.nunique() / len(valid_data) if len(valid_data) > 0 else 0
        
        # Rare categories (appearing less than 1% of the time)
        threshold = len(valid_data) * 0.01
        rare_categories = value_counts[value_counts < threshold].index.tolist()
        
        profile = {
            'type': 'categorical',
            'most_frequent': most_frequent,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': least_frequent,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'empty_strings': empty_strings,
            'whitespace_only': whitespace_only,
            'potential_capitalization_issues': potential_duplicates,
            'min_string_length': str_lengths.min() if len(str_lengths) > 0 else 0,
            'max_string_length': str_lengths.max() if len(str_lengths) > 0 else 0,
            'mean_string_length': str_lengths.mean() if len(str_lengths) > 0 else 0,
            'median_string_length': str_lengths.median() if len(str_lengths) > 0 else 0,
            'cardinality_ratio': cardinality_ratio,
            'rare_categories_count': len(rare_categories),
            'rare_categories': rare_categories[:10],  # Limit to top 10
            'category_frequencies': value_counts.head(20).to_dict()
        }
        
        return profile
    
    def _get_datetime_profile(self, series):
        """Get profile metrics for datetime columns."""
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            return {'type': 'datetime', 'error': 'No valid data'}
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(valid_data):
            try:
                valid_data = pd.to_datetime(valid_data, errors='coerce')
                valid_data = valid_data.dropna()
            except:
                return {'type': 'datetime', 'error': 'Could not convert to datetime'}
        
        if len(valid_data) == 0:
            return {'type': 'datetime', 'error': 'No valid datetime data'}
        
        # Time range
        min_date = valid_data.min()
        max_date = valid_data.max()
        time_range = max_date - min_date
        
        # Check for duplicates
        duplicate_timestamps = valid_data.duplicated().sum()
        
        # Try to infer frequency
        try:
            inferred_freq = pd.infer_freq(valid_data.sort_values())
        except:
            inferred_freq = None
        
        profile = {
            'type': 'datetime',
            'min_date': min_date,
            'max_date': max_date,
            'time_range_days': time_range.days,
            'duplicate_timestamps': duplicate_timestamps,
            'inferred_frequency': inferred_freq,
            'valid_count': len(valid_data),
            'invalid_count': len(series) - len(valid_data)
        }
        
        return profile
    
    def _get_boolean_profile(self, series):
        """Get profile metrics for boolean columns."""
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            return {'type': 'boolean', 'error': 'No valid data'}
        
        value_counts = valid_data.value_counts()
        
        profile = {
            'type': 'boolean',
            'true_count': value_counts.get(True, 0),
            'false_count': value_counts.get(False, 0),
            'true_percentage': (value_counts.get(True, 0) / len(valid_data)) * 100 if len(valid_data) > 0 else 0,
            'false_percentage': (value_counts.get(False, 0) / len(valid_data)) * 100 if len(valid_data) > 0 else 0
        }
        
        return profile
    
    def _format_memory(self, bytes_value):
        """Format memory usage in human-readable format."""
        if bytes_value < 1024:
            return f"{bytes_value} bytes"
        elif bytes_value < 1024**2:
            return f"{bytes_value/1024:.2f} KB"
        elif bytes_value < 1024**3:
            return f"{bytes_value/(1024**2):.2f} MB"
        else:
            return f"{bytes_value/(1024**3):.2f} GB"
    
    def get_all_columns_profile(self):
        """
        Generate profiles for all columns in the dataset.
        
        Returns:
            list: List of column profile dictionaries
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        profiles = []
        for col in df.columns:
            profile = self.get_column_profile(col)
            if profile:
                profiles.append(profile)
        
        return profiles
