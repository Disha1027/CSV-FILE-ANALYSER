import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataCleaner:
    """
    A class for cleaning and preprocessing data.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the DataCleaner class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
        self.cleaning_flag = False
        self.normalization_directive = None
    
    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values.
                'mean': Fill with mean (for numeric columns)
                'median': Fill with median (for numeric columns)
                'mode': Fill with mode (for all columns)
                'drop': Drop rows with missing values
                'forward': Forward fill
                'backward': Backward fill
            columns (list, optional): List of column names to process.
                If None, all columns will be processed.
                
        Returns:
            pandas.DataFrame: DataFrame with missing values handled
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df_cleaned.columns
        
        for col in columns:
            if col not in df_cleaned.columns:
                continue
            
            if strategy == 'mean':
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif strategy == 'median':
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strategy == 'mode':
                mode_value = df_cleaned[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
            elif strategy == 'drop':
                df_cleaned.dropna(subset=[col], inplace=True)
            elif strategy == 'forward':
                df_cleaned[col] = df_cleaned[col].ffill()
            elif strategy == 'backward':
                df_cleaned[col] = df_cleaned[col].bfill()
        
        self.cleaning_flag = True
        return df_cleaned
    
    def normalize_scale_data(self, columns=None, method='standard'):
        """
        Normalize or scale numeric columns.
        
        Args:
            columns (list, optional): List of column names to normalize.
                If None, all numeric columns will be used.
            method (str): Normalization method.
                'standard': StandardScaler (z-score normalization)
                'minmax': MinMaxScaler (scale to [0, 1])
                
        Returns:
            tuple: (normalized DataFrame, scaler object)
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None, None
        
        df_normalized = df.copy()
        
        if columns is None:
            columns = df_normalized.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) == 0:
            return df_normalized, None
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        self.normalization_directive = method
        
        return df_normalized, scaler
    
    def remove_duplicate_rows(self, subset=None, keep='first'):
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset (list, optional): List of column names to consider for identifying duplicates.
                If None, all columns will be considered.
            keep (str): Which duplicate to keep.
                'first': Keep first occurrence
                'last': Keep last occurrence
                False: Drop all duplicates
                
        Returns:
            pandas.DataFrame: DataFrame with duplicates removed
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        df_cleaned = df.copy()
        initial_count = len(df_cleaned)
        
        df_cleaned.drop_duplicates(subset=subset, keep=keep, inplace=True)
        
        removed_count = initial_count - len(df_cleaned)
        
        if removed_count > 0:
            self.cleaning_flag = True
        
        return df_cleaned
    
    def get_cleaning_summary(self):
        """
        Get a summary of cleaning operations performed.
        
        Returns:
            dict: Dictionary containing cleaning summary
        """
        return {
            'cleaning_flag': self.cleaning_flag,
            'normalization_directive': self.normalization_directive
        }
    
    def detect_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Detect outliers in numeric columns.
        
        Args:
            columns (list, optional): List of column names to analyze.
                If None, all numeric columns will be used.
            method (str): Method to use for outlier detection.
                'iqr': Interquartile Range method
                'zscore': Z-score method
            threshold (float): Threshold for outlier detection.
                
        Returns:
            dict: Dictionary containing outlier information for each column
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        outlier_info = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    outliers = pd.DataFrame()
                else:
                    z_scores = (df[col] - mean) / std
                    outliers = df[abs(z_scores) > threshold]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outlier_info
    
    def impute_data(self, columns=None, strategy='mean'):
        """
        Impute missing values with specified strategy.
        
        Args:
            columns (list, optional): List of column names to impute.
                If None, all columns with missing values will be used.
            strategy (str): Imputation strategy.
                'mean': Mean imputation
                'median': Median imputation
                'mode': Mode imputation
                'constant': Constant value (0)
                
        Returns:
            pandas.DataFrame: DataFrame with imputed values
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        df_imputed = df.copy()
        
        if columns is None:
            columns = df_imputed.columns[df_imputed.isnull().any()].tolist()
        
        for col in columns:
            if col not in df_imputed.columns:
                continue
            
            if strategy == 'mean' and df_imputed[col].dtype in ['int64', 'float64']:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            elif strategy == 'median' and df_imputed[col].dtype in ['int64', 'float64']:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            elif strategy == 'mode':
                mode_value = df_imputed[col].mode()
                if len(mode_value) > 0:
                    df_imputed[col] = df_imputed[col].fillna(mode_value[0])
            elif strategy == 'constant':
                df_imputed[col] = df_imputed[col].fillna(0)
        
        self.cleaning_flag = True
        return df_imputed
