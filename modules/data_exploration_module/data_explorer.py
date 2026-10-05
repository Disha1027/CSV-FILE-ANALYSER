import pandas as pd
import numpy as np


class DataExplorer:
    """
    A class for exploring and filtering data.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the DataExplorer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def filter_rows(self, filters):
        """
        Filter rows based on specified criteria.
        
        Args:
            filters (dict): Dictionary of column: value pairs for filtering.
                Can also include operators like 'column__gt', 'column__lt', etc.
                
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        df_filtered = df.copy()
        
        for key, value in filters.items():
            if '__' in key:
                column, operator = key.rsplit('__', 1)
                if column not in df_filtered.columns:
                    continue
                
                if operator == 'gt':
                    df_filtered = df_filtered[df_filtered[column] > value]
                elif operator == 'lt':
                    df_filtered = df_filtered[df_filtered[column] < value]
                elif operator == 'gte':
                    df_filtered = df_filtered[df_filtered[column] >= value]
                elif operator == 'lte':
                    df_filtered = df_filtered[df_filtered[column] <= value]
                elif operator == 'eq':
                    df_filtered = df_filtered[df_filtered[column] == value]
                elif operator == 'ne':
                    df_filtered = df_filtered[df_filtered[column] != value]
                elif operator == 'contains':
                    df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(str(value), case=False, na=False)]
                elif operator == 'startswith':
                    df_filtered = df_filtered[df_filtered[column].astype(str).str.startswith(str(value), na=False)]
                elif operator == 'endswith':
                    df_filtered = df_filtered[df_filtered[column].astype(str).str.endswith(str(value), na=False)]
            else:
                if key in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[key] == value]
        
        return df_filtered
    
    def sort_dataset(self, columns, ascending=True):
        """
        Sort the dataset by specified columns.
        
        Args:
            columns (list or str): Column name(s) to sort by
            ascending (bool or list): Sort ascending or descending.
                Can be a single boolean or a list of booleans for each column.
                
        Returns:
            pandas.DataFrame: Sorted DataFrame
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        df_sorted = df.copy()
        
        if isinstance(columns, str):
            columns = [columns]
        
        valid_columns = [col for col in columns if col in df_sorted.columns]
        
        if len(valid_columns) == 0:
            return df_sorted
        
        df_sorted = df_sorted.sort_values(by=valid_columns, ascending=ascending)
        
        return df_sorted
    
    def search_rows(self, search_term, columns=None, case_sensitive=False):
        """
        Search for rows containing a specific term.
        
        Args:
            search_term (str): Term to search for
            columns (list, optional): List of columns to search in.
                If None, all columns will be searched.
            case_sensitive (bool): Whether search should be case sensitive
                
        Returns:
            pandas.DataFrame: DataFrame with matching rows
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        if columns is None:
            columns = df.columns
        
        mask = pd.Series(False, index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if case_sensitive:
                col_mask = df[col].astype(str).str.contains(search_term, na=False)
            else:
                col_mask = df[col].astype(str).str.contains(search_term, case=False, na=False)
            
            mask = mask | col_mask
        
        return df[mask].copy()
    
    def get_unique_values(self, column, limit=None):
        """
        Get unique values from a column.
        
        Args:
            column (str): Column name
            limit (int, optional): Maximum number of unique values to return.
                If None, all unique values will be returned.
                
        Returns:
            list: List of unique values
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return []
        
        unique_values = df[column].unique().tolist()
        
        if limit is not None and len(unique_values) > limit:
            unique_values = unique_values[:limit]
        
        return unique_values
    
    def get_value_counts(self, column, normalize=False, limit=None):
        """
        Get value counts for a column.
        
        Args:
            column (str): Column name
            normalize (bool): Whether to return proportions instead of counts
            limit (int, optional): Maximum number of values to return.
                
        Returns:
            pandas.Series: Series containing value counts
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        counts = df[column].value_counts(normalize=normalize)
        
        if limit is not None and len(counts) > limit:
            counts = counts.head(limit)
        
        return counts
    
    def filter_by_range(self, column, min_value=None, max_value=None):
        """
        Filter rows by numeric range.
        
        Args:
            column (str): Column name
            min_value (float, optional): Minimum value (inclusive)
            max_value (float, optional): Maximum value (inclusive)
                
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        df_filtered = df.copy()
        
        if min_value is not None:
            df_filtered = df_filtered[df_filtered[column] >= min_value]
        
        if max_value is not None:
            df_filtered = df_filtered[df_filtered[column] <= max_value]
        
        return df_filtered
    
    def filter_by_date_range(self, column, start_date=None, end_date=None):
        """
        Filter rows by date range.
        
        Args:
            column (str): Date column name
            start_date (str or datetime, optional): Start date
            end_date (str or datetime, optional): End date
                
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        df_filtered = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[column]):
            df_filtered[column] = pd.to_datetime(df_filtered[column])
        
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered[column] >= start_date]
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            df_filtered = df_filtered[df_filtered[column] <= end_date]
        
        return df_filtered
    
    def get_sample(self, n=5, random_state=None):
        """
        Get a random sample of rows.
        
        Args:
            n (int): Number of rows to sample
            random_state (int, optional): Random seed for reproducibility
                
        Returns:
            pandas.DataFrame: Sampled DataFrame
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        if len(df) <= n:
            return df.copy()
        
        return df.sample(n=n, random_state=random_state)
    
    def get_head(self, n=5):
        """
        Get the first n rows.
        
        Args:
            n (int): Number of rows to return
                
        Returns:
            pandas.DataFrame: First n rows
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        return df.head(n).copy()
    
    def get_tail(self, n=5):
        """
        Get the last n rows.
        
        Args:
            n (int): Number of rows to return
                
        Returns:
            pandas.DataFrame: Last n rows
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        return df.tail(n).copy()
    
    def filter_by_multiple_conditions(self, conditions, logic='and'):
        """
        Filter rows by multiple conditions.
        
        Args:
            conditions (list): List of filter dictionaries
            logic (str): Logic to combine conditions ('and' or 'or')
                
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        df_filtered = df.copy()
        masks = []
        
        for condition in conditions:
            temp_df = df.copy()
            filtered = self.filter_rows(condition)
            mask = df.index.isin(filtered.index)
            masks.append(mask)
        
        if logic == 'and':
            final_mask = pd.Series(True, index=df.index)
            for mask in masks:
                final_mask = final_mask & mask
        else:  # logic == 'or'
            final_mask = pd.Series(False, index=df.index)
            for mask in masks:
                final_mask = final_mask | mask
        
        return df[final_mask].copy()
