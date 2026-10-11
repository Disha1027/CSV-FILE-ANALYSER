import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest
import streamlit as st


class AdvancedStatisticalAnalyzer:
    """
    A class for advanced statistical analysis including univariate, bivariate, and multivariate analysis.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the AdvancedStatisticalAnalyzer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def get_univariate_analysis(self, column):
        """
        Perform comprehensive univariate analysis on a column.
        
        Args:
            column (str): Column name to analyze
            
        Returns:
            dict: Dictionary containing univariate analysis results
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        col_data = df[column].dropna()
        
        if len(col_data) == 0:
            return None
        
        analysis = {
            'column': column,
            'dtype': str(df[column].dtype),
            'count': len(col_data),
            'missing_count': df[column].isnull().sum()
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            analysis.update(self._numeric_univariate_analysis(col_data))
        else:
            analysis.update(self._categorical_univariate_analysis(col_data))
        
        return analysis
    
    def _numeric_univariate_analysis(self, series):
        """Perform univariate analysis on numeric data."""
        analysis = {
            'type': 'numeric',
            'mean': series.mean(),
            'median': series.median(),
            'mode': series.mode()[0] if len(series.mode()) > 0 else None,
            'std': series.std(),
            'variance': series.var(),
            'min': series.min(),
            'max': series.max(),
            'range': series.max() - series.min(),
            'q1': series.quantile(0.25),
            'q2': series.quantile(0.5),
            'q3': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'coefficient_of_variation': (series.std() / series.mean()) if series.mean() != 0 else np.nan
        }
        
        # Normality tests
        if len(series) >= 8:
            # Shapiro-Wilk test (for small samples)
            if len(series) <= 5000:
                shapiro_stat, shapiro_p = shapiro(series)
                analysis['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            
            # D'Agostino's normality test (for larger samples)
            if len(series) >= 20:
                dagostino_stat, dagostino_p = normaltest(series)
                analysis['dagostino_normality'] = {
                    'statistic': dagostino_stat,
                    'p_value': dagostino_p,
                    'is_normal': dagostino_p > 0.05
                }
        
        # Outlier detection using multiple methods
        analysis['outliers'] = {
            'iqr': self._detect_outliers_iqr(series),
            'zscore': self._detect_outliers_zscore(series),
            'modified_zscore': self._detect_outliers_modified_zscore(series)
        }
        
        return analysis
    
    def _categorical_univariate_analysis(self, series):
        """Perform univariate analysis on categorical data."""
        value_counts = series.value_counts()
        
        analysis = {
            'type': 'categorical',
            'unique_count': series.nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'most_frequent_pct': (value_counts.iloc[0] / len(series)) * 100 if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'value_counts': value_counts.to_dict()
        }
        
        # Rare categories (less than 5% frequency)
        threshold = len(series) * 0.05
        rare_categories = value_counts[value_counts < threshold]
        analysis['rare_categories'] = {
            'count': len(rare_categories),
            'categories': rare_categories.index.tolist()[:10]
        }
        
        # Entropy (measure of diversity)
        from scipy.stats import entropy
        probabilities = value_counts / len(series)
        analysis['entropy'] = entropy(probabilities)
        
        return analysis
    
    def _detect_outliers_iqr(self, series, multiplier=1.5):
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(series)) * 100,
            'indices': outliers.index.tolist(),
            'values': outliers.tolist(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def _detect_outliers_zscore(self, series, threshold=3):
        """Detect outliers using Z-score method."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return {
                'count': 0,
                'percentage': 0,
                'indices': [],
                'values': [],
                'threshold': threshold
            }
        
        z_scores = np.abs((series - mean) / std)
        outliers = series[z_scores > threshold]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(series)) * 100,
            'indices': outliers.index.tolist(),
            'values': outliers.tolist(),
            'threshold': threshold
        }
    
    def _detect_outliers_modified_zscore(self, series, threshold=3.5):
        """Detect outliers using modified Z-score (median absolute deviation)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return {
                'count': 0,
                'percentage': 0,
                'indices': [],
                'values': [],
                'threshold': threshold
            }
        
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = series[np.abs(modified_z_scores) > threshold]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(series)) * 100,
            'indices': outliers.index.tolist(),
            'values': outliers.tolist(),
            'threshold': threshold
        }
    
    def get_bivariate_analysis(self, col1, col2):
        """
        Perform bivariate analysis between two columns.
        
        Args:
            col1 (str): First column name
            col2 (str): Second column name
            
        Returns:
            dict: Dictionary containing bivariate analysis results
        """
        df = self.data_loader.get_data()
        
        if df is None or col1 not in df.columns or col2 not in df.columns:
            return None
        
        # Determine column types
        col1_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col1]) else 'categorical'
        col2_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col2]) else 'categorical'
        
        analysis = {
            'col1': col1,
            'col2': col2,
            'col1_type': col1_type,
            'col2_type': col2_type,
            'pair_type': f"{col1_type}_vs_{col2_type}"
        }
        
        # Remove rows with missing values in either column
        clean_df = df[[col1, col2]].dropna()
        
        if len(clean_df) == 0:
            analysis['error'] = 'No valid data after removing missing values'
            return analysis
        
        if col1_type == 'numeric' and col2_type == 'numeric':
            analysis.update(self._numeric_vs_numeric_analysis(clean_df[col1], clean_df[col2]))
        elif col1_type == 'numeric' and col2_type == 'categorical':
            analysis.update(self._numeric_vs_categorical_analysis(clean_df[col1], clean_df[col2]))
        elif col1_type == 'categorical' and col2_type == 'numeric':
            analysis.update(self._numeric_vs_categorical_analysis(clean_df[col2], clean_df[col1]))
        else:
            analysis.update(self._categorical_vs_categorical_analysis(clean_df[col1], clean_df[col2]))
        
        return analysis
    
    def _numeric_vs_numeric_analysis(self, series1, series2):
        """Analyze relationship between two numeric variables."""
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(series1, series2)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(series1, series2)
        
        # Kendall correlation
        kendall_r, kendall_p = stats.kendalltau(series1, series2)
        
        return {
            'pearson_correlation': {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < 0.05
            },
            'spearman_correlation': {
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < 0.05
            },
            'kendall_correlation': {
                'correlation': kendall_r,
                'p_value': kendall_p,
                'significant': kendall_p < 0.05
            }
        }
    
    def _numeric_vs_categorical_analysis(self, numeric_series, cat_series):
        """Analyze relationship between numeric and categorical variables."""
        # Group statistics
        grouped = numeric_series.groupby(cat_series).agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        
        # ANOVA test (if more than 2 groups)
        unique_groups = cat_series.unique()
        
        if len(unique_groups) >= 2:
            # Prepare data for ANOVA
            groups = [numeric_series[cat_series == group].values for group in unique_groups]
            
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                anova_result = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                anova_result = {'error': 'Could not perform ANOVA test'}
        else:
            anova_result = None
        
        # Kruskal-Wallis test (non-parametric alternative)
        if len(unique_groups) >= 2:
            try:
                kw_stat, kw_p = stats.kruskal(*[numeric_series[cat_series == group].values for group in unique_groups])
                kruskal_result = {
                    'statistic': kw_stat,
                    'p_value': kw_p,
                    'significant': kw_p < 0.05
                }
            except:
                kruskal_result = {'error': 'Could not perform Kruskal-Wallis test'}
        else:
            kruskal_result = None
        
        return {
            'group_statistics': grouped.to_dict(),
            'anova_test': anova_result,
            'kruskal_wallis_test': kruskal_result
        }
    
    def _categorical_vs_categorical_analysis(self, series1, series2):
        """Analyze relationship between two categorical variables."""
        # Contingency table
        contingency_table = pd.crosstab(series1, series2)
        
        # Chi-square test of independence
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            chi_square_result = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
        except:
            chi_square_result = {'error': 'Could not perform Chi-square test'}
        
        # Cramér's V (effect size)
        try:
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if 'chi2_statistic' in chi_square_result else np.nan
        except:
            cramers_v = np.nan
        
        return {
            'contingency_table': contingency_table.to_dict(),
            'chi_square_test': chi_square_result,
            'cramers_v': cramers_v
        }
    
    def get_correlation_matrix(self, method='pearson', columns=None):
        """
        Calculate correlation matrix with specified method.
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            columns (list, optional): Columns to include
            
        Returns:
            pandas.DataFrame: Correlation matrix
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        if columns is None:
            columns = self.data_loader.get_numeric_columns()
        
        if len(columns) < 2:
            return None
        
        return df[columns].corr(method=method)
    
    def get_multivariate_outliers(self, columns=None, method='mahalanobis'):
        """
        Detect multivariate outliers.
        
        Args:
            columns (list, optional): Columns to analyze
            method (str): Method to use ('mahalanobis')
            
        Returns:
            dict: Dictionary containing outlier detection results
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        if columns is None:
            columns = self.data_loader.get_numeric_columns()
        
        if len(columns) < 2:
            return None
        
        # Clean data
        clean_df = df[columns].dropna()
        
        if len(clean_df) == 0:
            return None
        
        if method == 'mahalanobis':
            return self._mahalanobis_outliers(clean_df)
        
        return None
    
    def _mahalanobis_outliers(self, df):
        """Detect outliers using Mahalanobis distance."""
        try:
            from scipy.spatial.distance import mahalanobis
            from scipy.linalg import pinv
            
            # Calculate covariance matrix
            cov_matrix = df.cov()
            
            # Handle singular covariance matrix
            if np.linalg.det(cov_matrix) == 0:
                # Use pseudo-inverse
                inv_cov = pinv(cov_matrix)
            else:
                inv_cov = np.linalg.inv(cov_matrix)
            
            # Calculate Mahalanobis distance for each point
            mean = df.mean()
            mahalanobis_dist = df.apply(
                lambda row: mahalanobis(row, mean, inv_cov),
                axis=1
            )
            
            # Use chi-square distribution for threshold (p < 0.001)
            threshold = stats.chi2.ppf(0.999, df=len(df.columns))
            
            outliers = mahalanobis_dist > threshold
            
            return {
                'method': 'mahalanobis',
                'distances': mahalanobis_dist.tolist(),
                'threshold': threshold,
                'outlier_indices': df[outliers].index.tolist(),
                'outlier_count': outliers.sum(),
                'outlier_percentage': (outliers.sum() / len(df)) * 100
            }
        except Exception as e:
            return {'error': f'Could not calculate Mahalanobis distance: {str(e)}'}
