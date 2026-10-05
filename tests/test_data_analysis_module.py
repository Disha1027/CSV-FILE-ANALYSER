import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_analysis_module.statistical_analyzer import StatisticalAnalyzer


class MockDataLoader:
    """Mock DataLoader for testing"""
    def __init__(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
            'Category': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
        })
    
    def get_data(self):
        return self.data
    
    def get_numeric_columns(self):
        return ['A', 'B', 'C']


class TestStatisticalAnalyzer(unittest.TestCase):
    """Unit tests for StatisticalAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_loader = MockDataLoader()
        self.stat_analyzer = StatisticalAnalyzer(self.mock_loader)
    
    def test_initialization(self):
        """Test StatisticalAnalyzer initialization"""
        self.assertIsNotNone(self.stat_analyzer)
        self.assertEqual(self.stat_analyzer.data_loader, self.mock_loader)
    
    def test_get_descriptive_statistics(self):
        """Test descriptive statistics calculation"""
        stats = self.stat_analyzer.get_descriptive_statistics()
        
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIn('count', stats.columns)
        self.assertIn('mean', stats.columns)
        self.assertIn('std', stats.columns)
        self.assertEqual(len(stats), 3)  # 3 numeric columns
    
    def test_get_descriptive_statistics_specific_columns(self):
        """Test descriptive statistics for specific columns"""
        stats = self.stat_analyzer.get_descriptive_statistics(columns=['A', 'B'])
        
        self.assertIsNotNone(stats)
        self.assertEqual(len(stats), 2)
    
    def test_get_group_by_statistics(self):
        """Test group by statistics"""
        result = self.stat_analyzer.get_group_by_statistics(
            group_column='Category',
            agg_column='A',
            agg_func='Mean'
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Category', result.columns)
        self.assertIn('A', result.columns)
        self.assertEqual(len(result), 2)  # 2 categories
    
    def test_get_group_by_statistics_median(self):
        """Test group by with median aggregation"""
        result = self.stat_analyzer.get_group_by_statistics(
            group_column='Category',
            agg_column='B',
            agg_func='Median'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
    
    def test_get_group_by_statistics_sum(self):
        """Test group by with sum aggregation"""
        result = self.stat_analyzer.get_group_by_statistics(
            group_column='Category',
            agg_column='C',
            agg_func='Sum'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
    
    def test_get_correlation_matrix(self):
        """Test correlation matrix calculation"""
        corr_matrix = self.stat_analyzer.get_correlation_matrix()
        
        self.assertIsNotNone(corr_matrix)
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape[0], 3)
        self.assertEqual(corr_matrix.shape[1], 3)
    
    def test_get_correlation_matrix_specific_columns(self):
        """Test correlation matrix for specific columns"""
        corr_matrix = self.stat_analyzer.get_correlation_matrix(columns=['A', 'B'])
        
        self.assertIsNotNone(corr_matrix)
        self.assertEqual(corr_matrix.shape[0], 2)
        self.assertEqual(corr_matrix.shape[1], 2)
    
    def test_get_correlation_matrix_insufficient_columns(self):
        """Test correlation matrix with insufficient columns"""
        corr_matrix = self.stat_analyzer.get_correlation_matrix(columns=['A'])
        self.assertIsNone(corr_matrix)
    
    def test_get_summary_by_category(self):
        """Test summary statistics by category"""
        result = self.stat_analyzer.get_summary_by_category(
            category_column='Category',
            value_column='A'
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('count', result.columns)
        self.assertIn('mean', result.columns)
        self.assertIn('median', result.columns)
    
    def test_get_quantiles(self):
        """Test quantile calculation"""
        quantiles = self.stat_analyzer.get_quantiles(column='A')
        
        self.assertIsNotNone(quantiles)
        self.assertIsInstance(quantiles, pd.Series)
        self.assertEqual(len(quantiles), 5)  # Default quantiles
    
    def test_get_quantiles_custom(self):
        """Test quantile calculation with custom quantiles"""
        quantiles = self.stat_analyzer.get_quantiles(column='A', q=[0.1, 0.5, 0.9])
        
        self.assertIsNotNone(quantiles)
        self.assertEqual(len(quantiles), 3)
    
    def test_get_value_counts(self):
        """Test value counts calculation"""
        counts = self.stat_analyzer.get_value_counts(column='Category')
        
        self.assertIsNotNone(counts)
        self.assertIsInstance(counts, pd.Series)
        self.assertEqual(len(counts), 2)
    
    def test_get_value_counts_normalized(self):
        """Test value counts with normalization"""
        counts = self.stat_analyzer.get_value_counts(column='Category', normalize=True)
        
        self.assertIsNotNone(counts)
        # Check that values sum to approximately 1
        self.assertAlmostEqual(counts.sum(), 1.0, places=5)
    
    def test_get_value_counts_with_limit(self):
        """Test value counts with limit"""
        self.mock_loader.data = pd.DataFrame({
            'A': list(range(100))
        })
        
        counts = self.stat_analyzer.get_value_counts(column='A', limit=10)
        
        self.assertIsNotNone(counts)
        self.assertLessEqual(len(counts), 10)
    
    def test_get_missing_values_summary(self):
        """Test missing values summary"""
        # Add some missing values
        self.mock_loader.data.loc[0, 'A'] = np.nan
        self.mock_loader.data.loc[1, 'B'] = np.nan
        
        summary = self.stat_analyzer.get_missing_values_summary()
        
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('Column', summary.columns)
        self.assertIn('Missing Values', summary.columns)
        self.assertIn('Missing (%)', summary.columns)
    
    def test_get_missing_values_summary_no_missing(self):
        """Test missing values summary when no missing values"""
        summary = self.stat_analyzer.get_missing_values_summary()
        
        self.assertIsNotNone(summary)
        # Should still return a DataFrame but with zero missing values
        self.assertEqual(summary['Missing Values'].sum(), 0)
    
    def test_get_outliers_summary_iqr(self):
        """Test outlier detection using IQR method"""
        # Create data with outliers
        self.mock_loader.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        
        outliers = self.stat_analyzer.get_outliers_summary(columns=['A'], method='iqr')
        
        self.assertIsNotNone(outliers)
        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertIn('Column', outliers.columns)
        self.assertIn('Outlier Count', outliers.columns)
        self.assertGreater(outliers['Outlier Count'].sum(), 0)
    
    def test_get_outliers_summary_zscore(self):
        """Test outlier detection using Z-score method"""
        # Create data with outliers
        self.mock_loader.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        
        outliers = self.stat_analyzer.get_outliers_summary(columns=['A'], method='zscore')
        
        self.assertIsNotNone(outliers)
        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertGreater(outliers['Outlier Count'].sum(), 0)
    
    def test_get_outliers_summary_custom_threshold(self):
        """Test outlier detection with custom threshold"""
        outliers = self.stat_analyzer.get_outliers_summary(
            columns=['A'],
            method='iqr',
            threshold=3.0
        )
        
        self.assertIsNotNone(outliers)
        self.assertIsInstance(outliers, pd.DataFrame)
    
    def test_get_outliers_summary_invalid_method(self):
        """Test outlier detection with invalid method"""
        with self.assertRaises(ValueError):
            self.stat_analyzer.get_outliers_summary(method='invalid')
    
    def test_get_descriptive_statistics_no_data(self):
        """Test descriptive statistics when no data is loaded"""
        self.mock_loader.data = None
        stats = self.stat_analyzer.get_descriptive_statistics()
        
        self.assertIsNone(stats)
    
    def test_get_group_by_statistics_invalid_columns(self):
        """Test group by with invalid columns"""
        result = self.stat_analyzer.get_group_by_statistics(
            group_column='Invalid',
            agg_column='A',
            agg_func='Mean'
        )
        
        self.assertIsNone(result)
    
    def test_get_quantiles_invalid_column(self):
        """Test quantiles with invalid column"""
        quantiles = self.stat_analyzer.get_quantiles(column='Invalid')
        
        self.assertIsNone(quantiles)
    
    def test_get_value_counts_invalid_column(self):
        """Test value counts with invalid column"""
        counts = self.stat_analyzer.get_value_counts(column='Invalid')
        
        self.assertIsNone(counts)


if __name__ == '__main__':
    unittest.main()
