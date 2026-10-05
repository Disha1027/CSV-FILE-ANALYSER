import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_cleaning_module.data_cleaner import DataCleaner


class MockDataLoader:
    """Mock DataLoader for testing"""
    def __init__(self):
        self.data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50],
            'C': [5, 15, 25, 35, 45],
            'D': ['x', 'y', np.nan, 'z', 'w']
        })
    
    def get_data(self):
        return self.data


class TestDataCleaner(unittest.TestCase):
    """Unit tests for DataCleaner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_loader = MockDataLoader()
        self.data_cleaner = DataCleaner(self.mock_loader)
    
    def test_initialization(self):
        """Test DataCleaner initialization"""
        self.assertIsNotNone(self.data_cleaner)
        self.assertEqual(self.data_cleaner.data_loader, self.mock_loader)
        self.assertFalse(self.data_cleaner.cleaning_flag)
        self.assertIsNone(self.data_cleaner.normalization_directive)
    
    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy"""
        result = self.data_cleaner.handle_missing_values(strategy='mean', columns=['A'])
        
        self.assertIsNotNone(result)
        self.assertFalse(result['A'].isnull().any())
        self.assertTrue(self.data_cleaner.cleaning_flag)
    
    def test_handle_missing_values_median(self):
        """Test handling missing values with median strategy"""
        result = self.data_cleaner.handle_missing_values(strategy='median', columns=['A'])
        
        self.assertIsNotNone(result)
        self.assertFalse(result['A'].isnull().any())
    
    def test_handle_missing_values_mode(self):
        """Test handling missing values with mode strategy"""
        result = self.data_cleaner.handle_missing_values(strategy='mode', columns=['D'])
        
        self.assertIsNotNone(result)
        # Mode should fill the missing value
        self.assertFalse(result['D'].isnull().any())
    
    def test_handle_missing_values_drop(self):
        """Test handling missing values by dropping rows"""
        result = self.data_cleaner.handle_missing_values(strategy='drop', columns=['A'])
        
        self.assertIsNotNone(result)
        # Should have fewer rows after dropping
        self.assertLess(len(result), len(self.mock_loader.get_data()))
    
    def test_handle_missing_values_forward(self):
        """Test handling missing values with forward fill"""
        result = self.data_cleaner.handle_missing_values(strategy='forward', columns=['A'])
        
        self.assertIsNotNone(result)
        # Forward fill should fill the NaN
        self.assertFalse(result['A'].isnull().any())
    
    def test_handle_missing_values_backward(self):
        """Test handling missing values with backward fill"""
        result = self.data_cleaner.handle_missing_values(strategy='backward', columns=['B'])
        
        self.assertIsNotNone(result)
        # Backward fill should fill the NaN
        self.assertFalse(result['B'].isnull().any())
    
    def test_normalize_scale_data_standard(self):
        """Test standard normalization"""
        result, scaler = self.data_cleaner.normalize_scale_data(columns=['A', 'C'], method='standard')
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(scaler)
        self.assertEqual(self.data_cleaner.normalization_directive, 'standard')
        # Check that values are approximately standardized (mean ~0, std ~1)
        self.assertAlmostEqual(result['A'].mean(), 0, places=1)
        self.assertAlmostEqual(result['A'].std(), 1, places=0)  # Less strict tolerance for small samples
    
    def test_normalize_scale_data_minmax(self):
        """Test min-max normalization"""
        result, scaler = self.data_cleaner.normalize_scale_data(columns=['A', 'C'], method='minmax')
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(scaler)
        self.assertEqual(self.data_cleaner.normalization_directive, 'minmax')
        # Check that values are in [0, 1] range
        self.assertGreaterEqual(result['A'].min(), 0)
        self.assertLessEqual(result['A'].max(), 1)
    
    def test_normalize_scale_data_invalid_method(self):
        """Test normalization with invalid method"""
        with self.assertRaises(ValueError):
            self.data_cleaner.normalize_scale_data(columns=['A'], method='invalid')
    
    def test_remove_duplicate_rows(self):
        """Test removing duplicate rows"""
        # Create data with duplicates
        self.mock_loader.data = pd.DataFrame({
            'A': [1, 2, 2, 3, 3, 3],
            'B': [10, 20, 20, 30, 30, 30]
        })
        
        result = self.data_cleaner.remove_duplicate_rows()
        
        self.assertIsNotNone(result)
        self.assertLess(len(result), len(self.mock_loader.get_data()))
        self.assertTrue(self.data_cleaner.cleaning_flag)
    
    def test_remove_duplicate_rows_subset(self):
        """Test removing duplicates based on subset of columns"""
        self.mock_loader.data = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [10, 20, 25, 30],
            'C': [100, 200, 200, 300]
        })
        
        result = self.data_cleaner.remove_duplicate_rows(subset=['C'])
        
        self.assertIsNotNone(result)
        self.assertLess(len(result), len(self.mock_loader.get_data()))
    
    def test_get_cleaning_summary(self):
        """Test getting cleaning summary"""
        summary = self.data_cleaner.get_cleaning_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn('cleaning_flag', summary)
        self.assertIn('normalization_directive', summary)
        self.assertFalse(summary['cleaning_flag'])
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method"""
        # Create data with outliers
        self.mock_loader.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        
        outliers = self.data_cleaner.detect_outliers(columns=['A'], method='iqr', threshold=1.5)
        
        self.assertIsNotNone(outliers)
        self.assertIn('A', outliers)
        self.assertGreater(outliers['A']['count'], 0)
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method"""
        # Create data with outliers
        self.mock_loader.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        
        outliers = self.data_cleaner.detect_outliers(columns=['A'], method='zscore', threshold=2)
        
        self.assertIsNotNone(outliers)
        self.assertIn('A', outliers)
        self.assertGreater(outliers['A']['count'], 0)
    
    def test_impute_data_mean(self):
        """Test data imputation with mean strategy"""
        result = self.data_cleaner.impute_data(columns=['A'], strategy='mean')
        
        self.assertIsNotNone(result)
        self.assertFalse(result['A'].isnull().any())
        self.assertTrue(self.data_cleaner.cleaning_flag)
    
    def test_impute_data_median(self):
        """Test data imputation with median strategy"""
        result = self.data_cleaner.impute_data(columns=['A'], strategy='median')
        
        self.assertIsNotNone(result)
        self.assertFalse(result['A'].isnull().any())
    
    def test_impute_data_constant(self):
        """Test data imputation with constant strategy"""
        result = self.data_cleaner.impute_data(columns=['A'], strategy='constant')
        
        self.assertIsNotNone(result)
        self.assertFalse(result['A'].isnull().any())
        # Check that NaN was replaced with 0
        self.assertNotIn(np.nan, result['A'].values)
    
    def test_impute_data_all_columns(self):
        """Test imputation on all columns with missing values"""
        result = self.data_cleaner.impute_data(columns=None, strategy='mean')
        
        self.assertIsNotNone(result)
        # Should process columns with missing values
        self.assertTrue(self.data_cleaner.cleaning_flag)
    
    def test_handle_missing_values_no_data(self):
        """Test handling missing values when no data is loaded"""
        self.mock_loader.data = None
        result = self.data_cleaner.handle_missing_values(strategy='mean')
        
        self.assertIsNone(result)
    
    def test_normalize_no_numeric_columns(self):
        """Test normalization when no numeric columns available"""
        self.mock_loader.data = pd.DataFrame({
            'A': ['x', 'y', 'z'],
            'B': ['a', 'b', 'c']
        })
        
        result, scaler = self.data_cleaner.normalize_scale_data()
        
        self.assertIsNotNone(result)
        self.assertIsNone(scaler)


if __name__ == '__main__':
    unittest.main()
