import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_exploration_module.data_explorer import DataExplorer


class MockDataLoader:
    """Mock DataLoader for testing"""
    def __init__(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['apple', 'banana', 'apple', 'cherry', 'banana'],
            'D': [100, 200, 300, 400, 500]
        })
    
    def get_data(self):
        return self.data


class TestDataExplorer(unittest.TestCase):
    """Unit tests for DataExplorer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_loader = MockDataLoader()
        self.data_explorer = DataExplorer(self.mock_loader)
    
    def test_initialization(self):
        """Test DataExplorer initialization"""
        self.assertIsNotNone(self.data_explorer)
        self.assertEqual(self.data_explorer.data_loader, self.mock_loader)
    
    def test_filter_rows_simple(self):
        """Test simple row filtering"""
        filters = {'A': 1}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['A'].values[0], 1)
    
    def test_filter_rows_gt(self):
        """Test filtering with greater than operator"""
        filters = {'A__gt': 2}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result['A'] > 2))
    
    def test_filter_rows_lt(self):
        """Test filtering with less than operator"""
        filters = {'B__lt': 30}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['B'] < 30))
    
    def test_filter_rows_gte(self):
        """Test filtering with greater than or equal operator"""
        filters = {'A__gte': 3}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result['A'] >= 3))
    
    def test_filter_rows_lte(self):
        """Test filtering with less than or equal operator"""
        filters = {'B__lte': 30}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result['B'] <= 30))
    
    def test_filter_rows_contains(self):
        """Test filtering with contains operator"""
        filters = {'C__contains': 'app'}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
    
    def test_filter_rows_multiple(self):
        """Test filtering with multiple conditions"""
        filters = {'A__gt': 2, 'B__lt': 50}
        result = self.data_explorer.filter_rows(filters)
        
        self.assertIsNotNone(result)
        self.assertTrue(all(result['A'] > 2))
        self.assertTrue(all(result['B'] < 50))
    
    def test_sort_dataset_single_column(self):
        """Test sorting by single column"""
        result = self.data_explorer.sort_dataset('A', ascending=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(list(result['A']), [1, 2, 3, 4, 5])
    
    def test_sort_dataset_descending(self):
        """Test sorting in descending order"""
        result = self.data_explorer.sort_dataset('B', ascending=False)
        
        self.assertIsNotNone(result)
        self.assertEqual(list(result['B']), [50, 40, 30, 20, 10])
    
    def test_sort_dataset_multiple_columns(self):
        """Test sorting by multiple columns"""
        result = self.data_explorer.sort_dataset(['C', 'A'], ascending=True)
        
        self.assertIsNotNone(result)
        # Should be sorted by C then A
        self.assertEqual(result.iloc[0]['C'], 'apple')
        self.assertEqual(result.iloc[0]['A'], 1)
    
    def test_search_rows(self):
        """Test searching for rows"""
        result = self.data_explorer.search_rows('apple')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
    
    def test_search_rows_case_sensitive(self):
        """Test search with case sensitivity"""
        result = self.data_explorer.search_rows('APPLE', case_sensitive=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)  # No match with case sensitivity
    
    def test_search_rows_specific_columns(self):
        """Test search in specific columns"""
        result = self.data_explorer.search_rows('apple', columns=['C'])
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
    
    def test_get_unique_values(self):
        """Test getting unique values"""
        unique = self.data_explorer.get_unique_values('C')
        
        self.assertIsNotNone(unique)
        self.assertEqual(len(unique), 3)
        self.assertIn('apple', unique)
        self.assertIn('banana', unique)
        self.assertIn('cherry', unique)
    
    def test_get_unique_values_with_limit(self):
        """Test getting unique values with limit"""
        unique = self.data_explorer.get_unique_values('C', limit=2)
        
        self.assertIsNotNone(unique)
        self.assertLessEqual(len(unique), 2)
    
    def test_get_value_counts(self):
        """Test getting value counts"""
        counts = self.data_explorer.get_value_counts('C')
        
        self.assertIsNotNone(counts)
        self.assertIsInstance(counts, pd.Series)
        self.assertEqual(counts['apple'], 2)
        self.assertEqual(counts['banana'], 2)
        self.assertEqual(counts['cherry'], 1)
    
    def test_get_value_counts_normalized(self):
        """Test getting value counts with normalization"""
        counts = self.data_explorer.get_value_counts('C', normalize=True)
        
        self.assertIsNotNone(counts)
        self.assertAlmostEqual(counts.sum(), 1.0, places=5)
    
    def test_get_value_counts_with_limit(self):
        """Test getting value counts with limit"""
        counts = self.data_explorer.get_value_counts('C', limit=2)
        
        self.assertIsNotNone(counts)
        self.assertLessEqual(len(counts), 2)
    
    def test_filter_by_range(self):
        """Test filtering by numeric range"""
        result = self.data_explorer.filter_by_range('A', min_value=2, max_value=4)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertTrue(all((result['A'] >= 2) & (result['A'] <= 4)))
    
    def test_filter_by_range_min_only(self):
        """Test filtering by range with only minimum"""
        result = self.data_explorer.filter_by_range('A', min_value=3)
        
        self.assertIsNotNone(result)
        self.assertTrue(all(result['A'] >= 3))
    
    def test_filter_by_range_max_only(self):
        """Test filtering by range with only maximum"""
        result = self.data_explorer.filter_by_range('A', max_value=3)
        
        self.assertIsNotNone(result)
        self.assertTrue(all(result['A'] <= 3))
    
    def test_filter_by_date_range(self):
        """Test filtering by date range"""
        # Create data with dates
        self.mock_loader.data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5),
            'Value': [10, 20, 30, 40, 50]
        })
        
        result = self.data_explorer.filter_by_date_range(
            'Date',
            start_date='2020-01-02',
            end_date='2020-01-04'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
    
    def test_get_sample(self):
        """Test getting random sample"""
        sample = self.data_explorer.get_sample(n=3, random_state=42)
        
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample), 3)
    
    def test_get_sample_larger_than_data(self):
        """Test getting sample larger than data"""
        sample = self.data_explorer.get_sample(n=10)
        
        self.assertIsNotNone(sample)
        # Should return all data if sample size > data size
        self.assertEqual(len(sample), 5)
    
    def test_get_head(self):
        """Test getting first n rows"""
        head = self.data_explorer.get_head(n=3)
        
        self.assertIsNotNone(head)
        self.assertEqual(len(head), 3)
        self.assertEqual(list(head['A']), [1, 2, 3])
    
    def test_get_tail(self):
        """Test getting last n rows"""
        tail = self.data_explorer.get_tail(n=3)
        
        self.assertIsNotNone(tail)
        self.assertEqual(len(tail), 3)
        self.assertEqual(list(tail['A']), [3, 4, 5])
    
    def test_filter_by_multiple_conditions_and(self):
        """Test filtering with multiple conditions using AND logic"""
        conditions = [
            {'A__gt': 2},
            {'B__lt': 50}
        ]
        result = self.data_explorer.filter_by_multiple_conditions(conditions, logic='and')
        
        self.assertIsNotNone(result)
        self.assertTrue(all(result['A'] > 2))
        self.assertTrue(all(result['B'] < 50))
    
    def test_filter_by_multiple_conditions_or(self):
        """Test filtering with multiple conditions using OR logic"""
        conditions = [
            {'A': 1},
            {'A': 5}
        ]
        result = self.data_explorer.filter_by_multiple_conditions(conditions, logic='or')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
    
    def test_filter_rows_no_data(self):
        """Test filtering when no data is loaded"""
        self.mock_loader.data = None
        result = self.data_explorer.filter_rows({'A': 1})
        
        self.assertIsNone(result)
    
    def test_sort_dataset_no_data(self):
        """Test sorting when no data is loaded"""
        self.mock_loader.data = None
        result = self.data_explorer.sort_dataset('A')
        
        self.assertIsNone(result)
    
    def test_get_unique_values_invalid_column(self):
        """Test getting unique values for invalid column"""
        unique = self.data_explorer.get_unique_values('Invalid')
        
        self.assertEqual(unique, [])
    
    def test_filter_by_range_invalid_column(self):
        """Test filtering by range with invalid column"""
        result = self.data_explorer.filter_by_range('Invalid', min_value=1, max_value=10)
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
