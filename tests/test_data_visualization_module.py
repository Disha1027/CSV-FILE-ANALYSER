import unittest
import pandas as pd
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_visualization_module.visualizer import Visualizer


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


class TestVisualizer(unittest.TestCase):
    """Unit tests for Visualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_loader = MockDataLoader()
        self.visualizer = Visualizer(self.mock_loader)
    
    def test_initialization(self):
        """Test Visualizer initialization"""
        self.assertIsNotNone(self.visualizer)
        self.assertEqual(self.visualizer.data_loader, self.mock_loader)
    
    def test_create_distribution_plots(self):
        """Test creating distribution plots"""
        hist_fig, box_fig = self.visualizer.create_distribution_plots('A')
        
        self.assertIsNotNone(hist_fig)
        self.assertIsNotNone(box_fig)
        self.assertEqual(hist_fig.get_axes()[0].get_title(), 'Distribution of A')
        self.assertEqual(box_fig.get_axes()[0].get_title(), 'Box Plot of A')
    
    def test_create_distribution_plots_invalid_column(self):
        """Test creating distribution plots with invalid column"""
        hist_fig, box_fig = self.visualizer.create_distribution_plots('Invalid')
        
        self.assertIsNone(hist_fig)
        self.assertIsNone(box_fig)
    
    def test_create_correlation_heatmap(self):
        """Test creating correlation heatmap"""
        fig = self.visualizer.create_correlation_heatmap()
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_axes()[0].get_title(), 'Correlation Matrix')
    
    def test_create_correlation_heatmap_specific_columns(self):
        """Test creating correlation heatmap with specific columns"""
        fig = self.visualizer.create_correlation_heatmap(columns=['A', 'B'])
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_axes()[0].get_title(), 'Correlation Matrix')
    
    def test_create_correlation_heatmap_insufficient_columns(self):
        """Test creating correlation heatmap with insufficient columns"""
        fig = self.visualizer.create_correlation_heatmap(columns=['A'])
        
        self.assertIsNone(fig)
    
    def test_create_categorical_plot(self):
        """Test creating categorical plot"""
        fig = self.visualizer.create_categorical_plot('Category')
        
        self.assertIsNotNone(fig)
        self.assertIn('Category', fig.get_axes()[0].get_title())
    
    def test_create_categorical_plot_invalid_column(self):
        """Test creating categorical plot with invalid column"""
        fig = self.visualizer.create_categorical_plot('Invalid')
        
        self.assertIsNone(fig)
    
    def test_create_scatter_plot(self):
        """Test creating scatter plot"""
        fig = self.visualizer.create_scatter_plot('A', 'B')
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_axes()[0].get_title(), 'Scatter Plot: A vs B')
    
    def test_create_scatter_plot_invalid_columns(self):
        """Test creating scatter plot with invalid columns"""
        fig = self.visualizer.create_scatter_plot('Invalid', 'B')
        
        self.assertIsNone(fig)
    
    def test_create_group_by_plot(self):
        """Test creating group by plot"""
        grouped_data, fig = self.visualizer.create_group_by_plot(
            group_column='Category',
            agg_column='A',
            agg_func='Mean'
        )
        
        self.assertIsNotNone(grouped_data)
        self.assertIsNotNone(fig)
        self.assertIn('Category', fig.get_axes()[0].get_title())
    
    def test_create_group_by_plot_invalid_columns(self):
        """Test creating group by plot with invalid columns"""
        grouped_data, fig = self.visualizer.create_group_by_plot(
            group_column='Invalid',
            agg_column='A',
            agg_func='Mean'
        )
        
        self.assertIsNone(grouped_data)
        self.assertIsNone(fig)
    
    def test_create_pairplot(self):
        """Test creating pairplot"""
        pairplot = self.visualizer.create_pairplot(columns=['A', 'B', 'C'])
        
        self.assertIsNotNone(pairplot)
        # Check that the figure exists and has the expected title
        self.assertIsNotNone(pairplot.fig)
    
    def test_create_pairplot_insufficient_columns(self):
        """Test creating pairplot with insufficient columns"""
        pairplot = self.visualizer.create_pairplot(columns=['A'])
        
        self.assertIsNone(pairplot)
    
    def test_create_pairplot_with_hue(self):
        """Test creating pairplot with hue"""
        pairplot = self.visualizer.create_pairplot(columns=['A', 'B'], hue='Category')
        
        self.assertIsNotNone(pairplot)
    
    def test_create_time_series_plot(self):
        """Test creating time series plot"""
        # Create data with dates
        self.mock_loader.data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        fig = self.visualizer.create_time_series_plot('Date', 'Value')
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_axes()[0].get_title(), 'Time Series: Value')
    
    def test_create_time_series_plot_with_frequency(self):
        """Test creating time series plot with resampling"""
        # Create data with dates
        self.mock_loader.data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        fig = self.visualizer.create_time_series_plot('Date', 'Value', freq='2D')
        
        self.assertIsNotNone(fig)
    
    def test_create_time_series_plot_invalid_columns(self):
        """Test creating time series plot with invalid columns"""
        fig = self.visualizer.create_time_series_plot('Invalid', 'A')
        
        self.assertIsNone(fig)
    
    def test_create_distribution_plots_no_data(self):
        """Test creating distribution plots when no data is loaded"""
        self.mock_loader.data = None
        hist_fig, box_fig = self.visualizer.create_distribution_plots('A')
        
        self.assertIsNone(hist_fig)
        self.assertIsNone(box_fig)
    
    def test_create_correlation_heatmap_no_data(self):
        """Test creating correlation heatmap when no data is loaded"""
        self.mock_loader.data = None
        fig = self.visualizer.create_correlation_heatmap()
        
        self.assertIsNone(fig)
    
    def test_create_categorical_plot_no_data(self):
        """Test creating categorical plot when no data is loaded"""
        self.mock_loader.data = None
        fig = self.visualizer.create_categorical_plot('Category')
        
        self.assertIsNone(fig)
    
    def test_create_scatter_plot_no_data(self):
        """Test creating scatter plot when no data is loaded"""
        self.mock_loader.data = None
        fig = self.visualizer.create_scatter_plot('A', 'B')
        
        self.assertIsNone(fig)
    
    def test_create_group_by_plot_no_data(self):
        """Test creating group by plot when no data is loaded"""
        self.mock_loader.data = None
        grouped_data, fig = self.visualizer.create_group_by_plot(
            group_column='Category',
            agg_column='A',
            agg_func='Mean'
        )
        
        self.assertIsNone(grouped_data)
        self.assertIsNone(fig)
    
    def test_create_pairplot_no_data(self):
        """Test creating pairplot when no data is loaded"""
        self.mock_loader.data = None
        pairplot = self.visualizer.create_pairplot()
        
        self.assertIsNone(pairplot)
    
    def test_create_time_series_plot_no_data(self):
        """Test creating time series plot when no data is loaded"""
        self.mock_loader.data = None
        fig = self.visualizer.create_time_series_plot('Date', 'Value')
        
        self.assertIsNone(fig)
    
    def test_create_group_by_plot_different_aggregations(self):
        """Test group by plot with different aggregation functions"""
        for agg_func in ['Mean', 'Median', 'Sum', 'Min', 'Max', 'Count']:
            grouped_data, fig = self.visualizer.create_group_by_plot(
                group_column='Category',
                agg_column='A',
                agg_func=agg_func
            )
            
            self.assertIsNotNone(grouped_data)
            self.assertIsNotNone(fig)
            self.assertIn(agg_func, fig.get_axes()[0].get_title())
    
    def test_create_categorical_plot_many_categories(self):
        """Test categorical plot with many categories"""
        # Create data with many categories
        self.mock_loader.data = pd.DataFrame({
            'Category': [f'Cat{i}' for i in range(25)],
            'Value': list(range(25))
        })
        
        fig = self.visualizer.create_categorical_plot('Category')
        
        self.assertIsNotNone(fig)
        # Should limit to top 20 categories
        self.assertIn('Top 20', fig.get_axes()[0].get_title())


if __name__ == '__main__':
    unittest.main()
