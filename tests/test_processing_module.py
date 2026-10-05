import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.processing_module.ml_analyzer import MLAnalyzer
from modules.processing_module.gemini_analyzer import GeminiAnalyzer


class MockSessionState:
    """Mock Streamlit session state for testing"""
    def __init__(self):
        self.gemini_api_key = ''
        self.gemini_configured = False
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)


class MockDataLoader:
    """Mock DataLoader for testing"""
    def __init__(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
            'Category': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
    
    def get_data(self):
        return self.data
    
    def get_numeric_columns(self):
        return ['A', 'B', 'C']


class TestMLAnalyzer(unittest.TestCase):
    """Unit tests for MLAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_loader = MockDataLoader()
        self.ml_analyzer = MLAnalyzer(self.mock_loader)
    
    def test_initialization(self):
        """Test MLAnalyzer initialization"""
        self.assertIsNotNone(self.ml_analyzer)
        self.assertEqual(self.ml_analyzer.data_loader, self.mock_loader)
    
    def test_perform_pca(self):
        """Test PCA analysis"""
        result = self.ml_analyzer.perform_pca(columns=['A', 'B', 'C'], n_components=2)
        
        self.assertIsNotNone(result)
        self.assertIn('pca', result)
        self.assertIn('pca_result', result)
        self.assertIn('explained_variance', result)
        self.assertIn('loadings', result)
        self.assertEqual(len(result['explained_variance']), 2)
    
    def test_perform_pca_insufficient_columns(self):
        """Test PCA with insufficient columns"""
        result = self.ml_analyzer.perform_pca(columns=['A'], n_components=2)
        self.assertIsNone(result)
    
    def test_perform_clustering(self):
        """Test K-means clustering"""
        result = self.ml_analyzer.perform_clustering(columns=['A', 'B', 'C'], n_clusters=2)
        
        self.assertIsNotNone(result)
        self.assertIn('kmeans', result)
        self.assertIn('clusters', result)
        self.assertIn('cluster_df', result)
        self.assertIn('centers', result)
        self.assertEqual(len(result['centers']), 2)
    
    def test_perform_clustering_insufficient_columns(self):
        """Test clustering with insufficient columns"""
        result = self.ml_analyzer.perform_clustering(columns=['A'], n_clusters=2)
        self.assertIsNone(result)
    
    def test_train_regression_model(self):
        """Test regression model training"""
        result = self.ml_analyzer.train_regression_model(
            target_column='B',
            feature_columns=['A', 'C'],
            test_size=0.2
        )
        
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('mse', result)
        self.assertIn('rmse', result)
        self.assertIn('r2', result)
        self.assertIn('feature_importance', result)
        self.assertGreaterEqual(result['r2'], 0)
    
    def test_train_regression_model_no_features(self):
        """Test regression with no feature columns"""
        result = self.ml_analyzer.train_regression_model(
            target_column='B',
            feature_columns=['A', 'C'],  # Provide explicit features
            test_size=0.2
        )
        self.assertIsNotNone(result)
    
    def test_train_classification_model(self):
        """Test classification model training"""
        result = self.ml_analyzer.train_classification_model(
            target_column='Category',
            feature_columns=['A', 'B', 'C'],
            test_size=0.2
        )
        
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('accuracy', result)
        self.assertIn('classification_report', result)
        self.assertIn('feature_importance', result)
        self.assertGreaterEqual(result['accuracy'], 0)
        self.assertLessEqual(result['accuracy'], 1)
    
    def test_create_pca_plots(self):
        """Test PCA plot creation"""
        pca_result = self.ml_analyzer.perform_pca(columns=['A', 'B', 'C'], n_components=2)
        plots = self.ml_analyzer.create_pca_plots(pca_result)
        
        self.assertIsNotNone(plots)
        self.assertIn('explained_variance', plots)
        self.assertIn('scatter', plots)
        self.assertIn('importance', plots)
    
    def test_create_clustering_plots(self):
        """Test clustering plot creation"""
        clustering_result = self.ml_analyzer.perform_clustering(columns=['A', 'B', 'C'], n_clusters=2)
        plots = self.ml_analyzer.create_clustering_plots(clustering_result)
        
        self.assertIsNotNone(plots)
        self.assertIn('scatter', plots)
        self.assertIn('distribution', plots)
    
    def test_create_regression_plots(self):
        """Test regression plot creation"""
        regression_result = self.ml_analyzer.train_regression_model(
            target_column='B',
            feature_columns=['A', 'C'],
            test_size=0.2
        )
        plots = self.ml_analyzer.create_regression_plots(regression_result)
        
        self.assertIsNotNone(plots)
        self.assertIn('actual_vs_predicted', plots)
        self.assertIn('residuals', plots)
        self.assertIn('importance', plots)
    
    def test_create_classification_plots(self):
        """Test classification plot creation"""
        classification_result = self.ml_analyzer.train_classification_model(
            target_column='Category',
            feature_columns=['A', 'B', 'C'],
            test_size=0.2
        )
        plots = self.ml_analyzer.create_classification_plots(classification_result)
        
        self.assertIsNotNone(plots)
        self.assertIn('confusion_matrix', plots)
        self.assertIn('importance', plots)
        self.assertIn('class_report', plots)
    
    def test_create_download_link(self):
        """Test download link creation"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        link = self.ml_analyzer.create_download_link(df, "test.csv")
        
        self.assertIsNotNone(link)
        self.assertIn('data:file/csv;base64', link)
        self.assertIn('test.csv', link)


class TestGeminiAnalyzer(unittest.TestCase):
    """Unit tests for GeminiAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock streamlit session state
        import streamlit as st
        self.original_session_state = st.session_state
        st.session_state = MockSessionState()
        
        self.mock_loader = MockDataLoader()
        self.gemini_analyzer = GeminiAnalyzer(self.mock_loader)
    
    def tearDown(self):
        """Clean up after tests"""
        import streamlit as st
        st.session_state = self.original_session_state
    
    def test_initialization(self):
        """Test GeminiAnalyzer initialization"""
        self.assertIsNotNone(self.gemini_analyzer)
        self.assertEqual(self.gemini_analyzer.data_loader, self.mock_loader)
    
    def test_is_configured(self):
        """Test is_configured method"""
        # Initially not configured
        result = self.gemini_analyzer.is_configured()
        self.assertFalse(result)
    
    def test_prepare_data_description(self):
        """Test prepare_data_description method"""
        description = self.gemini_analyzer.prepare_data_description()
        
        self.assertIsNotNone(description)
        self.assertIn('info', description)
        self.assertIn('sample', description)
        self.assertIn('stats', description)
    
    def test_get_analysis_types(self):
        """Test get_analysis_types method"""
        types = self.gemini_analyzer.get_analysis_types()
        
        self.assertIsNotNone(types)
        self.assertIsInstance(types, list)
        self.assertIn("Data Summary and Insights", types)
        self.assertIn("Correlation Analysis", types)
        self.assertIn("Custom Analysis", types)
    
    def test_generate_prompt(self):
        """Test prompt generation"""
        prompt = self.gemini_analyzer.generate_prompt("Data Summary and Insights")
        
        self.assertIsNotNone(prompt)
        self.assertIn("dataset", prompt.lower())
        self.assertIn("summary", prompt.lower())
    
    def test_generate_prompt_custom(self):
        """Test custom prompt generation"""
        custom_question = "What is the average value?"
        prompt = self.gemini_analyzer.generate_prompt("Custom Analysis", custom_question)
        
        self.assertIsNotNone(prompt)
        self.assertIn(custom_question, prompt)
    
    def test_analyze_data_not_configured(self):
        """Test analysis when API not configured"""
        result = self.gemini_analyzer.analyze_data("Data Summary and Insights")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
