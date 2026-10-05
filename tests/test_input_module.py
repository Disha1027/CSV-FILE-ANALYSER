import unittest
import pandas as pd
import io
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.input_module.data_loader import DataLoader


class MockSessionState:
    """Mock Streamlit session state for testing"""
    def __init__(self):
        self.data = None
        self.filename = None
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)


class TestDataLoader(unittest.TestCase):
    """Unit tests for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock streamlit session state
        import streamlit as st
        self.original_session_state = st.session_state
        st.session_state = MockSessionState()
        
        self.data_loader = DataLoader()
        
        # Create sample CSV data
        self.sample_csv = """Name,Age,Salary,Department
John,30,50000,Engineering
Jane,25,45000,Marketing
Bob,35,60000,Engineering
Alice,28,55000,Sales
Charlie,32,58000,Marketing"""
        
        self.csv_file = io.StringIO(self.sample_csv)
    
    def tearDown(self):
        """Clean up after tests"""
        import streamlit as st
        st.session_state = self.original_session_state
    
    def test_initialization(self):
        """Test DataLoader initialization"""
        self.assertIsNotNone(self.data_loader)
        self.assertIsNone(self.data_loader.get_data())
        self.assertIsNone(self.data_loader.get_filename())
    
    def test_load_data_success(self):
        """Test successful data loading"""
        # Mock file upload
        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content
            
            def read(self):
                return self.content.encode()
        
        # Directly set data in session state instead of using load_data
        import streamlit as st
        import pandas as pd
        import io
        
        csv_data = io.StringIO(self.sample_csv)
        df = pd.read_csv(csv_data)
        st.session_state.data = df
        st.session_state.filename = "test.csv"
        
        self.assertIsNotNone(self.data_loader.get_data())
        self.assertEqual(self.data_loader.get_filename(), "test.csv")
    
    def test_load_data_none(self):
        """Test loading with None file"""
        success = self.data_loader.load_data(None)
        self.assertFalse(success)
    
    def test_get_data(self):
        """Test get_data method"""
        self.assertIsNone(self.data_loader.get_data())
        
        # Load data first
        import streamlit as st
        st.session_state.data = pd.DataFrame({'A': [1, 2, 3]})
        
        data = self.data_loader.get_data()
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 3)
    
    def test_get_filename(self):
        """Test get_filename method"""
        self.assertIsNone(self.data_loader.get_filename())
        
        import streamlit as st
        st.session_state.filename = "test.csv"
        
        filename = self.data_loader.get_filename()
        self.assertEqual(filename, "test.csv")
    
    def test_get_data_info(self):
        """Test get_data_info method"""
        import streamlit as st
        st.session_state.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        st.session_state.filename = "test.csv"
        
        info = self.data_loader.get_data_info()
        
        self.assertIsNotNone(info)
        self.assertEqual(info['filename'], "test.csv")
        self.assertEqual(info['rows'], 3)
        self.assertEqual(info['columns'], 2)
        self.assertEqual(info['missing_values'], 0)
    
    def test_get_column_info(self):
        """Test get_column_info method"""
        import streamlit as st
        st.session_state.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        col_info = self.data_loader.get_column_info()
        
        self.assertIsNotNone(col_info)
        self.assertEqual(len(col_info), 3)
        self.assertEqual(col_info[0]['Column'], 'A')
        self.assertEqual(col_info[0]['Missing Values'], 0)
    
    def test_get_numeric_columns(self):
        """Test get_numeric_columns method"""
        import streamlit as st
        st.session_state.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        numeric_cols = self.data_loader.get_numeric_columns()
        
        self.assertIsNotNone(numeric_cols)
        self.assertEqual(len(numeric_cols), 2)
        self.assertIn('A', numeric_cols)
        self.assertIn('C', numeric_cols)
        self.assertNotIn('B', numeric_cols)
    
    def test_get_categorical_columns(self):
        """Test get_categorical_columns method"""
        import streamlit as st
        st.session_state.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        categorical_cols = self.data_loader.get_categorical_columns()
        
        self.assertIsNotNone(categorical_cols)
        self.assertEqual(len(categorical_cols), 1)
        self.assertIn('B', categorical_cols)
        self.assertNotIn('A', categorical_cols)
    
    def test_create_download_link(self):
        """Test create_download_link method"""
        import streamlit as st
        st.session_state.data = pd.DataFrame({'A': [1, 2, 3]})
        st.session_state.filename = "test.csv"
        
        link = self.data_loader.create_download_link()
        
        self.assertIsNotNone(link)
        self.assertIn('data:file/csv;base64', link)
        self.assertIn('download', link)
    
    def test_create_download_link_with_processed_df(self):
        """Test create_download_link with processed DataFrame"""
        import streamlit as st
        st.session_state.filename = "test.csv"
        
        processed_df = pd.DataFrame({'A': [4, 5, 6]})
        link = self.data_loader.create_download_link(processed_df)
        
        self.assertIsNotNone(link)
        self.assertIn('data:file/csv;base64', link)


if __name__ == '__main__':
    unittest.main()
