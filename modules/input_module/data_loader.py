import pandas as pd
import streamlit as st
import io
import base64
import hashlib

class DataLoader:
    """
    A class for loading, preprocessing, and managing CSV data.
    """
    
    def __init__(self):
        """Initialize the DataLoader class."""
        # Initialize session state variables if they don't exist
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filename' not in st.session_state:
            st.session_state.filename = None
        if 'file_size_bytes' not in st.session_state:
            st.session_state.file_size_bytes = None
        if 'original_data' not in st.session_state:
            st.session_state.original_data = None
        if 'upload_signature' not in st.session_state:
            st.session_state.upload_signature = None
        if 'transformations' not in st.session_state:
            st.session_state.transformations = []
    
    def create_upload_widget(self):
        """
        Create a file uploader widget for CSV files.
        
        Returns:
            uploaded_file: The uploaded file object or None
        """
        uploaded_file = st.file_uploader("Upload your CSV file", type=None, help="Please upload .CSV files only")
        return uploaded_file
    
    def load_data(self, uploaded_file):
        """
        Load data from an uploaded CSV file.
        
        Args:
            uploaded_file: The uploaded file object
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if uploaded_file is not None:
            filename = getattr(uploaded_file, "name", "")
            if not filename.lower().endswith(".csv"):
                st.error("Please upload .CSV files only")
                return False

            try:
                file_id = getattr(uploaded_file, "file_id", None)
                file_size = getattr(uploaded_file, "size", None)
                if file_id is not None:
                    signature = f"{file_id}:{filename}:{file_size}"
                    if signature == st.session_state.upload_signature and st.session_state.data is not None:
                        return True
                    content_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
                else:
                    content_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
                    signature = hashlib.sha256(content_bytes).hexdigest()
                    if signature == st.session_state.upload_signature and st.session_state.data is not None:
                        return True

                if content_bytes[:5].startswith(b"%PDF"):
                    st.error("Please upload .CSV files only")
                    return False

                # Parse the upload once; future reruns return above using its stable upload id.
                df = pd.read_csv(io.BytesIO(content_bytes))
                if df.empty or df.shape[1] == 0:
                    raise ValueError("The CSV contains no data rows to analyze.")
                st.session_state.data = df
                st.session_state.original_data = df.copy(deep=True)
                st.session_state.transformations = []
                st.session_state.upload_signature = signature
                for key in ("eda_anomaly_result", "eda_time_result"):
                    st.session_state.pop(key, None)
                st.session_state.filename = uploaded_file.name
                st.session_state.file_size_bytes = getattr(uploaded_file, "size", None)
                return True
            except Exception:
                st.error("Please upload .CSV files only")
                return False
        return False
    
    def get_data(self):
        """
        Get the currently loaded data.
        
        Returns:
            pandas.DataFrame: The loaded data or None if no data is loaded
        """
        return st.session_state.data
    
    def get_filename(self):
        """
        Get the filename of the currently loaded data.
        
        Returns:
            str: The filename or None if no file is loaded
        """
        return st.session_state.filename
    
    def get_data_info(self):
        """
        Get basic information about the loaded data.
        
        Returns:
            dict: A dictionary containing basic information about the data
        """
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024**2:
                memory_str = f"{memory_usage/1024:.2f} KB"
            else:
                memory_str = f"{memory_usage/(1024**2):.2f} MB"
            
            # Count missing values
            missing_values = df.isnull().sum().sum()
            
            return {
                "filename": st.session_state.filename,
                "file_size_bytes": st.session_state.file_size_bytes,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage": memory_str,
                "missing_values": missing_values
            }
        return None

    def remove_duplicates(self, subset=None):
        """Apply explicit duplicate removal while retaining the original frame for reset."""
        if st.session_state.data is None:
            return 0
        before = len(st.session_state.data)
        st.session_state.data = st.session_state.data.drop_duplicates(subset=subset).copy()
        for key in ("eda_anomaly_result", "eda_time_result"):
            st.session_state.pop(key, None)
        removed = before - len(st.session_state.data)
        st.session_state.transformations.append({"operation": "remove_duplicates", "subset": subset, "rows_removed": removed})
        return removed

    def reset_data(self):
        """Restore the exact uploaded frame and clear recorded transformations."""
        if st.session_state.original_data is not None:
            st.session_state.data = st.session_state.original_data.copy(deep=True)
            st.session_state.transformations = []
            for key in ("eda_anomaly_result", "eda_time_result"):
                st.session_state.pop(key, None)
    
    def get_column_info(self):
        """
        Get detailed information about each column in the data.
        
        Returns:
            list: A list of dictionaries containing information about each column
        """
        if st.session_state.data is not None:
            df = st.session_state.data
            
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique = df[col].nunique()
                missing = df[col].isnull().sum()
                missing_pct = (missing / max(len(df), 1)) * 100
                
                col_info.append({
                    "Column": col,
                    "Data Type": dtype,
                    "Unique Values": unique,
                    "Missing Values": missing,
                    "Missing (%)": f"{missing_pct:.2f}%"
                })
            
            return col_info
        return None
    
    def get_numeric_columns(self):
        """
        Get a list of numeric columns in the data.
        
        Returns:
            list: A list of numeric column names
        """
        if st.session_state.data is not None:
            return st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        return []
    
    def get_categorical_columns(self):
        """
        Get a list of categorical columns in the data.
        
        Returns:
            list: A list of categorical column names
        """
        if st.session_state.data is not None:
            return st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        return []
    
    def create_download_link(self, processed_df=None):
        """
        Create a download link for the data.
        
        Args:
            processed_df (pandas.DataFrame, optional): A processed dataframe to download.
                If None, the original data will be used.
                
        Returns:
            str: HTML link for downloading the data
        """
        if processed_df is None and st.session_state.data is not None:
            processed_df = st.session_state.data
        
        if processed_df is not None and st.session_state.filename is not None:
            csv = processed_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            filename = st.session_state.filename.split(".")[0]
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}_processed.csv">Download CSV File</a>'
        return None
