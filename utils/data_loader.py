import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO

class DataLoader:
    """Handles data loading from various sources"""
    
    @st.cache_data
    def load_csv(_self, uploaded_file):
        """Load CSV file into DataFrame"""
        try:
            # Read the uploaded file
            bytes_data = uploaded_file.read()
            # Decode to string
            string_data = StringIO(bytes_data.decode("UTF-8"))
            # Load as DataFrame
            df = pd.read_csv(string_data)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None
    
    @st.cache_data
    def load_excel(_self, uploaded_file):
        """Load Excel file into DataFrame"""
        try:
            # Read the uploaded file
            bytes_data = uploaded_file.read()
            # Load as DataFrame
            df = pd.read_excel(BytesIO(bytes_data))
            return df
        except Exception as e:
            st.error(f"Error loading Excel: {str(e)}")
            return None
    
    @st.cache_data
    def load_from_url(_self, url):
        """Load CSV data from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Try to read as CSV
            df = pd.read_csv(StringIO(response.text))
            return df
        except requests.RequestException as e:
            st.error(f"Error fetching URL: {str(e)}")
            return None
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV from URL: {str(e)}")
            return None
    
    def detect_file_format(self, filename):
        """Detect file format from filename"""
        if filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith(('.xlsx', '.xls')):
            return 'excel'
        else:
            return 'unknown'
    
    def validate_dataframe(self, df):
        """Validate loaded DataFrame"""
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if df.shape[0] == 0:
            return False, "No rows in DataFrame"
        
        if df.shape[1] == 0:
            return False, "No columns in DataFrame"
        
        return True, "Valid DataFrame"
    
    def get_data_summary(self, df):
        """Get summary information about the DataFrame"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum()
        }
        return summary