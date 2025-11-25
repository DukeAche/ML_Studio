import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from utils.viz import Visualization
from utils.model import ModelTrainer
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="No-Code ML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Title and description
st.title("🤖 No-Code Machine Learning Studio")
st.markdown("**Transform your data into trained ML models without writing code**")

# Sidebar for data ingestion
with st.sidebar:
    st.header("📊 Data Ingestion")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # File upload options
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV", "Upload Excel", "URL Link"]
    )
    
    df = None
    
    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = data_loader.load_csv(uploaded_file)
    
    elif upload_option == "Upload Excel":
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
        if uploaded_file is not None:
            df = data_loader.load_excel(uploaded_file)
    
    elif upload_option == "URL Link":
        url = st.text_input("Enter data URL (CSV format)")
        if url:
            df = data_loader.load_from_url(url)
    
    # Store dataframe in session state
    if df is not None:
        st.session_state.df = df
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📈 EDA", "📊 Visualization", "🤖 ML Training", "📋 Results"])

with tab1:
    st.header("Exploratory Data Analysis")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Toggle for raw data display
        if st.checkbox("Show raw data"):
            st.dataframe(df)
        
        # Data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Column Information")
            info_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(info_df)
        
        with col2:
            st.subheader("Summary Statistics")
            st.dataframe(df.describe())
        
        # Missing values visualization
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if not missing_df.empty:
            fig_missing = px.bar(missing_df, x='Column', y='Missing Count', 
                               title='Missing Values by Column',
                               color='Missing Percentage',
                               color_continuous_scale='Reds')
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    else:
        st.info("📤 Please upload data in the sidebar to begin analysis")

with tab2:
    st.header("Custom Data Visualization")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        viz = Visualization()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_axis = st.selectbox("X-axis", df.columns.tolist())
        with col2:
            y_axis = st.selectbox("Y-axis", df.columns.tolist())
        with col3:
            color_col = st.selectbox("Color (optional)", ["None"] + df.columns.tolist())
        with col4:
            chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box"])
        
        if st.button("Generate Plot"):
            fig = viz.create_plot(df, x_axis, y_axis, color_col, chart_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📤 Please upload data in the sidebar to create visualizations")

with tab3:
    st.header("Machine Learning Training")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        model_trainer = ModelTrainer()
        
        # Preprocessing section
        st.subheader("🔧 Data Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing value strategy
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ["mean", "median", "mode", "drop"]
            )
            
            # Target variable selection
            target_col = st.selectbox("Target Variable (y)", df.columns.tolist())
        
        with col2:
            # Feature selection
            feature_cols = st.multiselect(
                "Feature Variables (X)", 
                [col for col in df.columns if col != target_col],
                default=[col for col in df.columns if col != target_col][:4]
            )
            
            # Train/test split
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                X, y, preprocessed_df = model_trainer.preprocess_data(
                    df, feature_cols, target_col, missing_strategy
                )
                st.session_state.preprocessed_df = preprocessed_df
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.test_size = test_size
                st.success("✅ Data preprocessed successfully!")
                st.dataframe(preprocessed_df.head())
        
        # Model training section
        if st.session_state.get('X') is not None:
            st.subheader("🎯 Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Task type selection
                task_type = st.selectbox("Task Type", ["Classification", "Regression"])
            
            with col2:
                # Algorithm selection based on task type
                if task_type == "Classification":
                    algorithms = ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"]
                else:
                    algorithms = ["Linear Regression", "Random Forest", "Decision Tree", "SVR"]
                
                algorithm = st.selectbox("Algorithm", algorithms)
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    try:
                        results = model_trainer.train_model(
                            st.session_state.X, 
                            st.session_state.y, 
                            task_type, 
                            algorithm,
                            test_size
                        )
                        st.session_state.model_results = results
                        st.session_state.model_trained = True
                        st.success("✅ Model trained successfully!")
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
    
    else:
        st.info("📤 Please upload data in the sidebar to start ML training")

with tab4:
    st.header("Model Results & Evaluation")
    
    if st.session_state.model_trained and st.session_state.model_results:
        results = st.session_state.model_results
        
        # Display metrics
        st.subheader("📊 Performance Metrics")
        
        if results['task_type'] == 'Classification':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{results['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{results['recall']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("F1-Score", f"{results['f1_score']:.4f}")
            with col2:
                st.metric("Training Time", f"{results['training_time']:.2f}s")
        
        else:  # Regression
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{results['r2_score']:.4f}")
            with col2:
                st.metric("Mean Squared Error", f"{results['mse']:.4f}")
            with col3:
                st.metric("Training Time", f"{results['training_time']:.2f}s")
        
        # Feature importance
        if 'feature_importance' in results:
            st.subheader("🔍 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': results['feature_names'],
                'Importance': results['feature_importance']
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(importance_df.head(10), 
                                  x='Importance', y='Feature',
                                  title='Top 10 Feature Importances',
                                  orientation='h')
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model parameters
        st.subheader("⚙️ Model Parameters")
        st.json(results['model_params'])
        
        # Download predictions
        if 'predictions' in results:
            st.subheader("📥 Download Results")
            predictions_df = pd.DataFrame({
                'Actual': results['y_test'],
                'Predicted': results['predictions']
            })
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="model_predictions.csv",
                mime="text/csv"
            )
    
    else:
        st.info("🎯 Train a model first to see results here")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | No-Code ML Studio")