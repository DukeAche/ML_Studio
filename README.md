# No-Code Machine Learning Studio

A comprehensive, modular Streamlit application that enables non-technical users to perform end-to-end machine learning workflows without writing code.

## 🚀 Features

### Data Ingestion
- **CSV Upload**: Direct file upload with drag-and-drop support
- **Excel Support**: Handle .xlsx and .xls files with automatic format detection
- **URL Loading**: Load data directly from web URLs
- **Format Auto-Detection**: Intelligent file format recognition

### Exploratory Data Analysis (EDA)
- **Data Overview**: Shape, memory usage, and column information
- **Summary Statistics**: Comprehensive statistical analysis with `df.describe()`
- **Missing Values Analysis**: Visual identification and quantification of missing data
- **Data Type Detection**: Automatic column type identification

### Interactive Visualizations
- **Custom Plot Builder**: User-friendly interface for creating various chart types
- **Chart Types Supported**:
  - Scatter plots
  - Line charts
  - Bar charts
  - Histograms
  - Box plots
- **Plotly Integration**: Interactive, publication-quality charts
- **Color Coding**: Optional color-based grouping for enhanced insights

### Machine Learning Pipeline
- **Data Preprocessing**:
  - Missing value imputation (mean, median, mode, drop)
  - One-hot encoding for categorical variables
  - Feature scaling and normalization
  - Automatic feature type detection

- **Model Training**:
  - **Classification Algorithms**:
    - Logistic Regression
    - Random Forest
    - Decision Tree
    - Support Vector Machine (SVM)
  
  - **Regression Algorithms**:
    - Linear Regression
    - Random Forest Regressor
    - Decision Tree Regressor
    - Support Vector Regressor (SVR)

- **Model Evaluation**:
  - **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
  - **Regression Metrics**: R² Score, Mean Squared Error, Root Mean Squared Error
  - **Feature Importance**: Visual analysis of feature contributions
  - **Training Time Tracking**: Performance optimization insights

### Advanced Features
- **Session State Management**: Data persistence across user interactions
- **Caching**: Optimized performance for expensive operations
- **Error Handling**: Comprehensive error catching and user-friendly messages
- **Download Capabilities**: Export predictions and results as CSV
- **Modular Architecture**: Clean, maintainable code structure

## 🏗️ Project Structure

```
no-code-ml-studio/
├── app.py                 # Main Streamlit application
├── utils/                 # Utility modules
│   ├── __init__.py       # Package initialization
│   ├── data_loader.py    # Data ingestion and validation
│   ├── viz.py           # Visualization functions
│   └── model.py         # ML training and evaluation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 📦 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the local URL provided by Streamlit (typically `http://localhost:8501`)

## 🎯 Usage Guide

### Getting Started
1. **Upload Data**: Use the sidebar to upload CSV/Excel files or provide a URL
2. **Explore Data**: Navigate to the EDA tab to understand your dataset
3. **Visualize**: Create custom charts to gain insights
4. **Train Models**: Configure preprocessing and select algorithms
5. **Evaluate**: Review performance metrics and feature importance

### Best Practices
- **Data Quality**: Ensure your data is clean before uploading
- **Feature Selection**: Choose relevant features for better model performance
- **Missing Values**: Select appropriate imputation strategies
- **Model Selection**: Try multiple algorithms to find the best performer

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **OpenPyXL**: Excel file handling

### Performance Features
- **@st.cache_data**: Efficient caching of expensive operations
- **Session State**: Persistent data storage across interactions
- **Error Handling**: Robust exception management
- **Memory Optimization**: Efficient data processing

## 🤝 Contributing

This application is designed to be modular and extensible. Key areas for enhancement:

1. **Additional Algorithms**: Integrate more ML models
2. **Advanced Visualizations**: Add more chart types and customization
3. **Data Export**: Expand download capabilities
4. **Model Deployment**: Add model saving/loading features
5. **Performance Optimization**: Enhance caching strategies

## 📄 License

This project is open-source. Feel free to modify and distribute according to your needs.

## 🆘 Support

For issues or questions:
1. Check the error messages displayed in the app
2. Verify your data format and quality
3. Ensure all dependencies are properly installed
4. Review the code comments for implementation details

---

**Built with ❤️ using Streamlit | Empowering non-technical users to harness machine learning**