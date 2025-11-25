import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import time

class ModelTrainer:
    """Handles machine learning model training and evaluation"""
    
    def __init__(self):
        self.models = {
            'Classification': {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'SVM': SVC(random_state=42)
            },
            'Regression': {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'SVM': SVR()
            }
        }
    
    def preprocess_data(self, df, feature_cols, target_col, missing_strategy='mean'):
        """Preprocess data for machine learning"""
        try:
            # Create a copy to avoid modifying original data
            df_copy = df.copy()
            
            # Separate features and target
            X = df_copy[feature_cols]
            y = df_copy[target_col]
            
            # Identify numeric and categorical columns
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create preprocessing pipelines
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=missing_strategy)),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            # Apply preprocessing
            X_processed = preprocessor.fit_transform(X)
            
            # Get feature names after preprocessing
            feature_names = []
            if numeric_features:
                feature_names.extend(numeric_features)
            if categorical_features:
                # Get feature names from one-hot encoding
                cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
                feature_names.extend(cat_features)
            
            # Handle target variable
            if y.dtype == 'object':
                # Encode categorical target
                label_encoder = LabelEncoder()
                y_processed = label_encoder.fit_transform(y)
                self.label_encoder = label_encoder
            else:
                y_processed = y.values
            
            # Create preprocessed DataFrame for display
            X_df = pd.DataFrame(X_processed, columns=feature_names)
            preprocessed_df = pd.concat([X_df, pd.Series(y_processed, name=target_col)], axis=1)
            
            return X_processed, y_processed, preprocessed_df
            
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            return None, None, None
    
    def train_model(self, X, y, task_type, algorithm, test_size=0.2):
        """Train machine learning model"""
        try:
            start_time = time.time()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Get the model
            model = self.models[task_type][algorithm]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            training_time = time.time() - start_time
            
            results = {
                'task_type': task_type,
                'algorithm': algorithm,
                'training_time': training_time,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'predictions': y_pred,
                'model': model,
                'model_params': model.get_params()
            }
            
            # Add task-specific metrics
            if task_type == 'Classification':
                results.update({
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                })
            else:  # Regression
                results.update({
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                })
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_
                results['feature_names'] = getattr(model, 'feature_names_in_', None)
            elif hasattr(model, 'coef_'):
                if task_type == 'Classification' and len(model.coef_.shape) > 1:
                    # For multi-class classification, use first class coefficients
                    results['feature_importance'] = np.abs(model.coef_[0])
                else:
                    results['feature_importance'] = np.abs(model.coef_)
                results['feature_names'] = getattr(model, 'feature_names_in_', None)
            
            return results
            
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            return None
    
    def get_model_comparison(self, results_list):
        """Compare multiple model results"""
        comparison_data = []
        
        for results in results_list:
            if results['task_type'] == 'Classification':
                comparison_data.append({
                    'Algorithm': results['algorithm'],
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'Training Time': results['training_time']
                })
            else:
                comparison_data.append({
                    'Algorithm': results['algorithm'],
                    'R² Score': results['r2_score'],
                    'MSE': results['mse'],
                    'RMSE': results['rmse'],
                    'Training Time': results['training_time']
                })
        
        return pd.DataFrame(comparison_data)
    
    def predict_new_data(self, model, X_new):
        """Make predictions on new data"""
        try:
            predictions = model.predict(X_new)
            return predictions
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None
