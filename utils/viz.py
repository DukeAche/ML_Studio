import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class Visualization:
    """Handles all data visualization using Plotly"""
    
    def create_plot(self, df, x_axis, y_axis, color_col="None", chart_type="Scatter"):
        """Create interactive Plotly chart based on user selections"""
        try:
            color_arg = None if color_col == "None" else color_col
            
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_arg,
                               title=f"{y_axis} vs {x_axis}")
            
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, color=color_arg,
                            title=f"{y_axis} over {x_axis}")
            
            elif chart_type == "Bar":
                if color_arg:
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_arg,
                               title=f"{y_axis} by {x_axis}")
                else:
                    # Aggregate for bar chart
                    agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
                    fig = px.bar(agg_df, x=x_axis, y=y_axis,
                               title=f"Average {y_axis} by {x_axis}")
            
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_axis, color=color_arg,
                                 title=f"Distribution of {x_axis}")
            
            elif chart_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis, color=color_arg,
                           title=f"{y_axis} distribution by {x_axis}")
            
            else:
                st.error(f"Unsupported chart type: {chart_type}")
                return None
            
            # Update layout for better appearance
            fig.update_layout(
                template="plotly_white",
                height=500,
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            return None
    
    def create_correlation_heatmap(self, df):
        """Create correlation heatmap for numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
                return None
            
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix,
                          text_auto=True,
                          aspect="auto",
                          color_continuous_scale="RdBu",
                          title="Correlation Matrix")
            
            fig.update_layout(height=600)
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def create_distribution_plots(self, df, column):
        """Create distribution plot for a single column"""
        try:
            if df[column].dtype in ['object', 'category']:
                # Categorical data
                value_counts = df[column].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribution of {column}")
            else:
                # Numeric data
                fig = px.histogram(df, x=column, title=f"Distribution of {column}")
            
            fig.update_layout(template="plotly_white", height=400)
            return fig
            
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")
            return None
    
    def create_pairplot(self, df, columns=None, color_col=None):
        """Create pairplot for selected columns"""
        try:
            if columns is None:
                # Use numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 5:  # Limit to avoid overcrowding
                    columns = numeric_cols[:5].tolist()
                else:
                    columns = numeric_cols.tolist()
            
            if len(columns) < 2:
                st.warning("Need at least 2 numeric columns for pairplot")
                return None
            
            fig = px.scatter_matrix(df[columns + ([color_col] if color_col else [])],
                                  dimensions=columns,
                                  color=color_col,
                                  title="Scatter Matrix")
            
            fig.update_layout(height=800)
            return fig
            
        except Exception as e:
            st.error(f"Error creating pairplot: {str(e)}")
            return None
    
    def create_time_series_plot(self, df, date_col, value_col, color_col=None):
        """Create time series plot"""
        try:
            # Ensure date column is datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            fig = px.line(df, x=date_col, y=value_col, color=color_col,
                        title=f"{value_col} over time")
            
            fig.update_layout(template="plotly_white", height=500)
            return fig
            
        except Exception as e:
            st.error(f"Error creating time series plot: {str(e)}")
            return None
    
    def create_3d_scatter(self, df, x_col, y_col, z_col, color_col=None):
        """Create 3D scatter plot"""
        try:
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                              title=f"3D Scatter: {x_col}, {y_col}, {z_col}")
            
            fig.update_layout(template="plotly_white", height=600)
            return fig
            
        except Exception as e:
            st.error(f"Error creating 3D scatter plot: {str(e)}")
            return None