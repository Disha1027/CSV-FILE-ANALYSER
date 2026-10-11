import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from scipy.stats import probplot


class AdvancedVisualizer:
    """
    A class for creating advanced data visualizations.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the AdvancedVisualizer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def create_ecdf_plot(self, column):
        """
        Create an Empirical Cumulative Distribution Function (ECDF) plot.
        
        Args:
            column (str): Column name to plot
            
        Returns:
            matplotlib.figure.Figure: The ECDF plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        data = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(data):
            return None
        
        # Calculate ECDF
        sorted_data = np.sort(data)
        n = len(sorted_data)
        ecdf = np.arange(1, n + 1) / n
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_data, ecdf, marker='.', linestyle='none')
        ax.set_xlabel(column)
        ax.set_ylabel('ECDF')
        ax.set_title(f'Empirical Cumulative Distribution Function - {column}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_qq_plot(self, column):
        """
        Create a Q-Q plot to check normality.
        
        Args:
            column (str): Column name to plot
            
        Returns:
            matplotlib.figure.Figure: The Q-Q plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        data = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(data):
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        probplot(data, dist='norm', plot=ax)
        ax.set_title(f'Q-Q Plot - {column}')
        
        return fig
    
    def create_violin_plot(self, column):
        """
        Create a violin plot for a numeric column.
        
        Args:
            column (str): Column name to plot
            
        Returns:
            matplotlib.figure.Figure: The violin plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        data = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(data):
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(y=data, ax=ax)
        ax.set_title(f'Violin Plot - {column}')
        
        return fig
    
    def create_boxen_plot(self, column):
        """
        Create a boxen plot (letter-value plot) for better visualization of distributions.
        
        Args:
            column (str): Column name to plot
            
        Returns:
            matplotlib.figure.Figure: The boxen plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        data = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(data):
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxenplot(y=data, ax=ax)
        ax.set_title(f'Boxen Plot - {column}')
        
        return fig
    
    def create_pareto_chart(self, column):
        """
        Create a Pareto chart for categorical data.
        
        Args:
            column (str): Column name to plot
            
        Returns:
            matplotlib.figure.Figure: The Pareto chart figure
        """
        df = self.data_loader.get_data()
        
        if df is None or column not in df.columns:
            return None
        
        value_counts = df[column].value_counts().sort_values(ascending=False)
        
        # Limit to top 20
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar plot
        ax1.bar(range(len(value_counts)), value_counts.values, color='steelblue')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Cumulative percentage line
        ax2 = ax1.twinx()
        cumulative = value_counts.cumsum() / value_counts.sum() * 100
        ax2.plot(range(len(value_counts)), cumulative, color='red', marker='o', linewidth=2)
        ax2.set_ylabel('Cumulative Percentage', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f'Pareto Chart - {column}')
        plt.tight_layout()
        
        return fig
    
    def create_stacked_bar_chart(self, cat_col1, cat_col2):
        """
        Create a stacked bar chart for two categorical columns.
        
        Args:
            cat_col1 (str): First categorical column
            cat_col2 (str): Second categorical column
            
        Returns:
            matplotlib.figure.Figure: The stacked bar chart figure
        """
        df = self.data_loader.get_data()
        
        if df is None or cat_col1 not in df.columns or cat_col2 not in df.columns:
            return None
        
        # Create contingency table
        contingency = pd.crosstab(df[cat_col1], df[cat_col2])
        
        # Limit if too many categories
        if contingency.shape[0] > 15:
            contingency = contingency.head(15)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        contingency.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Stacked Bar Chart - {cat_col1} vs {cat_col2}')
        ax.set_xlabel(cat_col1)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    def create_100_stacked_bar_chart(self, cat_col1, cat_col2):
        """
        Create a 100% stacked bar chart for two categorical columns.
        
        Args:
            cat_col1 (str): First categorical column
            cat_col2 (str): Second categorical column
            
        Returns:
            matplotlib.figure.Figure: The 100% stacked bar chart figure
        """
        df = self.data_loader.get_data()
        
        if df is None or cat_col1 not in df.columns or cat_col2 not in df.columns:
            return None
        
        # Create contingency table
        contingency = pd.crosstab(df[cat_col1], df[cat_col2], normalize='index') * 100
        
        # Limit if too many categories
        if contingency.shape[0] > 15:
            contingency = contingency.head(15)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        contingency.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'100% Stacked Bar Chart - {cat_col1} vs {cat_col2}')
        ax.set_xlabel(cat_col1)
        ax.set_ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    def create_hexbin_plot(self, x_col, y_col, gridsize=20):
        """
        Create a hexbin plot for dense datasets.
        
        Args:
            x_col (str): X-axis column
            y_col (str): Y-axis column
            gridsize (int): Size of the hexagon grid
            
        Returns:
            matplotlib.figure.Figure: The hexbin plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or x_col not in df.columns or y_col not in df.columns:
            return None
        
        clean_df = df[[x_col, y_col]].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        hb = ax.hexbin(clean_df[x_col], clean_df[y_col], gridsize=gridsize, cmap='viridis')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Hexbin Plot - {x_col} vs {y_col}')
        plt.colorbar(hb, label='Count')
        
        return fig
    
    def create_density_2d_plot(self, x_col, y_col):
        """
        Create a 2D density plot.
        
        Args:
            x_col (str): X-axis column
            y_col (str): Y-axis column
            
        Returns:
            matplotlib.figure.Figure: The 2D density plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or x_col not in df.columns or y_col not in df.columns:
            return None
        
        clean_df = df[[x_col, y_col]].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.kdeplot(data=clean_df, x=x_col, y=y_col, cmap='viridis', fill=True, ax=ax)
        ax.set_title(f'2D Density Plot - {x_col} vs {y_col}')
        
        return fig
    
    def create_missingness_heatmap(self):
        """
        Create a heatmap showing missing values pattern.
        
        Returns:
            matplotlib.figure.Figure: The missingness heatmap figure
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        # Create missing indicator matrix
        missing_matrix = df.isnull()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(missing_matrix, cbar=False, yticklabels=False, cmap='viridis', ax=ax)
        ax.set_title('Missing Values Heatmap')
        ax.set_xlabel('Columns')
        
        return fig
    
    def create_missingness_bar_chart(self):
        """
        Create a bar chart showing missing values percentage by column.
        
        Returns:
            matplotlib.figure.Figure: The missingness bar chart figure
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        
        if len(missing_pct) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_pct.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Missing Values Percentage by Column')
        ax.set_ylabel('Missing Percentage')
        ax.set_xlabel('Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def create_parallel_coordinates_plot(self, columns, hue=None):
        """
        Create a parallel coordinates plot.
        
        Args:
            columns (list): Columns to include
            hue (str, optional): Column to use for color encoding
            
        Returns:
            matplotlib.figure.Figure: The parallel coordinates plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        # Ensure columns exist
        columns = [col for col in columns if col in df.columns]
        
        if len(columns) < 3:
            return None
        
        # Sample if too many rows
        if len(df) > 1000:
            plot_df = df.sample(1000, random_state=42)
        else:
            plot_df = df.copy()
        
        # Normalize data
        numeric = plot_df[columns].apply(pd.to_numeric, errors="coerce")
        minima, maxima = numeric.min(), numeric.max()
        ranges = (maxima - minima).replace(0, 1)
        plot_df[columns] = (numeric - minima) / ranges
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if hue and hue in df.columns:
            pd.plotting.parallel_coordinates(plot_df, hue, color=plt.cm.tab10(np.linspace(0, 1, plot_df[hue].nunique())), alpha=0.5, ax=ax)
        else:
            plot_df["__plot_group"] = "All rows"
            pd.plotting.parallel_coordinates(plot_df[columns + ["__plot_group"]], "__plot_group", color=["#55bdb4"], alpha=0.5, ax=ax)
        
        ax.set_title('Parallel Coordinates Plot')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_residual_plot(self, x_col, y_col):
        """
        Create a residual plot for regression analysis.
        
        Args:
            x_col (str): X-axis column (predictor)
            y_col (str): Y-axis column (response)
            
        Returns:
            matplotlib.figure.Figure: The residual plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or x_col not in df.columns or y_col not in df.columns:
            return None
        
        clean_df = df[[x_col, y_col]].dropna()
        
        # Fit simple least-squares regression without scikit-learn.
        x = pd.to_numeric(clean_df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(clean_df[y_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 2 or np.ptp(x) == 0:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        residuals = y - y_pred

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residual Plot - {y_col} vs {x_col}')
        
        return fig
    
    def create_grouped_box_plot(self, numeric_col, cat_col):
        """
        Create a grouped box plot for numeric vs categorical analysis.
        
        Args:
            numeric_col (str): Numeric column
            cat_col (str): Categorical column
            
        Returns:
            matplotlib.figure.Figure: The grouped box plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or numeric_col not in df.columns or cat_col not in df.columns:
            return None
        
        # Limit categories if too many
        unique_cats = df[cat_col].unique()
        if len(unique_cats) > 20:
            top_cats = df[cat_col].value_counts().head(20).index
            plot_df = df[df[cat_col].isin(top_cats)]
        else:
            plot_df = df
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=plot_df, x=cat_col, y=numeric_col, ax=ax)
        ax.set_title(f'Box Plot of {numeric_col} by {cat_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def create_grouped_violin_plot(self, numeric_col, cat_col):
        """
        Create a grouped violin plot for numeric vs categorical analysis.
        
        Args:
            numeric_col (str): Numeric column
            cat_col (str): Categorical column
            
        Returns:
            matplotlib.figure.Figure: The grouped violin plot figure
        """
        df = self.data_loader.get_data()
        
        if df is None or numeric_col not in df.columns or cat_col not in df.columns:
            return None
        
        # Limit categories if too many
        unique_cats = df[cat_col].unique()
        if len(unique_cats) > 15:
            top_cats = df[cat_col].value_counts().head(15).index
            plot_df = df[df[cat_col].isin(top_cats)]
        else:
            plot_df = df
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=plot_df, x=cat_col, y=numeric_col, ax=ax)
        ax.set_title(f'Violin Plot of {numeric_col} by {cat_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
