#!/usr/bin/env python3
"""
CSV Data Analysis with Visualizations
Generates charts, statistics, and data quality reports
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_FILE = "sample.csv"  # Change this to your CSV file path
OUTPUT_DIR = "analysis_output"

# ============================================================================
# DATA LOADING & INITIAL EXPLORATION
# ============================================================================

def load_data(filepath):
    """Load CSV and display basic info"""
    print(f"\n{'='*70}")
    print(f"📊 CSV Data Analysis Report")
    print(f"{'='*70}\n")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ File loaded: {filepath}")
        print(f"📈 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        return None

def display_overview(df):
    """Display dataset overview"""
    print(f"\n{'─'*70}")
    print("1️⃣  DATASET OVERVIEW")
    print(f"{'─'*70}\n")
    
    print("First few rows:")
    print(df.head().to_string())
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nColumn Names: {df.columns.tolist()}")

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

def data_quality_report(df):
    """Analyze data quality issues"""
    print(f"\n{'─'*70}")
    print("2️⃣  DATA QUALITY REPORT")
    print(f"{'─'*70}\n")
    
    quality_data = {
        'Column': [],
        'Type': [],
        'Non-Null': [],
        'Missing': [],
        'Missing %': [],
        'Unique': []
    }
    
    for col in df.columns:
        quality_data['Column'].append(col)
        quality_data['Type'].append(df[col].dtype)
        non_null = df[col].notna().sum()
        missing = df[col].isna().sum()
        quality_data['Non-Null'].append(non_null)
        quality_data['Missing'].append(missing)
        quality_data['Missing %'].append(f"{(missing/len(df)*100):.1f}%")
        quality_data['Unique'].append(df[col].nunique())
    
    quality_df = pd.DataFrame(quality_data)
    print(quality_df.to_string(index=False))
    
    # Validate email format
    print(f"\n Email Validation:")
    email_col = [col for col in df.columns if 'email' in col.lower()]
    if email_col:
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        valid_emails = df[email_col[0]].fillna('').apply(
            lambda x: bool(re.match(email_regex, str(x)))
        )
        print(f"  Valid emails: {valid_emails.sum()}/{len(df)}")
    
    return quality_df

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

def statistical_summary(df):
    """Generate statistical summary"""
    print(f"\n{'─'*70}")
    print("3️⃣  STATISTICAL SUMMARY")
    print(f"{'─'*70}\n")
    
    print(df.describe().to_string())
    print(f"\n{df.describe(include='object').to_string()}")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def setup_plots():
    """Configure plot style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 9
    sns.set_palette("husl")

def plot_numeric_distributions(df):
    """Plot distributions for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("⚠️  No numeric columns found for distribution plots")
        return
    
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4*n_cols))
    
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(numeric_cols):
        # Histogram
        axes[idx, 0].hist(df[col].dropna(), bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx, 0].set_title(f'Distribution: {col}', fontweight='bold')
        axes[idx, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[idx, 1].boxplot(df[col].dropna(), vert=True)
        axes[idx, 1].set_title(f'Box Plot: {col}', fontweight='bold')
        axes[idx, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_numeric_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 01_numeric_distributions.png")
    plt.close()

def plot_categorical_distributions(df):
    """Plot distributions for categorical columns"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("⚠️  No categorical columns found")
        return
    
    # Select top 6 columns (avoid overcrowding)
    categorical_cols = categorical_cols[:6]
    n_cols = len(categorical_cols)
    
    fig, axes = plt.subplots((n_cols + 1)//2, 2, figsize=(15, 4*((n_cols+1)//2)))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts().head(10)
        axes[idx].barh(range(len(value_counts)), value_counts.values, color='coral')
        axes[idx].set_yticks(range(len(value_counts)))
        axes[idx].set_yticklabels(value_counts.index, fontsize=9)
        axes[idx].set_title(f'{col} (Top 10)', fontweight='bold')
        axes[idx].set_xlabel('Count')
    
    # Hide extra subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_categorical_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 02_categorical_distributions.png")
    plt.close()

def plot_missing_data(df):
    """Visualize missing data"""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if missing_data.empty:
        print("⚠️  No missing values to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_data.plot(kind='barh', ax=ax, color='salmon')
    ax.set_title('Missing Data by Column', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count of Missing Values')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_missing_data.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 03_missing_data.png")
    plt.close()

def plot_correlations(df):
    """Plot correlation heatmap for numeric data"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("⚠️  Not enough numeric columns for correlation plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_correlations.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 04_correlations.png")
    plt.close()

def plot_country_analysis(df):
    """Analyze data by country/geographic columns"""
    country_col = [col for col in df.columns if 'country' in col.lower()]
    city_col = [col for col in df.columns if 'city' in col.lower()]
    
    if not country_col:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Country distribution
    country_counts = df[country_col[0]].value_counts()
    axes[0].pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Distribution by Country', fontweight='bold', fontsize=12)
    
    # City distribution (top 10)
    if city_col:
        city_counts = df[city_col[0]].value_counts().head(10)
        axes[1].barh(range(len(city_counts)), city_counts.values, color='lightgreen')
        axes[1].set_yticks(range(len(city_counts)))
        axes[1].set_yticklabels(city_counts.index, fontsize=9)
        axes[1].set_title('Top 10 Cities', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_geographic_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 05_geographic_analysis.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load data
    df = load_data(CSV_FILE)
    if df is None:
        return
    
    # Display overview
    display_overview(df)
    
    # Data quality report
    data_quality_report(df)
    
    # Statistical summary
    statistical_summary(df)
    
    # Generate visualizations
    print(f"\n{'─'*70}")
    print("4️⃣  GENERATING VISUALIZATIONS")
    print(f"{'─'*70}\n")
    
    setup_plots()
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)
    plot_missing_data(df)
    plot_correlations(df)
    plot_country_analysis(df)
    
    print(f"\n{'='*70}")
    print(f"✅ Analysis Complete! Charts saved to '{OUTPUT_DIR}/' folder")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
