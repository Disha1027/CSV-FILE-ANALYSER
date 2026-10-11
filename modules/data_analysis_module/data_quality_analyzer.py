import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter


class DataQualityAnalyzer:
    """
    A class for comprehensive data quality analysis and health reporting.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the DataQualityAnalyzer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def get_quality_report(self):
        """
        Generate a comprehensive data quality report.
        
        Returns:
            dict: Dictionary containing quality issues and recommendations
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        issues = []
        
        # Check for missing values
        missing_issues = self._check_missing_values(df)
        issues.extend(missing_issues)
        
        # Check for duplicates
        duplicate_issues = self._check_duplicates(df)
        issues.extend(duplicate_issues)
        
        # Check for duplicate column names
        duplicate_col_issues = self._check_duplicate_columns(df)
        issues.extend(duplicate_col_issues)
        
        # Check for mixed data types
        mixed_type_issues = self._check_mixed_types(df)
        issues.extend(mixed_type_issues)
        
        # Check for constant columns
        constant_col_issues = self._check_constant_columns(df)
        issues.extend(constant_col_issues)
        
        # Check for high cardinality columns
        high_cardinality_issues = self._check_high_cardinality(df)
        issues.extend(high_cardinality_issues)
        
        # Check for identifier-like columns
        identifier_issues = self._check_identifier_columns(df)
        issues.extend(identifier_issues)
        
        # Check for whitespace issues
        whitespace_issues = self._check_whitespace(df)
        issues.extend(whitespace_issues)
        
        # Check for suspicious numeric ranges
        numeric_range_issues = self._check_numeric_ranges(df)
        issues.extend(numeric_range_issues)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(df, issues)
        
        return {
            'issues': issues,
            'health_score': health_score,
            'total_issues': len(issues),
            'severity_breakdown': self._get_severity_breakdown(issues)
        }
    
    def _check_missing_values(self, df):
        """Check for columns with high missingness."""
        issues = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_pct > 50:
                severity = 'critical'
            elif missing_pct > 20:
                severity = 'high'
            elif missing_pct > 5:
                severity = 'medium'
            elif missing_pct > 0:
                severity = 'low'
            else:
                continue
            
            issues.append({
                'type': 'Missing Values',
                'column': col,
                'severity': severity,
                'affected_rows': missing_count,
                'affected_percentage': missing_pct,
                'description': f"Column '{col}' has {missing_pct:.2f}% missing values",
                'recommendation': self._get_missing_recommendation(missing_pct)
            })
        
        return issues
    
    def _check_duplicates(self, df):
        """Check for duplicate rows."""
        issues = []
        
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / len(df)) * 100
        
        if duplicate_pct > 10:
            severity = 'high'
        elif duplicate_pct > 1:
            severity = 'medium'
        elif duplicate_pct > 0:
            severity = 'low'
        else:
            return issues
        
        issues.append({
            'type': 'Duplicate Rows',
            'column': 'All columns',
            'severity': severity,
            'affected_rows': duplicate_rows,
            'affected_percentage': duplicate_pct,
            'description': f"Dataset contains {duplicate_rows} duplicate rows ({duplicate_pct:.2f}%)",
            'recommendation': 'Review and remove duplicate rows if they represent data entry errors'
        })
        
        return issues
    
    def _check_duplicate_columns(self, df):
        """Check for duplicate column names."""
        issues = []
        
        col_counts = Counter(df.columns)
        duplicate_cols = [col for col, count in col_counts.items() if count > 1]
        
        if duplicate_cols:
            issues.append({
                'type': 'Duplicate Column Names',
                'column': ', '.join(duplicate_cols),
                'severity': 'high',
                'affected_rows': len(df),
                'affected_percentage': 100,
                'description': f"Duplicate column names found: {', '.join(duplicate_cols)}",
                'recommendation': 'Rename duplicate columns to ensure unique column names'
            })
        
        return issues
    
    def _check_mixed_types(self, df):
        """Check for columns with mixed data types."""
        issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                types = df[col].dropna().apply(type).nunique()
                if types > 1:
                    issues.append({
                        'type': 'Mixed Data Types',
                        'column': col,
                        'severity': 'medium',
                        'affected_rows': df[col].count(),
                        'affected_percentage': (df[col].count() / len(df)) * 100,
                        'description': f"Column '{col}' contains mixed data types ({types} different types)",
                        'recommendation': 'Consider standardizing the data type for this column'
                    })
        
        return issues
    
    def _check_constant_columns(self, df):
        """Check for constant or near-constant columns."""
        issues = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            
            if unique_count == 1:
                issues.append({
                    'type': 'Constant Column',
                    'column': col,
                    'severity': 'low',
                    'affected_rows': len(df),
                    'affected_percentage': 100,
                    'description': f"Column '{col}' has only one unique value (constant)",
                    'recommendation': 'Consider removing this column as it provides no information'
                })
            elif unique_count == 2 and df[col].dtype == 'object':
                # Could be a binary categorical
                pass
    
        return issues
    
    def _check_high_cardinality(self, df):
        """Check for columns with very high cardinality."""
        issues = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            cardinality_ratio = unique_count / len(df)
            
            # High cardinality: more than 95% unique values and more than 100 unique values
            if cardinality_ratio > 0.95 and unique_count > 100:
                issues.append({
                    'type': 'High Cardinality',
                    'column': col,
                    'severity': 'low',
                    'affected_rows': len(df),
                    'affected_percentage': 100,
                    'description': f"Column '{col}' has very high cardinality ({unique_count} unique values, {cardinality_ratio:.1%})",
                    'recommendation': 'This may be an identifier column; consider excluding from analysis'
                })
        
        return issues
    
    def _check_identifier_columns(self, df):
        """Check for identifier-like columns."""
        issues = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            cardinality_ratio = unique_count / len(df)
            
            # Identifier-like: very high cardinality, likely IDs
            if cardinality_ratio > 0.9 and unique_count > 10:
                issues.append({
                    'type': 'Identifier Column',
                    'column': col,
                    'severity': 'info',
                    'affected_rows': len(df),
                    'affected_percentage': 100,
                    'description': f"Column '{col}' appears to be an identifier column",
                    'recommendation': 'Consider excluding from statistical analysis and modeling'
                })
        
        return issues
    
    def _check_whitespace(self, df):
        """Check for whitespace issues in string columns."""
        issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for leading/trailing whitespace
                str_series = df[col].astype(str)
                has_leading = str_series.str.startswith(' ').any()
                has_trailing = str_series.str.endswith(' ').any()
                
                if has_leading or has_trailing:
                    issues.append({
                        'type': 'Whitespace Issues',
                        'column': col,
                        'severity': 'low',
                        'affected_rows': df[col].count(),
                        'affected_percentage': (df[col].count() / len(df)) * 100,
                        'description': f"Column '{col}' contains leading or trailing whitespace",
                        'recommendation': 'Consider stripping whitespace from this column'
                    })
        
        return issues
    
    def _check_numeric_ranges(self, df):
        """Check for suspicious numeric ranges."""
        issues = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            valid_data = df[col].dropna()
            
            if len(valid_data) == 0:
                continue
            
            # Check for extreme outliers using IQR method
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            extreme_outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
            
            if len(extreme_outliers) > 0:
                outlier_pct = (len(extreme_outliers) / len(valid_data)) * 100
                if outlier_pct > 5:  # More than 5% extreme outliers
                    issues.append({
                        'type': 'Suspicious Numeric Range',
                        'column': col,
                        'severity': 'medium',
                        'affected_rows': len(extreme_outliers),
                        'affected_percentage': outlier_pct,
                        'description': f"Column '{col}' has {outlier_pct:.2f}% extreme outliers",
                        'recommendation': 'Review these values for potential data entry errors'
                    })
        
        return issues
    
    def _get_missing_recommendation(self, missing_pct):
        """Get recommendation based on missing percentage."""
        if missing_pct > 50:
            return 'Consider removing this column or collecting more data'
        elif missing_pct > 20:
            return 'Consider imputation or investigate why values are missing'
        elif missing_pct > 5:
            return 'Consider imputation or flag missing values'
        else:
            return 'Minor missingness; imputation or deletion may be acceptable'
    
    def _calculate_health_score(self, df, issues):
        """Calculate an overall data health score (0-100)."""
        if len(issues) == 0:
            return 100
        
        # Weight issues by severity
        severity_weights = {
            'critical': 25,
            'high': 15,
            'medium': 8,
            'low': 3,
            'info': 1
        }
        
        total_penalty = sum(severity_weights.get(issue['severity'], 5) for issue in issues)
        score = max(0, 100 - total_penalty)
        
        return score
    
    def _get_severity_breakdown(self, issues):
        """Get breakdown of issues by severity."""
        breakdown = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        
        for issue in issues:
            severity = issue.get('severity', 'low')
            breakdown[severity] = breakdown.get(severity, 0) + 1
        
        return breakdown
    
    def get_missingness_analysis(self):
        """
        Perform detailed missingness analysis.
        
        Returns:
            dict: Dictionary containing missingness analysis results
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        # Missingness summary
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df)) * 100
        
        # Columns with missing values
        missing_cols = missing_summary[missing_summary > 0]
        
        # Rows with missing values
        rows_with_missing = df.isnull().any(axis=1).sum()
        
        # Missingness co-occurrence (which columns tend to be missing together)
        missing_patterns = df.isnull().astype(int)
        co_occurrence = missing_patterns.T.corr()
        
        return {
            'missing_summary': pd.DataFrame({
                'Column': missing_summary.index,
                'Missing Count': missing_summary.values,
                'Missing Percentage': missing_pct.values
            }).sort_values('Missing Count', ascending=False),
            'missing_columns': missing_cols.index.tolist(),
            'rows_with_missing': rows_with_missing,
            'rows_with_missing_pct': (rows_with_missing / len(df)) * 100,
            'complete_rows': len(df) - rows_with_missing,
            'co_occurrence_matrix': co_occurrence
        }
    
    def get_duplicate_analysis(self, subset=None):
        """
        Perform detailed duplicate analysis.
        
        Args:
            subset (list, optional): Columns to consider for duplicate detection
            
        Returns:
            dict: Dictionary containing duplicate analysis results
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        if subset:
            # Check if subset columns exist
            subset = [col for col in subset if col in df.columns]
            if not subset:
                subset = None
        
        # Find duplicates
        duplicates = df.duplicated(subset=subset, keep=False)
        duplicate_count = duplicates.sum()
        
        # Get duplicate rows
        duplicate_rows = df[duplicates]
        
        # Column-wise duplicate values
        column_duplicates = {}
        for col in df.columns:
            col_duplicates = df[col].duplicated().sum()
            column_duplicates[col] = {
                'count': col_duplicates,
                'percentage': (col_duplicates / len(df)) * 100
            }
        
        return {
            'total_duplicates': duplicate_count,
            'duplicate_percentage': (duplicate_count / len(df)) * 100,
            'duplicate_rows': duplicate_rows,
            'column_duplicates': column_duplicates,
            'subset_used': subset
        }
