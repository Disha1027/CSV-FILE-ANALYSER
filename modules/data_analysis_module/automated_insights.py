import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st


class AutomatedInsights:
    """
    A class for generating automated EDA insights and reports.
    """
    
    def __init__(self, data_loader, data_profiler, data_quality_analyzer, 
                 advanced_statistical_analyzer):
        """
        Initialize the AutomatedInsights class.
        
        Args:
            data_loader: An instance of the DataLoader class
            data_profiler: An instance of the DataProfiler class
            data_quality_analyzer: An instance of the DataQualityAnalyzer class
            advanced_statistical_analyzer: An instance of the AdvancedStatisticalAnalyzer class
        """
        self.data_loader = data_loader
        self.data_profiler = data_profiler
        self.data_quality_analyzer = data_quality_analyzer
        self.advanced_statistical_analyzer = advanced_statistical_analyzer
    
    def generate_eda_report(self):
        """
        Generate a comprehensive automated EDA report.
        
        Returns:
            dict: Dictionary containing the EDA report with insights
        """
        df = self.data_loader.get_data()
        
        if df is None:
            return None
        
        report = {
            'dataset_overview': self._get_dataset_overview(),
            'schema_analysis': self._get_schema_analysis(),
            'data_quality_summary': self._get_data_quality_summary(),
            'missing_value_insights': self._get_missing_value_insights(),
            'duplicate_insights': self._get_duplicate_insights(),
            'numeric_distribution_insights': self._get_numeric_distribution_insights(),
            'categorical_insights': self._get_categorical_insights(),
            'correlation_insights': self._get_correlation_insights(),
            'outlier_insights': self._get_outlier_insights(),
            'feature_insights': self._get_feature_insights(),
            'recommendations': self._get_recommendations()
        }
        
        return report
    
    def _get_dataset_overview(self):
        """Get dataset overview insights."""
        profile = self.data_profiler.get_dataset_profile()
        
        if profile is None:
            return []
        
        insights = []
        
        insights.append({
            'type': 'overview',
            'message': f"Dataset contains {profile['rows']:,} rows and {profile['columns']} columns.",
            'severity': 'info'
        })
        
        insights.append({
            'type': 'overview',
            'message': f"Memory usage: {profile['memory_usage']}.",
            'severity': 'info'
        })
        
        if profile['missing_percentage'] > 0:
            insights.append({
                'type': 'overview',
                'message': f"Overall missingness: {profile['missing_percentage']:.2f}% ({profile['missing_cells']:,} cells).",
                'severity': 'warning' if profile['missing_percentage'] > 5 else 'info'
            })
        
        if profile['duplicate_percentage'] > 0:
            insights.append({
                'type': 'overview',
                'message': f"Duplicate rows: {profile['duplicate_percentage']:.2f}% ({profile['duplicate_rows']:,} rows).",
                'severity': 'warning' if profile['duplicate_percentage'] > 1 else 'info'
            })
        
        return insights
    
    def _get_schema_analysis(self):
        """Get schema and column type insights."""
        profile = self.data_profiler.get_dataset_profile()
        
        if profile is None:
            return []
        
        insights = []
        
        insights.append({
            'type': 'schema',
            'message': f"Column composition: {profile['numeric_count']} numeric, {profile['categorical_count']} categorical, {profile['datetime_count']} datetime, {profile['boolean_count']} boolean.",
            'severity': 'info'
        })
        
        if profile['constant_columns']:
            insights.append({
                'type': 'schema',
                'message': f"Constant columns found (no variation): {', '.join(profile['constant_columns'][:5])}.",
                'severity': 'warning'
            })
        
        if profile['mixed_type_columns']:
            insights.append({
                'type': 'schema',
                'message': f"Columns with mixed data types: {', '.join(profile['mixed_type_columns'][:5])}.",
                'severity': 'warning'
            })
        
        if profile['identifier_columns']:
            insights.append({
                'type': 'schema',
                'message': f"Potential identifier columns detected: {', '.join(profile['identifier_columns'][:5])}.",
                'severity': 'info'
            })
        
        return insights
    
    def _get_data_quality_summary(self):
        """Get data quality summary insights."""
        quality_report = self.data_quality_analyzer.get_quality_report()
        
        if quality_report is None:
            return []
        
        insights = []
        
        health_score = quality_report.get('health_score', 0)
        
        if health_score >= 90:
            insights.append({
                'type': 'quality',
                'message': f"Overall data health score: {health_score}/100 (Excellent).",
                'severity': 'success'
            })
        elif health_score >= 70:
            insights.append({
                'type': 'quality',
                'message': f"Overall data health score: {health_score}/100 (Good).",
                'severity': 'info'
            })
        elif health_score >= 50:
            insights.append({
                'type': 'quality',
                'message': f"Overall data health score: {health_score}/100 (Fair).",
                'severity': 'warning'
            })
        else:
            insights.append({
                'type': 'quality',
                'message': f"Overall data health score: {health_score}/100 (Poor).",
                'severity': 'error'
            })
        
        severity_breakdown = quality_report.get('severity_breakdown', {})
        total_issues = quality_report.get('total_issues', 0)
        
        if total_issues > 0:
            insights.append({
                'type': 'quality',
                'message': f"Total quality issues detected: {total_issues} (Critical: {severity_breakdown.get('critical', 0)}, High: {severity_breakdown.get('high', 0)}, Medium: {severity_breakdown.get('medium', 0)}, Low: {severity_breakdown.get('low', 0)}).",
                'severity': 'warning'
            })
        
        return insights
    
    def _get_missing_value_insights(self):
        """Get missing value insights."""
        missingness = self.data_quality_analyzer.get_missingness_analysis()
        
        if missingness is None:
            return []
        
        insights = []
        
        missing_summary = missingness.get('missing_summary')
        
        if missing_summary is not None and len(missing_summary) > 0:
            # Get columns with high missingness
            high_missing = missing_summary[missing_summary['Missing Percentage'] > 20]
            
            if len(high_missing) > 0:
                for _, row in high_missing.head(5).iterrows():
                    insights.append({
                        'type': 'missing',
                        'message': f"Column '{row['Column']}' has high missingness: {row['Missing Percentage']:.2f}%.",
                        'severity': 'error' if row['Missing Percentage'] > 50 else 'warning'
                    })
            
            # Get columns with moderate missingness
            moderate_missing = missing_summary[(missing_summary['Missing Percentage'] > 5) & (missing_summary['Missing Percentage'] <= 20)]
            
            if len(moderate_missing) > 0:
                insights.append({
                    'type': 'missing',
                    'message': f"Columns with moderate missingness (5-20%): {', '.join(moderate_missing['Column'].head(5).tolist())}.",
                    'severity': 'info'
                })
        
        rows_with_missing_pct = missingness.get('rows_with_missing_pct', 0)
        if rows_with_missing_pct > 50:
            insights.append({
                'type': 'missing',
                'message': f"{rows_with_missing_pct:.2f}% of rows contain at least one missing value.",
                'severity': 'warning'
            })
        
        return insights
    
    def _get_duplicate_insights(self):
        """Get duplicate insights."""
        duplicate_analysis = self.data_quality_analyzer.get_duplicate_analysis()
        
        if duplicate_analysis is None:
            return []
        
        insights = []
        
        duplicate_pct = duplicate_analysis.get('duplicate_percentage', 0)
        
        if duplicate_pct > 0:
            insights.append({
                'type': 'duplicate',
                'message': f"Found {duplicate_analysis['total_duplicates']:,} duplicate rows ({duplicate_pct:.2f}% of dataset).",
                'severity': 'warning' if duplicate_pct > 1 else 'info'
            })
        
        # Check for column-wise duplicates
        column_duplicates = duplicate_analysis.get('column_duplicates', {})
        high_dup_cols = [col for col, info in column_duplicates.items() if info['percentage'] > 10]
        
        if high_dup_cols:
            insights.append({
                'type': 'duplicate',
                'message': f"Columns with many duplicate values (>10%): {', '.join(high_dup_cols[:5])}.",
                'severity': 'info'
            })
        
        return insights
    
    def _get_numeric_distribution_insights(self):
        """Get numeric distribution insights."""
        df = self.data_loader.get_data()
        
        if df is None:
            return []
        
        insights = []
        numeric_cols = self.data_loader.get_numeric_columns()
        
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Check skewness
            skewness = col_data.skew()
            
            if abs(skewness) > 2:
                direction = 'right' if skewness > 0 else 'left'
                insights.append({
                    'type': 'distribution',
                    'message': f"Column '{col}' is highly {direction}-skewed (skewness: {skewness:.2f}).",
                    'severity': 'info'
                })
            elif abs(skewness) > 1:
                direction = 'right' if skewness > 0 else 'left'
                insights.append({
                    'type': 'distribution',
                    'message': f"Column '{col}' is moderately {direction}-skewed (skewness: {skewness:.2f}).",
                    'severity': 'info'
                })
            
            # Check for near-zero variance
            if col_data.std() < col_data.mean() * 0.01 if col_data.mean() != 0 else False:
                insights.append({
                    'type': 'distribution',
                    'message': f"Column '{col}' has very low variance relative to its mean.",
                    'severity': 'warning'
                })
        
        return insights
    
    def _get_categorical_insights(self):
        """Get categorical insights."""
        df = self.data_loader.get_data()
        
        if df is None:
            return []
        
        insights = []
        categorical_cols = self.data_loader.get_categorical_columns()
        
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            col_data = df[col]
            value_counts = col_data.value_counts()
            
            if len(value_counts) == 0:
                continue
            
            # Check for dominant category
            max_pct = (value_counts.iloc[0] / len(col_data)) * 100
            
            if max_pct > 80:
                insights.append({
                    'type': 'categorical',
                    'message': f"Column '{col}' is dominated by a single category: '{value_counts.index[0]}' ({max_pct:.2f}%).",
                    'severity': 'info'
                })
            
            # Check for high cardinality
            cardinality_ratio = col_data.nunique() / len(col_data)
            if cardinality_ratio > 0.5 and col_data.nunique() > 50:
                insights.append({
                    'type': 'categorical',
                    'message': f"Column '{col}' has high cardinality: {col_data.nunique()} unique values ({cardinality_ratio:.1%}).",
                    'severity': 'info'
                })
        
        return insights
    
    def _get_correlation_insights(self):
        """Get correlation insights."""
        numeric_cols = self.data_loader.get_numeric_columns()
        
        if len(numeric_cols) < 2:
            return []
        
        insights = []
        
        # Get correlation matrix
        corr_matrix = self.advanced_statistical_analyzer.get_correlation_matrix(method='pearson')
        
        if corr_matrix is None:
            return insights
        
        # Find strong correlations
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        for corr in strong_correlations[:5]:
            direction = 'positive' if corr['correlation'] > 0 else 'negative'
            insights.append({
                'type': 'correlation',
                'message': f"Strong {direction} correlation ({corr['correlation']:.2f}) between '{corr['col1']}' and '{corr['col2']}'.",
                'severity': 'info'
            })
        
        return insights
    
    def _get_outlier_insights(self):
        """Get outlier insights."""
        df = self.data_loader.get_data()
        
        if df is None:
            return []
        
        insights = []
        numeric_cols = self.data_loader.get_numeric_columns()
        
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            univariate_analysis = self.advanced_statistical_analyzer.get_univariate_analysis(col)
            
            if univariate_analysis is None:
                continue
            
            outliers_iqr = univariate_analysis.get('outliers', {}).get('iqr', {})
            outlier_pct = outliers_iqr.get('percentage', 0)
            
            if outlier_pct > 5:
                insights.append({
                    'type': 'outlier',
                    'message': f"Column '{col}' has many outliers ({outlier_pct:.2f}% using IQR method).",
                    'severity': 'warning'
                })
            elif outlier_pct > 1:
                insights.append({
                    'type': 'outlier',
                    'message': f"Column '{col}' has some outliers ({outlier_pct:.2f}% using IQR method).",
                    'severity': 'info'
                })
        
        return insights
    
    def _get_feature_insights(self):
        """Get feature-level insights."""
        profile = self.data_profiler.get_dataset_profile()
        
        if profile is None:
            return []
        
        insights = []
        
        # Check for potential redundant features
        if profile['constant_columns']:
            insights.append({
                'type': 'feature',
                'message': f"Consider removing constant columns: {', '.join(profile['constant_columns'][:5])}.",
                'severity': 'warning'
            })
        
        # Check for potential ID columns
        if profile['identifier_columns']:
            insights.append({
                'type': 'feature',
                'message': f"Consider excluding identifier columns from analysis: {', '.join(profile['identifier_columns'][:5])}.",
                'severity': 'info'
            })
        
        return insights
    
    def _get_recommendations(self):
        """Get analysis recommendations."""
        df = self.data_loader.get_data()
        
        if df is None:
            return []
        
        recommendations = []
        
        numeric_cols = self.data_loader.get_numeric_columns()
        categorical_cols = self.data_loader.get_categorical_columns()
        
        # General recommendations
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'recommendation',
                'message': "Consider performing correlation analysis to identify relationships between numeric variables.",
                'action': 'correlation_analysis'
            })
        
        if len(categorical_cols) >= 2:
            recommendations.append({
                'type': 'recommendation',
                'message': "Consider analyzing categorical relationships using contingency tables and chi-square tests.",
                'action': 'categorical_analysis'
            })
        
        if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
            recommendations.append({
                'type': 'recommendation',
                'message': "Consider performing group comparisons between categorical and numeric variables.",
                'action': 'group_comparison'
            })
        
        # Check for datetime columns
        datetime_cols = self.data_profiler.get_dataset_profile().get('datetime_columns', [])
        if datetime_cols:
            recommendations.append({
                'type': 'recommendation',
                'message': f"Datetime columns detected: {', '.join(datetime_cols[:3])}. Consider time-series analysis.",
                'action': 'time_series_analysis'
            })
        
        return recommendations
    
    def format_report_for_display(self, report):
        """
        Format the EDA report for display in Streamlit.
        
        Args:
            report (dict): The EDA report dictionary
            
        Returns:
            str: Formatted markdown string
        """
        if report is None:
            return "No report generated."
        
        markdown = "# Automated EDA Report\n\n"
        
        # Dataset Overview
        if report.get('dataset_overview'):
            markdown += "## Dataset Overview\n\n"
            for insight in report['dataset_overview']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Schema Analysis
        if report.get('schema_analysis'):
            markdown += "## Schema Analysis\n\n"
            for insight in report['schema_analysis']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Data Quality
        if report.get('data_quality_summary'):
            markdown += "## Data Quality Summary\n\n"
            for insight in report['data_quality_summary']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Missing Values
        if report.get('missing_value_insights'):
            markdown += "## Missing Value Analysis\n\n"
            for insight in report['missing_value_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Duplicates
        if report.get('duplicate_insights'):
            markdown += "## Duplicate Analysis\n\n"
            for insight in report['duplicate_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Numeric Distributions
        if report.get('numeric_distribution_insights'):
            markdown += "## Numeric Distribution Insights\n\n"
            for insight in report['numeric_distribution_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Categorical Insights
        if report.get('categorical_insights'):
            markdown += "## Categorical Insights\n\n"
            for insight in report['categorical_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Correlations
        if report.get('correlation_insights'):
            markdown += "## Correlation Insights\n\n"
            for insight in report['correlation_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Outliers
        if report.get('outlier_insights'):
            markdown += "## Outlier Insights\n\n"
            for insight in report['outlier_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Feature Insights
        if report.get('feature_insights'):
            markdown += "## Feature Insights\n\n"
            for insight in report['feature_insights']:
                icon = self._get_severity_icon(insight['severity'])
                markdown += f"{icon} {insight['message']}\n\n"
        
        # Recommendations
        if report.get('recommendations'):
            markdown += "## Recommendations\n\n"
            for rec in report['recommendations']:
                markdown += f"💡 {rec['message']}\n\n"
        
        return markdown
    
    def _get_severity_icon(self, severity):
        """Get icon based on severity level."""
        icons = {
            'success': '✅',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌'
        }
        return icons.get(severity, '•')
