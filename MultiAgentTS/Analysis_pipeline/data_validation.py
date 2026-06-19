"""
Data validation module for time series data analysis.
This module provides functions to validate and compare synthetic time series data with real data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

class DataValidator:
    """Class for validating and comparing time series data."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        pass
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load time series data from a CSV file.
        
        :param file_path: Path to the CSV file
        :return: DataFrame with the loaded data
        """
        try:
            # Read CSV file with first column as index
            df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")
    
    @staticmethod
    def validate_data_structure(df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Validate the structure of the time series data.
        
        :param df: DataFrame with time series data
        :param expected_columns: List of expected column names
        :return: Dictionary with validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'index_type': str(type(df.index)),
            'has_missing_values': df.isnull().any().any(),
            'missing_value_counts': df.isnull().sum().to_dict()
        }
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            results['valid'] = False
            results['issues'].append("Index is not a DatetimeIndex")
        
        # Check for expected columns
        if expected_columns:
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                results['valid'] = False
                results['issues'].append(f"Missing expected columns: {missing_columns}")
        
        # Check for duplicate indices
        if df.index.duplicated().any():
            results['valid'] = False
            results['issues'].append("Duplicate indices found")
        
        # Check for empty data
        if df.empty:
            results['valid'] = False
            results['issues'].append("DataFrame is empty")
        
        return results
    
    @staticmethod
    def compare_datasets(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                        significance_level: float = 0.05) -> Dict[str, any]:
        """
        Compare real and synthetic datasets using statistical tests.
        
        :param real_data: DataFrame with real data
        :param synthetic_data: DataFrame with synthetic data
        :param significance_level: Significance level for statistical tests
        :return: Dictionary with comparison results
        """
        results = {
            'columns_compared': [],
            'statistical_tests': {},
            'similarity_scores': {},
            'overall_similarity': 0.0
        }
        
        # Get common columns
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        results['columns_compared'] = common_columns
        
        similarity_scores = []
        
        for column in common_columns:
            real_series = real_data[column].dropna()
            synthetic_series = synthetic_data[column].dropna()
            
            # Skip if either series is empty
            if len(real_series) == 0 or len(synthetic_series) == 0:
                continue
            
            # Kolmogorov-Smirnov test for distribution similarity
            try:
                ks_stat, ks_p_value = stats.ks_2samp(real_series, synthetic_series)
                ks_similarity = 1 - ks_stat  # Convert to similarity score
            except Exception:
                ks_stat, ks_p_value, ks_similarity = np.nan, np.nan, 0.0
            
            # Pearson correlation
            try:
                correlation = real_series.corr(synthetic_series)
                # Handle case where correlation might be NaN
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0
            
            # Mean absolute error
            try:
                # Align series by index and interpolate missing values for comparison
                combined_index = real_data.index.union(synthetic_data.index)
                real_aligned = real_data[column].reindex(combined_index).interpolate()
                synthetic_aligned = synthetic_data[column].reindex(combined_index).interpolate()
                
                mae = np.mean(np.abs(real_aligned - synthetic_aligned))
                rmse = np.sqrt(np.mean((real_aligned - synthetic_aligned) ** 2))
            except Exception:
                mae, rmse = np.nan, np.nan
            
            # Store results
            results['statistical_tests'][column] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'ks_similarity': ks_similarity,
                'correlation': correlation,
                'mae': mae,
                'rmse': rmse,
                'real_mean': real_series.mean(),
                'synthetic_mean': synthetic_series.mean(),
                'real_std': real_series.std(),
                'synthetic_std': synthetic_series.std()
            }
            
            # Calculate weighted similarity score (higher weight for correlation and KS similarity)
            similarity_score = (0.4 * max(0, ks_similarity) + 
                              0.4 * abs(correlation) + 
                              0.2 * (1 - min(1, mae / (abs(real_series.mean()) + 1e-8))))
            results['similarity_scores'][column] = max(0, min(1, similarity_score))
            similarity_scores.append(results['similarity_scores'][column])
        
        # Calculate overall similarity
        if similarity_scores:
            results['overall_similarity'] = np.mean(similarity_scores)
        
        return results
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """
        Check the quality of time series data.
        
        :param df: DataFrame with time series data
        :return: Dictionary with data quality metrics
        """
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'outliers': {},
            'data_types': {},
            'summary_stats': {}
        }
        
        for column in df.columns:
            series = df[column]
            
            # Missing values
            missing_count = series.isnull().sum()
            quality_metrics['missing_values'][column] = {
                'count': missing_count,
                'percentage': (missing_count / len(series)) * 100
            }
            
            # Data type
            quality_metrics['data_types'][column] = str(series.dtype)
            
            # Summary statistics for numeric columns
            if pd.api.types.is_numeric_dtype(series):
                # Outlier detection using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                quality_metrics['outliers'][column] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(series)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                # Summary statistics
                quality_metrics['summary_stats'][column] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurt()
                }
            else:
                # For non-numeric columns
                quality_metrics['outliers'][column] = {
                    'count': 0,
                    'percentage': 0.0
                }
                quality_metrics['summary_stats'][column] = {
                    'unique_values': series.nunique(),
                    'most_common': series.value_counts().index[0] if len(series.value_counts()) > 0 else None,
                    'most_common_count': series.value_counts().iloc[0] if len(series.value_counts()) > 0 else 0
                }
        
        return quality_metrics
    
    @staticmethod
    def generate_validation_report(real_data_path: str, synthetic_data_path: str, 
                                 output_path: Optional[str] = None) -> Dict[str, any]:
        """
        Generate a comprehensive validation report comparing real and synthetic data.
        
        :param real_data_path: Path to the real data CSV file
        :param synthetic_data_path: Path to the synthetic data CSV file
        :param output_path: Path to save the report (optional)
        :return: Dictionary with validation report
        """
        # Load data
        real_data = DataValidator.load_data(real_data_path)
        synthetic_data = DataValidator.load_data(synthetic_data_path)
        
        # Validate data structure
        real_validation = DataValidator.validate_data_structure(real_data)
        synthetic_validation = DataValidator.validate_data_structure(synthetic_data)
        
        # Compare datasets
        comparison_results = DataValidator.compare_datasets(real_data, synthetic_data)
        
        # Check data quality
        real_quality = DataValidator.check_data_quality(real_data)
        synthetic_quality = DataValidator.check_data_quality(synthetic_data)
        
        # Generate report
        report = {
            'report_title': 'Time Series Data Validation Report',
            'real_data_path': real_data_path,
            'synthetic_data_path': synthetic_data_path,
            'real_data_validation': real_validation,
            'synthetic_data_validation': synthetic_validation,
            'data_comparison': comparison_results,
            'real_data_quality': real_quality,
            'synthetic_data_quality': synthetic_quality,
            'overall_assessment': {
                'structure_valid': real_validation['valid'] and synthetic_validation['valid'],
                'overall_similarity': comparison_results['overall_similarity'],
                'quality_score': (sum([v['percentage'] for v in real_quality['missing_values'].values()]) + 
                                sum([v['percentage'] for v in synthetic_quality['missing_values'].values()])) / 
                               (2 * len(real_quality['missing_values']) + 1e-8)
            }
        }
        
        # Add recommendations
        recommendations = []
        if not report['overall_assessment']['structure_valid']:
            recommendations.append("Fix data structure issues in one or both datasets")
        
        if report['overall_assessment']['overall_similarity'] < 0.7:
            recommendations.append("Improve synthetic data generation to better match real data patterns")
        
        missing_data_real = sum([v['percentage'] for v in real_quality['missing_values'].values()])
        missing_data_synthetic = sum([v['percentage'] for v in synthetic_quality['missing_values'].values()])
        if missing_data_real > 5 or missing_data_synthetic > 5:
            recommendations.append("Address missing data issues in datasets")
        
        report['recommendations'] = recommendations
        
        # Save report to file if path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("=== Time Series Data Validation Report ===\n\n")
                    f.write(f"Real Data Path: {real_data_path}\n")
                    f.write(f"Synthetic Data Path: {synthetic_data_path}\n\n")
                    
                    f.write("== Data Structure Validation ==\n")
                    f.write(f"Real Data Valid: {real_validation['valid']}\n")
                    f.write(f"Synthetic Data Valid: {synthetic_validation['valid']}\n")
                    if real_validation['issues']:
                        f.write(f"Real Data Issues: {', '.join(real_validation['issues'])}\n")
                    if synthetic_validation['issues']:
                        f.write(f"Synthetic Data Issues: {', '.join(synthetic_validation['issues'])}\n")
                    f.write("\n")
                    
                    f.write("== Data Comparison ==\n")
                    f.write(f"Overall Similarity: {comparison_results['overall_similarity']:.4f}\n")
                    f.write("Column Similarities:\n")
                    for col, score in comparison_results['similarity_scores'].items():
                        f.write(f"  {col}: {score:.4f}\n")
                    f.write("\n")
                    
                    f.write("== Recommendations ==\n")
                    for rec in recommendations:
                        f.write(f"- {rec}\n")
                        
                print(f"Validation report saved to {output_path}")
            except Exception as e:
                print(f"Error saving report: {e}")
        
        return report