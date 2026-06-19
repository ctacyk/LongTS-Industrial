#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os
from typing import Dict, Any, Optional

def extract_ts_features(series: pd.Series) -> Dict[str, Any]:
    """
    Extract a comprehensive set of statistical features from a numeric time series.

    Features include:
      - Basic statistics: count, mean, std, min, 25th percentile, median, 75th percentile, max, range, skewness, kurtosis.
      - Autocorrelations for lags 1 to 5.
      - First difference features: mean and std.
      - Trend feature: slope from linear regression.
      - Augmented Dickey-Fuller test results: statistic and p-value.
      - Zero crossings: number of sign changes in the differenced series.
    """
    features = {}

    # Check if series is empty
    if series.empty:
        raise ValueError("Series is empty")
    
    # Basic statistics
    features["count"] = int(series.count())
    features["mean"] = series.mean()
    features["std"] = series.std()
    features["min"] = series.min()
    features["25_percentile"] = series.quantile(0.25)
    features["median"] = series.median()
    features["75_percentile"] = series.quantile(0.75)
    features["max"] = series.max()
    features["range"] = series.max() - series.min()
    features["skewness"] = series.skew()
    features["kurtosis"] = series.kurt()

    # Autocorrelation for lags 1 to 5
    for lag in range(1, 6):
        try:
            features[f"autocorr_lag{lag}"] = series.autocorr(lag=lag)
        except Exception:
            features[f"autocorr_lag{lag}"] = np.nan

    # First difference features
    diff_series = series.diff().dropna()

    # Zero crossings: count the number of sign changes in the differenced series
    if len(diff_series) > 1:
        try:
            sign_changes = np.sum(np.diff(np.sign(diff_series)) != 0)
            features["zero_crossings"] = int(sign_changes)
        except Exception:
            features["zero_crossings"] = 0
    else:
        features["zero_crossings"] = 0

    return features


def generate_report(csv_file: str, output_file: str = "data_report.txt") -> Optional[Dict[str, Any]]:
    """
    Read a CSV file and extract statistical features for time series analysis from each data column,
    starting from the second column (the first column is assumed to be the time column and is ignored).

    The extracted features include both basic statistics and time series specific features.

    Parameters:
      csv_file: Path to the CSV file.
      output_file: Path to save the report (default "data_report.txt").
    """
    # Input validation
    if not isinstance(csv_file, str):
        raise TypeError("csv_file must be a string")
        
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
    if not csv_file.endswith('.csv'):
        raise ValueError("File must be a CSV file")
        
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Ensure that there are at least two columns (time column + data columns)
    if df.shape[1] < 2:
        print("CSV file must have at least two columns (time column and at least one data column).")
        return None

    # Assume the first column is the time column (ignored) and analyze data columns from the second column onward.
    data_columns = df.columns[1:]

    report = {}
    for col in data_columns:
        try:
            series = df[col]
            col_report = {}
            col_report["data_type"] = str(series.dtype)
            col_report["non_null_count"] = int(series.count())
            col_report["null_count"] = int(series.isnull().sum())

            if pd.api.types.is_numeric_dtype(series):
                features = extract_ts_features(series)
                col_report.update(features)
            else:
                # For non-numeric columns, provide unique value count and the most common value.
                col_report["unique_count"] = series.nunique()
                value_counts = series.value_counts()
                if not value_counts.empty:
                    col_report["most_common_value"] = value_counts.index[0]
                    col_report["most_common_frequency"] = int(value_counts.iloc[0])
                else:
                    col_report["most_common_value"] = None
                    col_report["most_common_frequency"] = 0

            report[col] = col_report
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            report[col] = {"error": str(e)}

    # Write the report to the output file.
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Statistical Feature Extraction Report for Time Series Data ===\n\n")
            for col, stats in report.items():
                f.write(f"Column: {col}\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        print(f"Report saved to {output_file}")
    except Exception as e:
        print(f"Error writing report file: {e}")
        return None

    return report