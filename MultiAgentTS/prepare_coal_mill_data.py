"""
Prepare 7-day reference data from merged.csv for coal mill simulation
"""
import pandas as pd
import os

def prepare_seven_day_data():
    """Extract first 7 days of data from merged.csv"""
    
    # Read merged data
    input_path = "../aligned_output/merged.csv"
    output_path = "Dataset/CoalMill/coal_mill_7days.csv"
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(input_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the first timestamp
    start_time = df['timestamp'].min()
    
    # Calculate 7 days from start
    end_time = start_time + pd.Timedelta(days=7)
    
    # Filter data for first 7 days
    seven_day_data = df[df['timestamp'] < end_time].copy()
    
    print(f"Original data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total rows in original data: {len(df)}")
    print(f"\n7-day data range: {seven_day_data['timestamp'].min()} to {seven_day_data['timestamp'].max()}")
    print(f"Total rows in 7-day data: {len(seven_day_data)}")
    print(f"\nColumns: {list(seven_day_data.columns)}")
    print(f"\nData shape: {seven_day_data.shape}")
    
    # Save to CSV
    seven_day_data.to_csv(output_path, index=False)
    print(f"\n7-day reference data saved to: {output_path}")
    
    # Display basic statistics
    print("\n=== Basic Statistics ===")
    print(seven_day_data.describe())
    
    return output_path

if __name__ == "__main__":
    prepare_seven_day_data()

