import pandas as pd
from pathlib import Path

def check_data_quality():
    print("=== DATA QUALITY CHECK ===\n")
    
    # Check fused data
    fused_files = list(Path('data/fused').glob('*.parquet'))
    
    for file in fused_files:
        print(f"📊 {file.name}:")
        df = pd.read_parquet(file)
        print(f"   Records: {len(df):,}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Sensors: {df['sensor_id'].nunique()}")
        print(f"   Features: {len(df.columns)}")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        print(f"   Missing values: {missing}")
        print()

if __name__ == "__main__":
    check_data_quality()