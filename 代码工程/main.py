import yaml
from pathlib import Path
import sys
import os

print("=== Starting Traffic Analysis Pipeline ===\n")

current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Source exists: {src_dir.exists()}\n")

# Test imports one by one with detailed error messages
print("=== Testing Imports ===")

try:
    from data_preprocessing.largest_processor import LargeSTProcessor
    print("✅ SUCCESS: LargeSTProcessor")
except Exception as e:
    print(f"❌ FAILED: LargeSTProcessor - {e}")
    sys.exit(1)

try:
    from data_preprocessing.noaa_processor import NOAAProcessor
    print("✅ SUCCESS: NOAAProcessor")
except Exception as e:
    print(f"❌ FAILED: NOAAProcessor - {e}")
    sys.exit(1)

try:
    from data_preprocessing.twitter_processor import TwitterProcessor
    print("✅ SUCCESS: TwitterProcessor")
except Exception as e:
    print(f"❌ FAILED: TwitterProcessor - {e}")
    sys.exit(1)

try:
    from data_preprocessing.data_fusion import DataFusion
    print("✅ SUCCESS: DataFusion")
except Exception as e:
    print(f"❌ FAILED: DataFusion - {e}")
    sys.exit(1)

print("\n✅ ALL IMPORTS SUCCESSFUL!\n")

def main():
    """Main data processing pipeline"""
    print("🚀 Starting multi-source data processing pipeline...")
    
    # Load configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
    except FileNotFoundError:
        print("❌ config.yaml not found. Using default configuration.")
        config = {
            'data_paths': {
                'largest_raw': 'data/raw/LargeST',
                'noaa_raw': 'data/raw/NOAA',
                'twitter_raw': 'data/raw/twitter',
                'processed': 'data/processed',
                'fused': 'data/fused'
            },
            'largest': {
                'target_cities': ['beijing', 'shanghai', 'guangzhou']
            }
        }
    
    # 1. Process traffic data
    print("\n" + "="*50)
    print("📊 STEP 1: Processing Traffic Data")
    print("="*50)
    
    traffic_processor = LargeSTProcessor()
    for city in config['largest']['target_cities']:
        print(f"\nProcessing {city}...")
        try:
            df = traffic_processor.load_largest_data(city)
            print(f"  Loaded {len(df)} records")
            
            df_processed = traffic_processor.preprocess_traffic_data(df)
            print(f"  Processed {len(df_processed)} records")
            
            traffic_processor.save_processed_data(df_processed, city)
            print(f"  ✅ {city} traffic data saved")
        except Exception as e:
            print(f"  ❌ Error processing {city}: {e}")
    
    # 2. Process weather data
    print("\n" + "="*50)
    print("🌤️  STEP 2: Processing Weather Data")
    print("="*50)
    
    weather_processor = NOAAProcessor()
    try:
        weather_df = weather_processor.load_noaa_data()
        print(f"Loaded {len(weather_df)} weather records")
        
        weather_processed = weather_processor.preprocess_weather_data(weather_df)
        print(f"Processed {len(weather_processed)} records")
        
        weather_resampled = weather_processor.resample_weather_data(weather_processed)
        print(f"Resampled to {len(weather_resampled)} records")
        
        output_path = Path(config['data_paths']['processed']) / "noaa_weather_processed.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weather_resampled.to_parquet(output_path, index=False)
        print(f"✅ Weather data saved to {output_path}")
    except Exception as e:
        print(f"❌ Error processing weather data: {e}")
    
    # 3. Process Twitter data
    print("\n" + "="*50)
    print("🐦 STEP 3: Processing Twitter Data")
    print("="*50)
    
    twitter_processor = TwitterProcessor()
    try:
        twitter_df = twitter_processor.load_twitter_data()
        print(f"Loaded {len(twitter_df)} Twitter records")
        
        events_processed = twitter_processor.preprocess_twitter_data(twitter_df)
        print(f"Processed {len(events_processed)} event records")
        
        events_output_path = Path(config['data_paths']['processed']) / "twitter_events_processed.parquet"
        events_processed.to_parquet(events_output_path, index=False)
        print(f"✅ Twitter data saved to {events_output_path}")
    except Exception as e:
        print(f"❌ Error processing Twitter data: {e}")
    
    # 4. Data fusion
    print("\n" + "="*50)
    print("🔗 STEP 4: Data Fusion")
    print("="*50)
    
    fusion = DataFusion()
    for city in config['largest']['target_cities']:
        print(f"\nFusing data for {city}...")
        try:
            fused_df = fusion.create_fused_dataset(city)
            if fused_df is not None and not fused_df.empty:
                print(f"  ✅ {city}: Fused {len(fused_df)} records")
                print(f"  Columns: {list(fused_df.columns)}")
            else:
                print(f"  ⚠️  {city}: No data fused")
        except Exception as e:
            print(f"  ❌ Error fusing {city}: {e}")
    
    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nGenerated files:")
    
    # List generated files
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        print("\n📁 Processed files:")
        for file in processed_dir.glob('*.parquet'):
            print(f"  - {file.name}")
    
    fused_dir = Path('data/fused')
    if fused_dir.exists():
        print("\n📁 Fused files:")
        for file in fused_dir.glob('*.parquet'):
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()