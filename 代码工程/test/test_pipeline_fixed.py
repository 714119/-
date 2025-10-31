import sys
from pathlib import Path

# 添加src到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'src'))


def test_imports():
    """测试所有导入"""
    print("=== Testing Imports ===")
    try:
        from models.traffic_predictor import SpatialTemporalPredictor
        print("✅ SpatialTemporalPredictor import successful")

        from training.trainer import TrafficTrainer
        print("✅ TrafficTrainer import successful")

        from analysis.multi_granularity_analysis import GranularityAnalyzer
        print("✅ GranularityAnalyzer import successful")

        from utils.data_loader import TrafficDataLoader
        print("✅ TrafficDataLoader import successful")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """测试配置文件加载"""
    print("\n=== Testing Config Loading ===")
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ Config file loaded successfully")
        print(f"   - Target cities: {config['largest']['target_cities']}")
        print(f"   - Model features: {config['model']['features']}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_data_files():
    """测试数据文件是否存在"""
    print("\n=== Testing Data Files ===")
    try:
        from utils.data_loader import TrafficDataLoader
        data_loader = TrafficDataLoader()

        available_cities = data_loader.get_available_cities()
        print(f"✅ Available cities: {available_cities}")

        if available_cities:
            # 测试第一个城市的数据加载
            city = available_cities[0]
            print(f"Testing data loading for {city}...")
            df = data_loader.load_processed_traffic_data(city)
            print(f"✅ Data loaded successfully: {len(df)} records")

            # 测试序列创建
            X, y = data_loader.create_sequences(df)
            print(f"✅ Sequences created: X{X.shape}, y{y.shape}")

            return True
        else:
            print("❌ No processed data found. Please run main.py first to generate sample data.")
            return False

    except Exception as e:
        print(f"❌ Data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing Pipeline Components ===")

    success = True
    success &= test_imports()
    success &= test_config_loading()
    success &= test_data_files()

    if success:
        print("\n🎉 All tests passed! Ready to run training pipeline.")
        print("\nNext step: Run 'python src/training/train_pipeline.py'")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")