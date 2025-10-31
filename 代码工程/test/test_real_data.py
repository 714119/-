import sys
from pathlib import Path

# 添加src到路径
current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

print(f"Python path: {sys.path}")

# 直接导入模块
try:
    from data_preprocessing.largest_processor import LargeSTProcessor

    print("✅ Successfully imported LargeSTProcessor")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    # 尝试其他导入方式
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("largest_processor",
                                                      src_dir / "data_preprocessing" / "largest_processor.py")
        largest_processor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(largest_processor)
        LargeSTProcessor = largest_processor.LargeSTProcessor
        print("✅ Successfully imported via importlib")
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)


def test_real_data_loading():
    """测试真实数据加载"""
    print("=== Testing Real LargeST Data Loading ===")

    processor = LargeSTProcessor()

    # 测试数据加载
    real_df = processor.load_real_largest_data(year=2019)

    if real_df is not None and not real_df.empty:
        print("✅ Successfully loaded real data!")
        print(f"DataFrame shape: {real_df.shape}")
        print(f"Columns: {list(real_df.columns)}")
        print(f"Sample data:\n{real_df.head()}")

        # 测试元数据加载
        meta_df = processor.load_metadata()
        if meta_df is not None:
            print(f"✅ Metadata loaded: {meta_df.shape}")

        # 测试邻接矩阵加载
        adj_matrix = processor.load_adjacency_matrix()
        if adj_matrix is not None:
            print(f"✅ Adjacency matrix loaded: {adj_matrix.shape}")

    else:
        print("❌ Failed to load real data")


if __name__ == "__main__":
    test_real_data_loading()