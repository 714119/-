import h5py
from pathlib import Path


def explore_h5_files_deep():
    data_path = Path("data/raw/LargeST")

    for h5_file in data_path.glob("*.h5"):
        print(f"\n=== {h5_file.name} ===")
        with h5py.File(h5_file, 'r') as f:
            print("Root level keys:", list(f.keys()))

            # 递归探索所有组和数据集
            def explore_group(group, level=0):
                indent = "  " * level
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        print(f"{indent}📁 Group: {key}")
                        explore_group(item, level + 1)
                    elif isinstance(item, h5py.Dataset):
                        print(f"{indent}📊 Dataset: {key} - Shape: {item.shape}, Dtype: {item.dtype}")
                    else:
                        print(f"{indent}❓ Unknown: {key} - Type: {type(item)}")

            explore_group(f)


def check_metadata_files():
    """检查元数据文件"""
    data_path = Path("data/raw/LargeST")

    # 检查元数据文件
    meta_file = data_path / "ca_meta.csv"
    if meta_file.exists():
        print(f"\n=== {meta_file.name} ===")
        import pandas as pd
        meta_df = pd.read_csv(meta_file)
        print(f"Shape: {meta_df.shape}")
        print(f"Columns: {list(meta_df.columns)}")
        print("First few rows:")
        print(meta_df.head())

    # 检查邻接矩阵
    adj_file = data_path / "ca_rn_adj.npy"
    if adj_file.exists():
        print(f"\n=== {adj_file.name} ===")
        import numpy as np
        adj_matrix = np.load(adj_file)
        print(f"Shape: {adj_matrix.shape}")
        print(f"Dtype: {adj_matrix.dtype}")
        print(f"Sample values:\n{adj_matrix[:5, :5]}")


if __name__ == "__main__":
    explore_h5_files_deep()
    check_metadata_files()