import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import h5py
from datetime import datetime, timedelta


class LargeSTProcessor:
    def __init__(self, config_path=None):
        # 修复：使用绝对路径
        if config_path is None:
            current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")  # 项目根目录
            config_path = current_dir / 'config.yaml'

        print(f"Loading config from: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 确保数据路径是绝对路径
        self.data_path = Path(self.config['data_paths']['largest_raw'])
        if not self.data_path.is_absolute():
            self.data_path = Path(__file__).parent.parent.parent / self.data_path

        print(f"Data path: {self.data_path}")
        print(f"Data path exists: {self.data_path.exists()}")

    def load_real_largest_data(self, year=2019):
        """加载真实的 LargeST 加州交通数据"""
        print(f"Loading REAL LargeST California data for {year}...")

        h5_file = self.data_path / f"ca_his_raw_{year}.h5"

        print(f"Looking for file: {h5_file}")
        print(f"File exists: {h5_file.exists()}")

        if not h5_file.exists():
            # 列出目录中的所有文件来调试
            print(f"Files in data directory: {list(self.data_path.glob('*'))}")
            print(f"❌ File not found: {h5_file}")
            return None

        try:
            with h5py.File(h5_file, 'r') as f:
                # 提取数据
                traffic_data = f['t/block0_values'][:]  # 形状: (时间点, 传感器)
                sensor_ids = f['t/axis0'][:]  # 传感器ID
                time_indices = f['t/axis1'][:]  # 时间索引

                print(f"✅ Loaded traffic data: {traffic_data.shape}")
                print(f"   - Time points: {traffic_data.shape[0]}")
                print(f"   - Sensors: {traffic_data.shape[1]}")
                print(f"   - Sensor IDs sample: {sensor_ids[:5]}")
                print(f"   - Time indices sample: {time_indices[:5]}")

                # 转换为DataFrame
                df = self.create_dataframe_from_array(traffic_data, sensor_ids, time_indices, year)
                return df

        except Exception as e:
            print(f"❌ Error reading {h5_file}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_dataframe_from_array(self, traffic_data, sensor_ids, time_indices, year):
        """将数组数据转换为DataFrame"""
        print("Converting array data to DataFrame...")

        # 传感器ID解码（从bytes到string）
        sensor_ids_str = [id.decode('utf-8') if isinstance(id, bytes) else str(id) for id in sensor_ids]

        # 创建时间戳（假设5分钟间隔，从年初开始）
        start_date = datetime(year, 1, 1)
        timestamps = [start_date + timedelta(minutes=5 * i) for i in range(len(time_indices))]

        # 为了性能，先处理部分数据测试
        n_sensors_to_process = min(50, len(sensor_ids_str))  # 先处理50个传感器测试
        n_time_points = min(500, len(timestamps))  # 先处理500个时间点

        print(f"Processing {n_sensors_to_process} sensors and {n_time_points} time points for testing...")

        df_list = []
        for time_idx in range(n_time_points):
            timestamp = timestamps[time_idx]
            for sensor_idx in range(n_sensors_to_process):
                sensor_id = sensor_ids_str[sensor_idx]
                flow = traffic_data[time_idx, sensor_idx]

                # 跳过无效数据
                if np.isnan(flow) or flow < 0:
                    continue

                df_list.append({
                    'sensor_id': sensor_id,
                    'timestamp': timestamp,
                    'flow': flow,
                    # 可以添加其他特征，如速度、占用率（如果数据中有）
                })

        df = pd.DataFrame(df_list)
        print(f"✅ Created DataFrame with {len(df)} records")
        return df

    def load_metadata(self):
        """加载传感器元数据"""
        meta_file = self.data_path / "ca_meta.csv"
        print(f"Looking for metadata: {meta_file}")
        print(f"Metadata exists: {meta_file.exists()}")

        if meta_file.exists():
            meta_df = pd.read_csv(meta_file)
            # 确保ID列是字符串类型，用于匹配
            meta_df['ID'] = meta_df['ID'].astype(str)
            print(f"✅ Loaded metadata for {len(meta_df)} sensors")
            print(f"Metadata columns: {list(meta_df.columns)}")
            return meta_df
        else:
            print("❌ Metadata file not found")
            return None

    def load_adjacency_matrix(self):
        """加载邻接矩阵"""
        adj_file = self.data_path / "ca_rn_adj.npy"
        print(f"Looking for adjacency matrix: {adj_file}")
        print(f"Adjacency matrix exists: {adj_file.exists()}")

        if adj_file.exists():
            adj_matrix = np.load(adj_file)
            print(f"✅ Loaded adjacency matrix: {adj_matrix.shape}")
            return adj_matrix
        else:
            print("❌ Adjacency matrix file not found")
            return None

    def enrich_with_metadata(self, traffic_df, meta_df):
        """用元数据丰富交通数据"""
        if meta_df is None:
            return traffic_df

        print("Enriching traffic data with metadata...")

        # 合并元数据
        enriched_df = traffic_df.merge(
            meta_df[['ID', 'Lat', 'Lng', 'District', 'Fwy', 'Lanes', 'Direction']],
            left_on='sensor_id',
            right_on='ID',
            how='left'
        )

        # 清理
        enriched_df = enriched_df.drop('ID', axis=1)
        enriched_df = enriched_df.rename(columns={'Lat': 'latitude', 'Lng': 'longitude'})

        print(f"✅ Enriched data: {len(enriched_df)} records")
        print(f"   Columns: {list(enriched_df.columns)}")
        return enriched_df

    def preprocess_traffic_data(self, df):
        """预处理真实的交通数据"""
        print("Preprocessing REAL California traffic data...")

        # 基础清洗
        df = df.drop_duplicates()
        df = df.sort_values(['sensor_id', 'timestamp'])

        # 处理缺失值 - 修复：使用 ffill() 方法
        df['flow'] = df['flow'].ffill()  # 改为 ffill()

        # 过滤异常值
        df = df[(df['flow'] >= 0) & (df['flow'] <= 2000)]  # 合理的流量范围

        # 添加时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month

        print(f"✅ Preprocessed {len(df)} real California records")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Unique sensors: {df['sensor_id'].nunique()}")
        print(f"   Flow stats - Mean: {df['flow'].mean():.1f}, Std: {df['flow'].std():.1f}")

        return df


# 更新主程序
if __name__ == "__main__":
    processor = LargeSTProcessor()

    # 1. 加载真实数据（使用2019年数据测试）
    real_df = processor.load_real_largest_data(year=2019)

    if real_df is not None and not real_df.empty:
        # 2. 加载元数据和邻接矩阵
        meta_df = processor.load_metadata()
        adj_matrix = processor.load_adjacency_matrix()

        # 3. 用元数据丰富交通数据
        enriched_df = processor.enrich_with_metadata(real_df, meta_df)

        # 4. 预处理
        processed_df = processor.preprocess_traffic_data(enriched_df)

        # 5. 保存处理后的数据
        output_path = Path(processor.config['data_paths']['processed']) / "california_traffic_processed.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_parquet(output_path, index=False)
        print(f"✅ Real California traffic data saved to {output_path}")

        # 6. 构建图结构
        if adj_matrix is not None:
            sensor_ids = processed_df['sensor_id'].unique()
            # 这里可以添加图构建逻辑
    else:
        print("❌ Failed to load real data")