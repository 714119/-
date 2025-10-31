import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler

class TrafficDataLoader:
    def __init__(self, config_path="config.yaml"):
        # 修复：使用绝对路径
        if config_path is None:
            current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")
            config_path = current_dir / 'config.yaml'
        else:
            # 如果传入的是字符串，转换为Path对象
            if isinstance(config_path, str):
                config_path = Path(config_path)
            current_dir = config_path.parent

        print(f"Loading config from: {config_path}")

        with open(config_path, 'r',encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.processed_path = Path(self.config['data_paths']['processed'])
        if not self.processed_path.is_absolute():
            self.processed_path = current_dir / self.processed_path

        print(f"Data loader initialized. Processed path: {self.processed_path}")
        print(f"Path exists: {self.processed_path.exists()}")

        # 添加标准化器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False


    def load_processed_traffic_data(self, city):
        """加载处理后的交通数据"""
        file_path = self.processed_path / f"{city}_traffic_processed.parquet"

        print(f"Looking for data file: {file_path}")
        print(f"File exists: {file_path.exists()}")

        # 列出目录中的所有文件，帮助调试
        if self.processed_path.exists():
            files = list(self.processed_path.glob("*.parquet"))
            print(f"Available parquet files: {[f.name for f in files]}")

        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"Loaded {len(df)} records for {city}")
            return df
        else:
            raise FileNotFoundError(f"Processed data for {city} not found at {file_path}")

    def create_sequences(self, df, target_column='flow'):
        """创建训练序列并进行标准化"""
        features = self.config['model']['features']
        seq_len = self.config['model']['seq_len']
        pred_len = self.config['model']['pred_len']

        # 确保特征存在
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError(f"No available features found. Requested: {features}, Available: {df.columns.tolist()}")

        print(f"Using features: {available_features}")

        # 分离特征和目标
        feature_data = df[available_features].values
        target_data = df[target_column].values.reshape(-1, 1)

        # 数据标准化 - 为每个城市重新拟合
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
        target_data_scaled = self.target_scaler.fit_transform(target_data)

        print("✅ Scalers fitted on city data")
        print(f"   Target mean: {self.target_scaler.mean_[0]:.2f}, std: {np.sqrt(self.target_scaler.var_[0]):.2f}")

        X, y = [], []
        for i in range(len(feature_data_scaled) - seq_len - pred_len):
            X.append(feature_data_scaled[i:i + seq_len])
            # 目标值已经是标准化后的
            y.append(target_data_scaled[i + seq_len:i + seq_len + pred_len, 0])

        print(f"Created sequences - X: {len(X)}, each shape: {X[0].shape if X else 'None'}")
        return np.array(X), np.array(y)

    def inverse_transform_target(self, y_scaled):
        """将标准化后的目标值转换回原始尺度"""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def get_available_cities(self):
        """获取可用的城市数据"""
        available_cities = []
        if self.processed_path.exists():
            for city_file in self.processed_path.glob("*_traffic_processed.parquet"):
                city_name = city_file.name.replace("_traffic_processed.parquet", "")
                available_cities.append(city_name)
        return available_cities