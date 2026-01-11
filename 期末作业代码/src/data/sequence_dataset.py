# src/data/sequence_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


class TennisSequenceDataset(Dataset):
    """网球比赛序列数据集，用于LSTM/GRU"""

    def __init__(
            self,
            df: pd.DataFrame,
            sequence_length: int = 10,
            feature_cols: Optional[List[str]] = None,
            target_col: str = 'target',
            match_id_col: str = 'match_id'
    ):
        """
        初始化序列数据集

        参数:
            df: 处理后的数据框
            sequence_length: 序列长度（历史点数）
            feature_cols: 使用的特征列，如果为None则使用所有数值列
            target_col: 目标列名
            match_id_col: 比赛ID列名
        """
        self.df = df.copy()
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.match_id_col = match_id_col

        # 选择特征列
        if feature_cols is None:
            # 默认使用所有数值列（排除ID列和目标列）
            exclude_cols = [match_id_col, 'player1', 'player2', 'elapsed_time',
                            target_col, 'next_point_victor', 'point_victor']
            self.feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
        else:
            self.feature_cols = feature_cols

        print(f"使用特征数量: {len(self.feature_cols)}")
        print(f"特征示例: {self.feature_cols[:10]}...")

        # 数据标准化
        self.scaler = StandardScaler()
        self._prepare_data()

    def _prepare_data(self):
        """准备序列数据"""
        self.sequences = []
        self.targets = []
        self.match_ids = []

        # 为每个比赛单独处理
        for match_id in self.df[self.match_id_col].unique():
            match_df = self.df[self.df[self.match_id_col] == match_id].copy()

            # 标准化特征（仅用当前比赛的数据）
            features = match_df[self.feature_cols].values

            # 创建序列
            for i in range(len(match_df) - self.sequence_length):
                # 特征序列
                sequence = features[i:i + self.sequence_length]
                # 目标（序列后一个点的结果）
                target = match_df[self.target_col].iloc[i + self.sequence_length]

                self.sequences.append(sequence)
                self.targets.append(target)
                self.match_ids.append(match_id)

        # 转换为numpy数组
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        print(f"创建了 {len(self.sequences)} 个序列")
        print(f"序列形状: {self.sequences.shape}")
        print(f"目标分布: Player1赢={self.targets.mean():.3f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])  # 保持形状一致
        return sequence, target

    def get_feature_names(self):
        """获取特征名称"""
        return self.feature_cols.copy()


def create_data_loaders(
        df: pd.DataFrame,
        sequence_length: int = 10,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        random_state: int = 42,
        feature_cols: list = None  # 添加这个参数
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器

    参数:
        df: 处理后的数据框
        sequence_length: 序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批大小
        random_state: 随机种子

    返回:
        train_loader, val_loader, test_loader
    """
    # 按比赛划分数据集，避免数据泄露
    match_ids = df['match_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(match_ids)

    n_matches = len(match_ids)
    n_train = int(n_matches * train_ratio)
    n_val = int(n_matches * val_ratio)

    train_matches = match_ids[:n_train]
    val_matches = match_ids[n_train:n_train + n_val]
    test_matches = match_ids[n_train + n_val:]

    print(f"比赛划分:")
    print(f"  训练比赛: {len(train_matches)}场")
    print(f"  验证比赛: {len(val_matches)}场")
    print(f"  测试比赛: {len(test_matches)}场")

    # 创建数据集
    train_df = df[df['match_id'].isin(train_matches)]
    val_df = df[df['match_id'].isin(val_matches)]
    test_df = df[df['match_id'].isin(test_matches)]

    print(f"数据点划分:")
    print(f"  训练数据点: {len(train_df)}")
    print(f"  验证数据点: {len(val_df)}")
    print(f"  测试数据点: {len(test_df)}")

    # 创建数据集时传递feature_cols
    train_dataset = TennisSequenceDataset(train_df, sequence_length, feature_cols)
    val_dataset = TennisSequenceDataset(val_df, sequence_length, feature_cols)
    test_dataset = TennisSequenceDataset(test_df, sequence_length, feature_cols)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows设为0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader