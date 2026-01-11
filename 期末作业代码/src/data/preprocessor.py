# src/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TennisDataPreprocessor:
    """网球比赛数据预处理器"""

    def __init__(self):
        self.score_map = {
            '0': 0, '15': 1, '30': 2, '40': 3,
            'AD': 4, '40-AD': 3.5
        }
        self.processed_features = []

    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """加载并验证数据"""
        df = pd.read_csv(filepath)
        print(f"✅ 加载数据: {df.shape[0]}行 × {df.shape[1]}列")
        print(f"✅ 比赛数量: {df['match_id'].nunique()}")
        print(f"✅ 总点数: {len(df)}")

        # 检查关键字段
        required_cols = ['match_id', 'point_victor', 'server', 'p1_score', 'p2_score']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        df = df.copy()

        # 1. 处理时间格式
        if 'elapsed_time' in df.columns:
            df['elapsed_seconds'] = pd.to_timedelta(df['elapsed_time']).dt.total_seconds()
            self.processed_features.append('elapsed_seconds')

        # 2. 处理文本比分
        df['p1_score_num'] = df['p1_score'].map(lambda x: self.score_map.get(str(x), 0))
        df['p2_score_num'] = df['p2_score'].map(lambda x: self.score_map.get(str(x), 0))
        self.processed_features.extend(['p1_score_num', 'p2_score_num'])

        # 3. 处理缺失值
        # 数值列前向填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # 类别列用众数填充
        categorical_cols = ['serve_width', 'serve_depth', 'return_depth', 'winner_shot_type']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df = df.copy()

        # === 基础特征 ===
        df['score_diff'] = df['p1_score_num'] - df['p2_score_num']
        df['total_points_won_diff'] = df['p1_points_won'] - df['p2_points_won']
        df['games_diff'] = df['p1_games'] - df['p2_games']
        df['sets_diff'] = df['p1_sets'] - df['p2_sets']

        # === 发球特征 ===
        df['is_server_p1'] = (df['server'] == 1).astype(int)
        df['serve_no_binary'] = (df['serve_no'] == 1).astype(int)  # 1=一发，0=二发

        # === 关键分特征 ===
        df['is_break_point'] = (df['p1_break_pt'] == 1) | (df['p2_break_pt'] == 1)
        df['is_game_point'] = (
                ((df['p1_score_num'] >= 3) & (df['p1_score_num'] > df['p2_score_num'])) |
                ((df['p2_score_num'] >= 3) & (df['p2_score_num'] > df['p1_score_num']))
        )

        # === 球员表现特征 ===
        df['p1_winners_total'] = df.groupby('match_id')['p1_winner'].transform('cumsum')
        df['p2_winners_total'] = df.groupby('match_id')['p2_winner'].transform('cumsum')
        df['p1_errors_total'] = df.groupby('match_id')['p1_unf_err'].transform('cumsum')
        df['p2_errors_total'] = df.groupby('match_id')['p2_unf_err'].transform('cumsum')

        # === 势头特征 ===
        for match_id in df['match_id'].unique():
            match_mask = df['match_id'] == match_id

            # 过去5分获胜率
            df.loc[match_mask, 'p1_win_rate_5'] = (
                df.loc[match_mask, 'point_victor']
                .rolling(window=5, min_periods=1)
                .apply(lambda x: (x == 1).mean())
            )
            df.loc[match_mask, 'p2_win_rate_5'] = 1 - df.loc[match_mask, 'p1_win_rate_5']

            # 过去3局获胜情况
            df.loc[match_mask, 'p1_game_wins_3'] = (
                df.loc[match_mask, 'game_victor']
                .rolling(window=20, min_periods=1)  # 大约3局
                .apply(lambda x: (x == 1).sum())
            )

        # === 比赛进程特征 ===
        df['point_in_match'] = df.groupby('match_id').cumcount() + 1
        df['total_points_in_match'] = df.groupby('match_id')['point_no'].transform('max')
        df['match_progress'] = df['point_in_match'] / df['total_points_in_match']

        # 记录新特征
        new_features = [col for col in df.columns if col not in self.processed_features]
        self.processed_features.extend(new_features)

        print(f"✅ 创建了 {len(new_features)} 个新特征")

        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建预测目标：下一分获胜者"""
        df = df.copy()

        # 下一分获胜者（1=player1赢，2=player2赢）
        df['next_point_victor'] = df.groupby('match_id')['point_victor'].shift(-1)

        # 删除最后一行（没有下一分）
        df = df.dropna(subset=['next_point_victor'])

        # 二分类目标：1=player1赢下一分，0=player2赢下一分
        df['target'] = (df['next_point_victor'] == 1).astype(int)

        # 重要：删除泄露的特征！
        if 'next_point_victor' in df.columns:
            df = df.drop(columns=['next_point_victor'])

        print(f"✅ 目标变量分布: Player1赢下一分的比例 = {df['target'].mean():.3f}")

        return df

    def preprocess(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """完整预处理流程"""
        print("=" * 50)
        print("🎾 网球比赛数据预处理开始")
        print("=" * 50)

        # 1. 加载数据
        df = self.load_and_validate(input_path)

        # 2. 数据清洗
        print("\n🔄 步骤1: 数据清洗")
        df = self.clean_data(df)

        # 3. 特征工程
        print("\n🔄 步骤2: 特征工程")
        df = self.create_features(df)

        # 4. 创建目标
        print("\n🔄 步骤3: 创建预测目标")
        df = self.create_target(df)

        # 5. 保存结果
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\n💾 数据已保存到: {output_path}")

        # 6. 总结报告
        print("\n" + "=" * 50)
        print("📊 预处理完成报告")
        print("=" * 50)
        print(f"最终数据形状: {df.shape}")
        print(f"特征数量: {len(df.columns)}")
        print(f"目标变量:")
        print(f"  - Player1赢下一分: {df['target'].sum()}次 ({df['target'].mean():.1%})")
        print(f"  - Player2赢下一分: {(1 - df['target']).sum()}次 ({1 - df['target'].mean():.1%})")
        print(f"比赛数量: {df['match_id'].nunique()}")
        print(f"总点数: {len(df)}")

        return df