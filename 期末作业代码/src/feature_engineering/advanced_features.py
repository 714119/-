# src/feature_engineering/advanced_features.py
import pandas as pd
import numpy as np
from typing import List, Dict


class AdvancedTennisFeatures:
    """高级网球特征工程"""

    def __init__(self):
        self.feature_names = []

    def create_player_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建球员特定特征"""
        df = df.copy()

        # 球员表现指标（累计）
        player_features = []

        # 1. 发球表现
        df['ace_rate'] = df.groupby('match_id')['p1_ace'].transform(lambda x: x.expanding().mean())
        player_features.append('ace_rate')

        df['double_fault_rate'] = df.groupby('match_id')['p1_double_fault'].transform(lambda x: x.expanding().mean())
        player_features.append('double_fault_rate')

        # 2. 进攻表现
        df['winner_rate'] = df.groupby('match_id')['p1_winner'].transform(lambda x: x.expanding().mean())
        player_features.append('winner_rate')

        df['unforced_error_rate'] = df.groupby('match_id')['p1_unf_err'].transform(lambda x: x.expanding().mean())
        player_features.append('unforced_error_rate')

        # 3. 网前表现
        if 'p1_net_pt' in df.columns and 'p1_net_pt_won' in df.columns:
            df['net_pt_success'] = df['p1_net_pt_won'] / (df['p1_net_pt'] + 1e-6)
            player_features.append('net_pt_success')

        self.feature_names.extend(player_features)
        print(f"创建球员特定特征: {len(player_features)}个")

        return df

    def create_match_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建比赛上下文特征"""
        df = df.copy()
        context_features = []

        # 1. 比赛阶段特征
        df['is_first_set'] = (df['set_no'] == 1).astype(float)
        context_features.append('is_first_set')

        df['is_final_set'] = (df['set_no'] == df.groupby('match_id')['set_no'].transform('max')).astype(float)
        context_features.append('is_final_set')

        # 2. 关键局标识
        df['is_early_game'] = (df['game_no'] <= 2).astype(float)  # 开局
        context_features.append('is_early_game')

        df['is_late_game'] = (df['game_no'] >= 5).astype(float)  # 后期
        context_features.append('is_late_game')

        # 3. 比分压力
        df['score_pressure'] = abs(df['score_diff']) / 4  # 标准化到[0,1]
        context_features.append('score_pressure')

        # 4. 连续得分/失分
        for match_id in df['match_id'].unique():
            match_mask = df['match_id'] == match_id

            # 连续得分（只使用过去信息）
            points = df.loc[match_mask, 'target'].shift(1).values
            streaks = []
            current_streak = 0

            for i in range(len(points)):
                if i == 0 or np.isnan(points[i]):
                    current_streak = 0
                elif points[i] == 1:
                    current_streak = max(current_streak + 1, 1) if current_streak >= 0 else 1
                else:
                    current_streak = min(current_streak - 1, -1) if current_streak <= 0 else -1

                streaks.append(current_streak)

            df.loc[match_mask, 'momentum_streak'] = streaks

        context_features.append('momentum_streak')

        self.feature_names.extend(context_features)
        print(f"创建比赛上下文特征: {len(context_features)}个")

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        df = df.copy()
        interaction_features = []

        # 1. 发球优势 × 比分压力
        if 'is_server_p1' in df.columns and 'score_pressure' in df.columns:
            df['server_pressure'] = df['is_server_p1'] * df['score_pressure']
            interaction_features.append('server_pressure')

        # 2. 势头 × 比赛阶段
        if 'momentum_streak' in df.columns and 'set_progress' in df.columns:
            df['momentum_progress'] = df['momentum_streak'] * df['set_progress']
            interaction_features.append('momentum_progress')

        # 3. 发球质量 × 关键分
        if 'serve_speed_norm' in df.columns and 'is_game_point' in df.columns:
            df['critical_serve'] = df['serve_speed_norm'] * df['is_game_point']
            interaction_features.append('critical_serve')

        self.feature_names.extend(interaction_features)
        print(f"创建交互特征: {len(interaction_features)}个")

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有高级特征"""
        print("开始高级特征工程...")

        df = self.create_player_specific_features(df)
        df = self.create_match_context_features(df)
        df = self.create_interaction_features(df)

        print(f"总共创建了 {len(self.feature_names)} 个高级特征")
        print("特征示例:", self.feature_names[:15])

        return df

    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.feature_names.copy()