# src/feature_selection.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import matplotlib.pyplot as plt


def select_best_features(df, target_col='target', k=20):
    """选择最佳特征"""
    # 排除非特征列
    exclude_cols = [
        'match_id', 'player1', 'player2', 'elapsed_time',
        target_col, 'next_point_victor', 'point_victor',
        'p1_score', 'p2_score'  # 原始比分文本
    ]

    # 获取数值特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"候选特征数量: {len(feature_cols)}")

    X = df[feature_cols].fillna(0).values
    y = df[target_col].values

    # 计算互信息
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    # 选择top-k特征
    selected_features = mi_df.head(k)['feature'].tolist()

    print(f"\nTop-{k} 特征:")
    print(mi_df.head(k).to_string())

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.barh(range(min(20, len(mi_df))), mi_df.head(20)['mi_score'][::-1])
    plt.yticks(range(min(20, len(mi_df))), mi_df.head(20)['feature'][::-1])
    plt.xlabel('互信息得分')
    plt.title('特征重要性 (互信息)')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=300)
    plt.show()

    return selected_features


def create_momentum_features(df):
    """创建更有效的势头特征"""
    df = df.copy()

    # 基础特征
    features = []

    # 1. 发球优势
    df['server_advantage'] = (df['server'] == 1).astype(int) * 0.6  # 发球方优势系数

    # 2. 比分差值（标准化）
    if 'score_diff' in df.columns:
        df['score_diff_norm'] = df['score_diff'] / 4  # 最大差值为4

    # 3. 近期势头（过去3分）
    for match_id in df['match_id'].unique():
        match_mask = df['match_id'] == match_id
        # 过去3分的胜率
        df.loc[match_mask, 'momentum_3'] = df.loc[match_mask, 'point_victor'].rolling(
            window=3, min_periods=1
        ).apply(lambda x: (x == 1).mean())

        # 发球连续性
        df.loc[match_mask, 'serve_consistency'] = df.loc[match_mask, 'server'].rolling(
            window=3, min_periods=1
        ).apply(lambda x: (x == x.iloc[0]).mean())

    # 4. 关键分标识
    df['is_critical_point'] = (
            (df['is_break_point'] == 1) |
            (df['is_game_point'] == 1) |
            (df['score_diff'].abs() >= 3)  # 40-0 或 0-40
    ).astype(int)

    # 5. 比赛进程
    df['match_momentum'] = df['p1_win_rate_5'] - 0.5  # 中心化

    # 6. 发球质量（如果有相关数据）
    if 'speed_mph' in df.columns:
        df['serve_speed_norm'] = (df['speed_mph'] - 100) / 50  # 标准化

    # 收集特征名
    momentum_features = [
        'server_advantage', 'score_diff_norm', 'momentum_3',
        'serve_consistency', 'is_critical_point', 'match_momentum'
    ]

    if 'serve_speed_norm' in df.columns:
        momentum_features.append('serve_speed_norm')

    # 添加其他重要特征
    additional_features = [
        'p1_games', 'p2_games', 'p1_sets', 'p2_sets',
        'p1_points_won', 'p2_points_won', 'rally_count'
    ]

    all_features = momentum_features + additional_features

    print(f"创建了 {len(all_features)} 个核心特征")

    return df, all_features