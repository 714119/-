# scripts/train_fixed.py
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 创建必要的目录
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('experiments/fixed_models', exist_ok=True)


def create_proper_features(df):
    """创建正确的特征（不包含未来信息）"""
    print("创建特征集...")

    # 确保删除泄露的特征
    leaky_features = ['next_point_victor', 'point_victor']
    for feat in leaky_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    # 基础特征（当前状态）
    features = []

    # 1. 发球方特征（最重要！）
    df['is_server_p1'] = (df['server'] == 1).astype(float)
    features.append('is_server_p1')

    # 2. 比分状态
    if 'p1_score_num' in df.columns and 'p2_score_num' in df.columns:
        df['score_diff'] = df['p1_score_num'] - df['p2_score_num']
        features.append('score_diff')

        # 是否关键分（局点、破发点）
        df['is_game_point'] = (
                ((df['p1_score_num'] >= 3) & (df['score_diff'] > 0)) |
                ((df['p2_score_num'] >= 3) & (df['score_diff'] < 0))
        ).astype(float)
        features.append('is_game_point')

    # 3. 比赛进程
    df['game_in_set'] = df['game_no'] / 7  # 假设最多7局
    features.append('game_in_set')

    df['set_progress'] = df['set_no'] / 5  # 假设最多5盘
    features.append('set_progress')

    # 4. 累计统计（不包含当前点）
    if 'p1_points_won' in df.columns:
        df['points_diff'] = df['p1_points_won'] - df['p2_points_won']
        features.append('points_diff')

    # 5. 历史表现（使用shift避免数据泄露）
    for match_id in df['match_id'].unique():
        match_mask = df['match_id'] == match_id

        # 过去5分胜率（不包括当前点！）
        df.loc[match_mask, 'past_5_win_rate'] = df.loc[match_mask, 'target'].shift(1).rolling(
            window=5, min_periods=1
        ).mean()

        # 发球连续性
        df.loc[match_mask, 'serve_streak'] = df.loc[match_mask, 'server'].shift(1).rolling(
            window=3, min_periods=1
        ).apply(lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1)

    features.append('past_5_win_rate')
    features.append('serve_streak')

    # 6. 发球质量特征
    if 'speed_mph' in df.columns:
        df['serve_speed_norm'] = (df['speed_mph'] - df['speed_mph'].mean()) / df['speed_mph'].std()
        features.append('serve_speed_norm')

    if 'rally_count' in df.columns:
        df['rally_length'] = np.log1p(df['rally_count'])
        features.append('rally_length')

    print(f"使用 {len(features)} 个特征:")
    for i, feat in enumerate(features):
        print(f"  {i + 1:2d}. {feat}")

    return df, features


class TennisSequenceDataset(Dataset):
    """修正的序列数据集"""

    def __init__(self, df, feature_cols, target_col='target', seq_length=10):
        self.sequences = []
        self.targets = []

        for match_id in df['match_id'].unique():
            match_df = df[df['match_id'] == match_id]

            # 确保数据按时间排序
            match_df = match_df.sort_values(['set_no', 'game_no', 'point_no'])

            if len(match_df) < seq_length + 1:
                continue

            features = match_df[feature_cols].fillna(0).values
            targets = match_df[target_col].values

            for i in range(len(match_df) - seq_length):
                # 确保没有数据泄露：使用过去的数据预测未来
                self.sequences.append(features[i:i + seq_length])
                self.targets.append(targets[i + seq_length])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        print(f"创建 {len(self.sequences)} 个序列")
        print(f"目标分布: Player1赢 = {self.targets.mean():.3f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor([self.targets[idx]]))


class SimpleLSTM(nn.Module):
    """简化但有效的LSTM模型"""

    def __init__(self, input_size, hidden_size=32, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # 取最后一个时间步
        return self.fc(lstm_out)


def train_and_evaluate(df, feature_cols, model_name="fixed_lstm"):
    """训练和评估模型"""
    print(f"\n{'=' * 60}")
    print(f"训练 {model_name}")
    print('=' * 60)

    # 划分数据集（按比赛划分，避免数据泄露）
    match_ids = df['match_id'].unique()
    np.random.seed(42)
    np.random.shuffle(match_ids)

    n_train = int(0.7 * len(match_ids))
    n_val = int(0.15 * len(match_ids))

    train_ids = match_ids[:n_train]
    val_ids = match_ids[n_train:n_train + n_val]
    test_ids = match_ids[n_train + n_val:]

    print(f"比赛划分: 训练={len(train_ids)}, 验证={len(val_ids)}, 测试={len(test_ids)}")

    train_df = df[df['match_id'].isin(train_ids)]
    val_df = df[df['match_id'].isin(val_ids)]
    test_df = df[df['match_id'].isin(test_ids)]

    # 创建数据集
    seq_length = 10
    train_dataset = TennisSequenceDataset(train_df, feature_cols, 'target', seq_length)
    val_dataset = TennisSequenceDataset(val_df, feature_cols, 'target', seq_length)
    test_dataset = TennisSequenceDataset(test_df, feature_cols, 'target', seq_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型
    input_size = len(feature_cols)
    model = SimpleLSTM(input_size, hidden_size=32, dropout=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # 训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0
    patience = 8
    patience_counter = 0

    # 训练循环
    for epoch in range(50):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = (output > 0.5).float()
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(device), target.to(device)
                output = model(seq)
                loss = criterion(output, target)

                val_loss += loss.item()
                predicted = (output > 0.5).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_accs.append(val_acc)

        # 学习率调整
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1:2d}: "
              f"训练损失={train_losses[-1]:.4f}, 训练准确率={train_accs[-1]:.4f}, "
              f"验证准确率={val_acc:.4f}")

        # 早停和保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'experiments/fixed_models/{model_name}_best.pth')
            print(f"  ✅ 保存最佳模型 (准确率: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️  早停触发")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(f'experiments/fixed_models/{model_name}_best.pth'))

    # 在测试集上评估
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for seq, target in test_loader:
            seq, target = seq.to(device), target.to(device)
            output = model(seq)
            test_preds.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())

    test_preds = np.array(test_preds).flatten()
    test_targets = np.array(test_targets).flatten()

    test_preds_binary = (test_preds > 0.5).astype(int)
    test_acc = accuracy_score(test_targets, test_preds_binary)
    test_auc = roc_auc_score(test_targets, test_preds)

    print(f"\n测试结果:")
    print(f"  准确率: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(val_losses, 'r-', label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='训练准确率')
    plt.plot(val_accs, 'r-', label='验证准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.title('准确率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reports/figures/{model_name}_training.png', dpi=300)
    plt.show()

    return test_acc, test_auc, train_accs, val_accs


def compare_baselines(df, feature_cols):
    """与基线模型比较"""
    print(f"\n{'=' * 60}")
    print("与基线模型比较")
    print('=' * 60)

    # 简单基线：总是预测发球方赢
    baseline_acc = (df['is_server_p1'] == df['target']).mean() if 'is_server_p1' in df.columns else 0.5
    print(f"基线模型（预测发球方赢）: {baseline_acc:.4f}")

    # 随机森林基线
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # 准备数据
        X = df[feature_cols].fillna(0).values
        y = df['target'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_test, y_test)

        print(f"随机森林基线: {rf_acc:.4f}")

        # 特征重要性
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n随机森林特征重要性 (Top 5):")
        print(importances.head(5).to_string())

        return baseline_acc, rf_acc

    except Exception as e:
        print(f"随机森林失败: {e}")
        return baseline_acc, None


def main():
    print("=" * 80)
    print("修正数据泄露后的网球比赛预测")
    print("=" * 80)

    # 1. 加载数据
    data_path = os.path.join(project_root, 'data', 'processed', 'tennis_matches_processed.csv')
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")

    # 2. 删除泄露的特征
    if 'next_point_victor' in df.columns:
        df = df.drop(columns=['next_point_victor'])

    # 3. 创建正确的特征
    df, feature_cols = create_proper_features(df)
    print(f"处理后数据形状: {df.shape}")

    # 4. 与基线比较
    baseline_acc, rf_acc = compare_baselines(df, feature_cols)

    # 5. 训练LSTM模型
    lstm_acc, lstm_auc, train_history, val_history = train_and_evaluate(df, feature_cols)

    # 6. 结果汇总
    print(f"\n{'=' * 80}")
    print("最终结果汇总")
    print("=" * 80)

    results = {
        '总是预测Player1赢': 0.511,  # 从之前分析得到
        '预测发球方赢': baseline_acc,
        'LSTM模型': lstm_acc
    }

    if rf_acc:
        results['随机森林'] = rf_acc

    for model_name, accuracy in results.items():
        print(f"{model_name:20s}: {accuracy:.4f}")

    # 7. 可视化比较
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = list(results.values())

    colors = ['gray', 'blue', 'green', 'red']
    bars = plt.bar(range(len(results)), accuracies, color=colors[:len(results)])

    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='随机水平')
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('不同模型准确率比较')
    plt.xticks(range(len(results)), model_names, rotation=45, ha='right')
    plt.ylim([0.4, 0.7])
    plt.legend()

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=300)
    plt.show()

    # 8. 保存结果
    results_df = pd.DataFrame([{
        'model': name,
        'accuracy': acc,
        'baseline_improvement': acc - 0.511
    } for name, acc in results.items()])

    results_path = 'experiments/fixed_models/results_summary.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n结果保存至: {results_path}")

    # 9. 结论
    print(f"\n{'=' * 80}")
    print("结论")
    print("=" * 80)

    if lstm_acc > baseline_acc + 0.02:
        print("✅ LSTM模型显著优于基线！深度学习有效！")
    elif lstm_acc > baseline_acc:
        print("👍 LSTM模型略优于基线")
    else:
        print("⚠️  LSTM模型没有超越简单基线，建议：")
        print("  1. 尝试更复杂的特征工程")
        print("  2. 增加更多比赛数据")
        print("  3. 尝试其他模型架构（如Transformer）")
        print("  4. 考虑预测更宏观的目标（如整局胜负）")


if __name__ == "__main__":
    main()