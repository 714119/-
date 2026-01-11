# scripts/train_ensemble.py (修复版本)
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 尝试导入可选库
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    print("警告: XGBoost 未安装，跳过该模型")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("警告: LightGBM 未安装，跳过该模型")
    LIGHTGBM_AVAILABLE = False

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 创建目录
os.makedirs('experiments/ensemble', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)


def prepare_data(df, feature_cols, target_col='target'):
    """准备数据 - 修复数据类型问题"""
    from sklearn.model_selection import train_test_split

    print(f"准备数据，使用 {len(feature_cols)} 个特征...")

    # 确保所有特征都是数值类型
    for col in feature_cols:
        if col in df.columns:
            # 转换为数值类型，非数值转为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 移除缺失值
    df_clean = df.dropna(subset=feature_cols + [target_col])

    print(f"清理后数据形状: {df_clean.shape}")

    # 确保所有特征都是数值类型
    X = df_clean[feature_cols].astype(np.float32).values
    y = df_clean[target_col].astype(np.float32).values

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集（按比赛划分）
    match_ids = df_clean['match_id'].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)

    train_mask = df_clean['match_id'].isin(train_ids)
    test_mask = df_clean['match_id'].isin(test_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # 进一步划分训练集为训练和验证
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train_final.shape} (正样本: {y_train_final.sum() / len(y_train_final):.1%})")
    print(f"  验证集: {X_val.shape} (正样本: {y_val.sum() / len(y_val):.1%})")
    print(f"  测试集: {X_test.shape} (正样本: {y_test.sum() / len(y_test):.1%})")

    return X_train_final, X_val, y_train_final, y_val, X_test, y_test, scaler


class EnsembleTennisPredictor:
    """集成学习网球预测器 - 简化版本"""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.models = {}
        self.results = {}

    def train_models(self, X_train, y_train, X_val, y_val):
        """训练多个模型 - 修复版本"""
        print("\n训练集成模型...")

        # 1. 随机森林（表现最好的基线）
        print("1. 训练随机森林...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        # 2. XGBoost (可选)
        if XGBOOST_AVAILABLE:
            print("2. 训练XGBoost...")
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train, y_train)
                self.models['xgboost'] = xgb_model
            except Exception as e:
                print(f"XGBoost训练失败: {e}")

        # 3. LightGBM (可选)
        if LIGHTGBM_AVAILABLE:
            print("3. 训练LightGBM...")
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1  # 减少输出
                )
                lgb_model.fit(X_train, y_train)
                self.models['lightgbm'] = lgb_model
            except Exception as e:
                print(f"LightGBM训练失败: {e}")

        # 4. 梯度提升树
        print("4. 训练梯度提升树...")
        gbdt = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gbdt.fit(X_train, y_train)
        self.models['gradient_boosting'] = gbdt

        # 5. 逻辑回归（作为线性基准）
        print("5. 训练逻辑回归...")
        lr = LogisticRegression(
            max_iter=1000,
            C=0.1,
            random_state=42,
            solver='lbfgs'
        )
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr

        # 6. 简单神经网络（修复数据类型）
        print("6. 训练神经网络...")
        nn_model = self._train_neural_network(X_train, y_train, X_val, y_val)
        self.models['neural_network'] = nn_model

        return self.models

    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """训练简单神经网络 - 修复版本"""

        class SimpleNN(nn.Module):
            def __init__(self, input_size):
                super(SimpleNN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.network(x)

        # 确保数据类型正确
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        # 转换数据
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

        # 创建模型
        model = SimpleNN(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 训练
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if epoch % 20 == 0:
                    print(f"   神经网络 Epoch {epoch}: 早停触发")
                break

            if epoch % 20 == 0:
                print(f"   神经网络 Epoch {epoch}: 训练损失={loss.item():.4f}, 验证损失={val_loss.item():.4f}")

        # 加载最佳模型
        model.load_state_dict(best_model_state)
        return model

    def evaluate_models(self, X_test, y_test):
        """评估所有模型"""
        print("\n评估模型表现...")

        results = {}

        # 评估各个模型
        for name, model in self.models.items():
            if name == 'neural_network':
                with torch.no_grad():
                    model.eval()
                    X_tensor = torch.FloatTensor(X_test.astype(np.float32))
                    prob = model(X_tensor).numpy().flatten()
            else:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_test)[:, 1]
                else:
                    prob = model.predict(X_test).astype(float)

            pred = (prob > 0.5).astype(int)
            acc = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, prob)

            results[name] = {
                'accuracy': acc,
                'auc': auc,
                'predictions': pred,
                'probabilities': prob
            }

            print(f"  {name:20s}: 准确率={acc:.4f}, AUC={auc:.4f}")

        # 集成预测（简单平均）
        if len(results) > 1:
            print("\n计算集成预测...")
            all_probs = []
            for name, metrics in results.items():
                if 'probabilities' in metrics:
                    all_probs.append(metrics['probabilities'])

            if all_probs:
                avg_prob = np.mean(all_probs, axis=0)
                ensemble_pred = (avg_prob > 0.5).astype(int)
                ensemble_acc = accuracy_score(y_test, ensemble_pred)
                ensemble_auc = roc_auc_score(y_test, avg_prob)

                results['ensemble'] = {
                    'accuracy': ensemble_acc,
                    'auc': ensemble_auc,
                    'predictions': ensemble_pred,
                    'probabilities': avg_prob
                }

                print(f"  {'ensemble':20s}: 准确率={ensemble_acc:.4f}, AUC={ensemble_auc:.4f}")

        # 保存结果
        self.results = results

        return results

    def plot_comparison(self, results):
        """绘制模型比较图"""
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        aucs = [results[m]['auc'] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 准确率比较
        x = range(len(models))
        bars1 = ax1.bar(x, accuracies, color='skyblue', alpha=0.8)
        ax1.set_xlabel('模型')
        ax1.set_ylabel('准确率')
        ax1.set_title('模型准确率比较')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.axhline(y=0.6149, color='red', linestyle='--', alpha=0.5, label='发球方基线')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='随机水平')
        ax1.legend()

        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        # AUC比较
        bars2 = ax2.bar(x, aucs, color='lightgreen', alpha=0.8)
        ax2.set_xlabel('模型')
        ax2.set_ylabel('AUC')
        ax2.set_title('模型AUC比较')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='随机水平')
        ax2.legend()

        # 添加数值标签
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{auc:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('reports/figures/ensemble_comparison.png', dpi=300)
        plt.show()

        return fig


def main():
    print("=" * 80)
    print("集成学习网球比赛预测 - 修复版本")
    print("=" * 80)

    # 1. 加载数据
    data_path = os.path.join(project_root, 'data', 'processed', 'tennis_matches_processed.csv')
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")

    # 2. 选择特征（使用之前验证有效的特征）
    base_features = [
        'is_server_p1',
        'score_diff',
        'is_game_point',
        'points_diff',
        'past_5_win_rate',
        'serve_speed_norm',
        'rally_length',
        'game_in_set',
        'set_progress',
        'serve_streak'
    ]

    # 添加可能的高级特征
    advanced_features = [
        'ace_rate', 'double_fault_rate', 'winner_rate',
        'unforced_error_rate', 'net_pt_success', 'is_first_set',
        'is_final_set', 'is_early_game', 'is_late_game',
        'score_pressure', 'momentum_streak', 'server_pressure'
    ]

    # 合并特征，只保留实际存在的
    all_features = []
    for feat in base_features + advanced_features:
        if feat in df.columns:
            all_features.append(feat)

    print(f"\n使用 {len(all_features)} 个特征:")
    for i, feat in enumerate(all_features[:20]):  # 显示前20个
        print(f"  {i + 1:2d}. {feat}")

    if len(all_features) > 20:
        print(f"  ... 还有 {len(all_features) - 20} 个特征")

    # 3. 准备数据
    X_train, X_val, y_train, y_val, X_test, y_test, scaler = prepare_data(
        df, all_features, 'target'
    )

    # 4. 训练集成模型
    ensemble = EnsembleTennisPredictor(all_features)
    models = ensemble.train_models(X_train, y_train, X_val, y_val)

    # 5. 评估
    results = ensemble.evaluate_models(X_test, y_test)

    # 6. 可视化
    ensemble.plot_comparison(results)

    # 7. 保存结果
    results_df = pd.DataFrame([
        {'model': name, 'accuracy': metrics['accuracy'], 'auc': metrics['auc']}
        for name, metrics in results.items()
    ])

    results_path = 'experiments/ensemble/results_fixed.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n结果保存至: {results_path}")

    # 8. 分析
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.4f})")

    baseline = 0.6149  # 发球方基线

    if best_model[1]['accuracy'] > baseline + 0.02:
        print(f"✅ 显著超越基线! (+{best_model[1]['accuracy'] - baseline:.4f})")
    elif best_model[1]['accuracy'] > baseline:
        print(f"👍 略优于基线 (+{best_model[1]['accuracy'] - baseline:.4f})")
    else:
        print(f"⚠️  未达到基线水平 (-{baseline - best_model[1]['accuracy']:.4f})")

    # 9. 特征重要性分析
    if 'random_forest' in models:
        print(f"\n随机森林特征重要性 (Top 10):")
        rf_model = models['random_forest']
        importances = pd.DataFrame({
            'feature': all_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(importances.head(10).to_string())

        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        top_features = importances.head(15)
        plt.barh(range(len(top_features)), top_features['importance'][::-1])
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('重要性')
        plt.title('Top 15 特征重要性 (随机森林)')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance_rf.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    main()