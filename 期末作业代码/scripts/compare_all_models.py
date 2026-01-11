# scripts/compare_all_models.py
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data.sequence_dataset import TennisSequenceDataset


class RNNModel(nn.Module):
    """简单RNN模型"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        return output


class GRUModel(nn.Module):
    """GRU模型"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output


class LSTMModel(nn.Module):
    """LSTM模型（简化版）"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class TransformerModel(nn.Module):
    """简化Transformer模型"""

    def __init__(self, input_size, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入投影
        x = self.input_projection(x) * np.sqrt(self.d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        transformer_out = self.transformer_encoder(x)

        # 取最后一个时间步
        output = self.fc(transformer_out[:, -1, :])
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, model_name, device='cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # 创建保存目录
        self.save_dir = os.path.join(project_root, 'experiments', 'deep_learning_models')
        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = os.path.join(self.save_dir, f"{model_name}_{timestamp}.pth")

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        return epoch_loss / len(train_loader), correct / total

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return val_loss / len(val_loader), correct / total

    def train(self, train_loader, val_loader, epochs=30):
        print(f"\n训练 {self.model_name} 模型...")

        best_val_acc = 0
        patience = 8
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # 学习率调整
            self.scheduler.step(val_acc)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch + 1:3d}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.4f}, "
                      f"验证准确率={val_acc:.4f}")

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发")
                    break

        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.model_path))
        return best_val_acc

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(sequences)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        from sklearn.metrics import accuracy_score, roc_auc_score
        preds_binary = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_targets, preds_binary)
        auc = roc_auc_score(all_targets, all_preds)

        return accuracy, auc


def compare_models(df, feature_cols, sequence_length=10):
    """比较所有深度学习模型"""
    print("=" * 80)
    print("深度学习模型对比实验")
    print("=" * 80)

    # 创建数据集
    from src.data.sequence_dataset import create_data_loaders

    train_loader, val_loader, test_loader = create_data_loaders(
        df=df,
        sequence_length=sequence_length,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32,
        random_state=42,
        feature_cols=feature_cols
    )

    # 获取输入维度
    sample_batch, _ = next(iter(train_loader))
    input_size = sample_batch.shape[2]

    print(f"输入维度: {input_size}")
    print(f"序列长度: {sequence_length}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")

    # 定义要比较的模型
    models_config = [
        ('RNN', RNNModel(input_size, hidden_size=32)),
        ('GRU', GRUModel(input_size, hidden_size=32)),
        ('LSTM', LSTMModel(input_size, hidden_size=32)),
        ('Transformer', TransformerModel(input_size, d_model=32, nhead=4, num_layers=2))
    ]

    # 训练和评估所有模型
    results = {}
    training_histories = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    for model_name, model in models_config:
        print(f"\n{'=' * 40}")
        print(f"处理模型: {model_name}")
        print('=' * 40)

        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {num_params:,}")

        # 训练模型
        trainer = ModelTrainer(model, model_name, device)
        val_acc = trainer.train(train_loader, val_loader, epochs=30)

        # 在测试集上评估
        test_acc, test_auc = trainer.evaluate(test_loader)

        results[model_name] = {
            'validation_accuracy': val_acc,
            'test_accuracy': test_acc,
            'test_auc': test_auc,
            'parameters': num_params
        }

        training_histories[model_name] = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies
        }

        print(f"  验证准确率: {val_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  测试AUC: {test_auc:.4f}")

    return results, training_histories


def plot_comparison(results, training_histories):
    """绘制对比图表"""
    # 1. 性能对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 准确率对比
    models = list(results.keys())
    test_accuracies = [results[m]['test_accuracy'] for m in models]
    test_aucs = [results[m]['test_auc'] for m in models]
    parameters = [results[m]['parameters'] for m in models]

    # 准确率柱状图
    bars1 = axes[0, 0].bar(models, test_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_xlabel('模型')
    axes[0, 0].set_ylabel('测试准确率')
    axes[0, 0].set_title('模型准确率对比')
    axes[0, 0].axhline(y=0.6149, color='red', linestyle='--', alpha=0.5, label='发球方基线')
    axes[0, 0].legend()

    # 添加数值标签
    for bar, acc in zip(bars1, test_accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom')

    # AUC对比
    bars2 = axes[0, 1].bar(models, test_aucs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 1].set_xlabel('模型')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('模型AUC对比')
    axes[0, 1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='随机水平')
    axes[0, 1].legend()

    # 添加数值标签
    for bar, auc in zip(bars2, test_aucs):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{auc:.3f}', ha='center', va='bottom')

    # 参数量对比
    bars3 = axes[0, 2].bar(models, parameters, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 2].set_xlabel('模型')
    axes[0, 2].set_ylabel('参数量')
    axes[0, 2].set_title('模型复杂度对比')

    # 添加数值标签
    for bar, param in zip(bars3, parameters):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{param:,}', ha='center', va='bottom', fontsize=9)

    # 2. 训练历史曲线
    colors = {'RNN': '#FF6B6B', 'GRU': '#4ECDC4', 'LSTM': '#45B7D1', 'Transformer': '#96CEB4'}

    # 损失曲线
    for i, model_name in enumerate(models):
        if model_name in training_histories:
            history = training_histories[model_name]
            epochs = range(1, len(history['train_losses']) + 1)
            axes[1, 0].plot(epochs, history['train_losses'], color=colors[model_name],
                            linestyle='-', label=f'{model_name}训练损失')
            axes[1, 0].plot(epochs, history['val_losses'], color=colors[model_name],
                            linestyle='--', label=f'{model_name}验证损失')

    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('损失值')
    axes[1, 0].set_title('训练损失曲线')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)

    # 准确率曲线
    for model_name in models:
        if model_name in training_histories:
            history = training_histories[model_name]
            epochs = range(1, len(history['train_accuracies']) + 1)
            axes[1, 1].plot(epochs, history['train_accuracies'], color=colors[model_name],
                            linestyle='-', label=f'{model_name}训练准确率')
            axes[1, 1].plot(epochs, history['val_accuracies'], color=colors[model_name],
                            linestyle='--', label=f'{model_name}验证准确率')

    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_title('训练准确率曲线')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)

    # 3. 模型对比表格
    axes[1, 2].axis('off')
    table_data = []
    for model_name in models:
        table_data.append([
            model_name,
            f"{results[model_name]['test_accuracy']:.4f}",
            f"{results[model_name]['test_auc']:.4f}",
            f"{results[model_name]['parameters']:,}"
        ])

    table = axes[1, 2].table(
        cellText=table_data,
        colLabels=['模型', '测试准确率', '测试AUC', '参数量'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.tight_layout()
    plt.savefig('reports/figures/deep_learning_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def analyze_model_performance(results):
    """分析模型性能"""
    print("\n" + "=" * 80)
    print("深度学习模型性能分析")
    print("=" * 80)

    # 找到最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    worst_model = min(results.items(), key=lambda x: x[1]['test_accuracy'])

    print(f"🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]['test_accuracy']:.4f})")
    print(f"📉 最差模型: {worst_model[0]} (准确率: {worst_model[1]['test_accuracy']:.4f})")

    # 计算相对于基线的提升
    baseline = 0.6149  # 发球方基线
    print(f"\n📈 相对于发球方基线的提升:")
    for model_name, metrics in results.items():
        improvement = metrics['test_accuracy'] - baseline
        percent_improvement = (improvement / baseline) * 100
        print(f"  {model_name:12s}: {metrics['test_accuracy']:.4f} "
              f"(+{improvement:.4f}, +{percent_improvement:.1f}%)")

    # 模型效率分析
    print(f"\n⚡ 模型效率分析 (准确率/参数量):")
    for model_name, metrics in results.items():
        efficiency = metrics['test_accuracy'] / (metrics['parameters'] / 1000)  # 每千参数准确率
        print(f"  {model_name:12s}: {efficiency:.6f} 准确率/千参数")

    return best_model


def main():
    """主函数"""
    print("=" * 80)
    print("深度学习模型对比实验 - RNN/GRU/LSTM/Transformer")
    print("=" * 80)

    # 1. 加载数据
    data_path = os.path.join(project_root, 'data', 'processed', 'tennis_matches_processed.csv')
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")

    # 2. 选择特征（使用之前验证有效的特征）
    feature_cols = [
        'is_server_p1',
        'score_diff',
        'is_game_point',
        'points_diff',
        'past_5_win_rate',
        'serve_speed_norm',
        'rally_length'
    ]

    # 确保特征存在
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"使用 {len(available_features)} 个特征")

    # 3. 比较所有深度学习模型
    results, training_histories = compare_models(df, available_features, sequence_length=10)

    # 4. 绘制对比图表
    plot_comparison(results, training_histories)

    # 5. 分析性能
    best_model = analyze_model_performance(results)

    # 6. 保存结果
    results_df = pd.DataFrame([
        {
            'model': name,
            'validation_accuracy': metrics['validation_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
            'test_auc': metrics['test_auc'],
            'parameters': metrics['parameters']
        }
        for name, metrics in results.items()
    ])

    results_path = 'experiments/deep_learning_comparison_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n结果保存至: {results_path}")

    # 7. 结论
    print("\n" + "=" * 80)
    print("实验结论")
    print("=" * 80)

    if best_model[1]['test_accuracy'] > 0.62:
        print(f"✅ 深度学习模型表现优秀！{best_model[0]}模型达到{best_model[1]['test_accuracy']:.1%}准确率")
        print("   成功完成了期末作业的所有研究任务！")
    else:
        print(f"⚠️  深度学习模型表现一般，可能原因：")
        print("   - 数据量有限")
        print("   - 特征工程可以进一步优化")
        print("   - 模型参数需要调优")

    print("\n📋 完成的研究任务:")
    print("  1. ✅ 构建时间序列（按每一分的得失序列）")
    print("  2. ✅ 使用LSTM预测下一个时刻选手得分概率")
    print("  3. ✅ 对比RNN/GRU/Transformer模型的预测性能")


if __name__ == "__main__":
    main()