import yaml
import torch
import numpy as np
from pathlib import Path
import sys


def main():
    print("=== Starting Traffic Prediction Training Pipeline ===")

    # 修复：使用绝对路径加载配置
    current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")
    config_path = current_dir / 'config.yaml'

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 添加src到路径
    sys.path.append(str(current_dir / 'src'))

    from models.traffic_predictor import EnhancedTrafficPredictor
    from training.trainer import TrafficTrainer
    from utils.data_loader import TrafficDataLoader

    # 初始化组件
    data_loader = TrafficDataLoader(config_path=config_path)

    # 使用加州数据
    cities = ['california']  # 改为加州

    performance_results = {}

    for city in cities:
        print(f"\n🚀 Training model for {city}...")

        try:
            # 加载数据
            df = data_loader.load_processed_traffic_data(city)
            print(f"Loaded {len(df)} records for {city}")

            # 创建序列
            X, y = data_loader.create_sequences(df)
            print(f"Created {len(X)} sequences for training")
            print(f"Sequence shape: {X[0].shape}, Target shape: {y[0].shape}")

            # 分割数据集
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            val_idx = int(0.8 * len(X_train))
            X_train, X_val = X_train[:val_idx], X_train[val_idx:]
            y_train, y_val = y_train[:val_idx], y_train[val_idx:]

            print(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

            # 初始化模型
            input_dim = X_train.shape[-1]
            seq_len = config['model']['seq_len']
            pred_len = config['model']['pred_len']

            print(f"Model parameters - Input: {input_dim}, Sequence: {seq_len}, Predict: {pred_len}")

            # 使用增强模型
            model = EnhancedTrafficPredictor(
                input_dim=input_dim,
                hidden_dim=64,
                seq_len=seq_len,
                pred_len=pred_len,
                num_layers=2
            )

            # 训练模型
            trainer = TrafficTrainer(model, config, data_loader)
            train_losses, val_losses = trainer.train(X_train, y_train, X_val, y_val)

            # 评估模型
            predictions, mae, rmse = trainer.evaluate(X_test, y_test)

            # 保存性能结果
            performance_results[city] = {
                'mae': mae,
                'rmse': rmse,
                'train_losses': train_losses,
                'val_losses': val_losses
            }

            print(f"📊 {city} Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

            # 绘制预测结果
            trainer.plot_predictions(X_test, y_test, city, n_samples=30)

            # 保存模型
            model_save_dir = current_dir / 'saved_models'
            model_save_dir.mkdir(exist_ok=True)
            model_save_path = model_save_dir / f"{city}_enhanced_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Model saved to {model_save_path}")

        except Exception as e:
            print(f"❌ Error training {city}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 性能总结
    print(f"\n{'=' * 50}")
    print("📈 TRAINING SUMMARY")
    print(f"{'=' * 50}")

    for city, results in performance_results.items():
        print(f"{city.upper():12} | MAE: {results['mae']:6.2f} | RMSE: {results['rmse']:6.2f}")

    # 绘制训练历史
    if performance_results:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        for city, results in performance_results.items():
            plt.plot(results['train_losses'], label=f'{city} Train', alpha=0.7)
            plt.plot(results['val_losses'], label=f'{city} Val', alpha=0.7, linestyle='--')

        plt.title('Training History - California Traffic Prediction')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history_california.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Training history saved to: training_history_california.png")


if __name__ == "__main__":
    main()