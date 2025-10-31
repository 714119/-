import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_training_history(train_losses, val_losses, city):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.title(f'Training History - {city.title()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'training_history_{city}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, city, sample_size=200):
    """绘制预测值与真实值对比"""
    plt.figure(figsize=(12, 6))

    # 随机采样一些点以避免过于密集
    indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
    y_true_sample = y_true[indices]
    y_pred_sample = y_pred[indices]

    plt.scatter(y_true_sample, y_pred_sample, alpha=0.6, s=20)
    plt.plot([y_true_sample.min(), y_true_sample.max()],
             [y_true_sample.min(), y_true_sample.max()], 'r--', alpha=0.8)
    plt.xlabel('Actual Traffic Flow')
    plt.ylabel('Predicted Traffic Flow')
    plt.title(f'Predictions vs Actual - {city.title()}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'predictions_actual_{city}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_summary(performance_dict):
    """创建性能总结"""
    cities = list(performance_dict.keys())
    maes = [performance_dict[city]['mae'] for city in cities]
    rmses = [performance_dict[city]['rmse'] for city in cities]

    # 创建表格
    summary_df = pd.DataFrame({
        'City': cities,
        'MAE': maes,
        'RMSE': rmses
    })

    print("📈 Performance Summary:")
    print(summary_df.to_string(index=False))

    # 绘制性能比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MAE 比较
    bars1 = ax1.bar(cities, maes, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.7)
    ax1.set_title('MAE Comparison by City')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.bar_label(bars1, fmt='%.1f')

    # RMSE 比较
    bars2 = ax2.bar(cities, rmses, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.7)
    ax2.set_title('RMSE Comparison by City')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.bar_label(bars2, fmt='%.1f')

    plt.tight_layout()
    plt.savefig('city_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return summary_df