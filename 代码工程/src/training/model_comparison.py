import torch
import sys
from pathlib import Path

# 添加src到路径
current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")  # 项目根目录
sys.path.append(str(current_dir / 'src'))

from models.traffic_predictor import EnhancedTrafficPredictor, CNNLSTMPredictor, SimpleTrafficPredictor


def compare_models():
    """比较不同模型的复杂度"""
    input_dim = 5
    seq_len = 12
    pred_len = 3

    print("=== Model Comparison ===")

    models = {
        'Simple LSTM': SimpleTrafficPredictor(input_dim, 32, seq_len, pred_len),
        'Enhanced LSTM': EnhancedTrafficPredictor(input_dim, 64, seq_len, pred_len),
        'CNN-LSTM': CNNLSTMPredictor(input_dim, 64, seq_len, pred_len)
    }

    # 测试数据
    test_input = torch.randn(32, seq_len, input_dim)

    for name, model in models.items():
        try:
            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # 测试前向传播
            with torch.no_grad():
                output = model(test_input)

            print(f"\n🔧 {name}:")
            print(f"   Parameters: {total_params:,} (Trainable: {trainable_params:,})")
            print(f"   Output shape: {output.shape}")
            print(f"   ✅ Model works correctly")

        except Exception as e:
            print(f"   ❌ Model failed: {e}")


if __name__ == "__main__":
    compare_models()