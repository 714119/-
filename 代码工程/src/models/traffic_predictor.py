import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedTrafficPredictor(nn.Module):
    """增强版交通预测模型"""

    def __init__(self, input_dim, hidden_dim=128, seq_len=12, pred_len=3, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # 更深的LSTM网络
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True  # 双向LSTM
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # 双向所以是2倍
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # 更深的预测网络
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, pred_len)
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        print(f"Enhanced Model: input_dim={input_dim}, hidden_dim={hidden_dim}, bidirectional=True")

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]

        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden_dim*2]

        # 注意力机制
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        attended_out = self.layer_norm1(attended_out + lstm_out)  # 残差连接

        # 取最后时间步
        last_hidden = attended_out[:, -1, :]  # [batch, hidden_dim*2]

        # 预测
        predictions = self.predictor(last_hidden)  # [batch, pred_len]

        return predictions


class CNNLSTMPredictor(nn.Module):
    """CNN+LSTM混合模型"""

    def __init__(self, input_dim, hidden_dim=64, seq_len=12, pred_len=3):
        super().__init__()

        # 1D CNN 提取局部时序模式
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(seq_len // 2)  # 降采样
        )

        # LSTM
        self.lstm = nn.LSTM(
            64, hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, pred_len)
        )

        print(f"CNN-LSTM Model: input_dim={input_dim}, seq_len={seq_len}")

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.shape[0]

        # CNN处理: [batch, input_dim, seq_len]
        cnn_input = x.transpose(1, 2)
        cnn_features = self.conv1d(cnn_input)  # [batch, 64, seq_len//2]

        # 转回LSTM输入格式: [batch, seq_len//2, 64]
        lstm_input = cnn_features.transpose(1, 2)

        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # 预测
        predictions = self.predictor(last_hidden)

        return predictions


# 简化但有效的模型
class SimpleTrafficPredictor(nn.Module):
    """简化版交通预测模型 - 保证能运行"""

    def __init__(self, input_dim, hidden_dim=32, seq_len=12, pred_len=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.linear = nn.Linear(hidden_dim, pred_len)

        print(f"Simple Model: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])  # 用最后一个时间步预测
        return predictions
