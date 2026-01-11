# src/models/lstm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TennisLSTM(nn.Module):
    """网球比赛预测LSTM模型"""

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = False
    ):
        """
        初始化LSTM模型

        参数:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            bidirectional: 是否使用双向LSTM
        """
        super(TennisLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 注意力机制（可选）
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * (2 if bidirectional else 1), 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # 全连接层
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入序列 [batch_size, seq_len, input_size]

        返回:
            预测概率 [batch_size, 1]
        """
        batch_size = x.size(0)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_size * num_directions]

        if self.use_attention:
            # 注意力机制
            attention_weights = F.softmax(self.attention(lstm_out), dim=1)
            # attention_weights: [batch_size, seq_len, 1]

            # 加权求和
            context = torch.sum(attention_weights * lstm_out, dim=1)
            # context: [batch_size, hidden_size * num_directions]
        else:
            # 使用最后一个时间步的输出
            context = lstm_out[:, -1, :]

        # 全连接层
        output = self.fc(context)

        return output

    def predict_proba(self, x):
        """预测概率"""
        with torch.no_grad():
            self.eval()
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # 添加batch维度
            return self.forward(x).cpu().numpy()

    def get_attention_weights(self, x):
        """获取注意力权重（用于可视化）"""
        with torch.no_grad():
            self.eval()
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)

            lstm_out, _ = self.lstm(x)
            attention_weights = F.softmax(self.attention(lstm_out), dim=1)

            return attention_weights.squeeze().cpu().numpy()


class TennisGRU(nn.Module):
    """GRU模型（与LSTM对比用）"""

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3
    ):
        super(TennisGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output


class TennisRNN(nn.Module):
    """RNN模型"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(TennisRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        return output


class TennisTransformer(nn.Module):
    """Transformer模型"""

    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super(TennisTransformer, self).__init__()
        self.d_model = d_model

        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
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


def get_model(model_name, input_size, **kwargs):
    """获取指定模型"""
    models = {
        'rnn': TennisRNN,
        'gru': TennisGRU,
        'lstm': TennisLSTM,
        'transformer': TennisTransformer
    }

    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}。可选: {list(models.keys())}")

    return models[model_name.lower()](input_size, **kwargs)