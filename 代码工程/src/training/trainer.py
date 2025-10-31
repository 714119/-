import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class TrafficTrainer:
    def __init__(self, model, config, data_loader=None):  # 修复：添加 data_loader 参数，默认为 None
        self.model = model
        self.config = config
        self.data_loader = data_loader  # 存储 data_loader 引用
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)

    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=True
        )

        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        patience = 20
        patience_counter = 0

        print("Starting training...")
        for epoch in range(self.config['model']['epochs']):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                self.optimizer.step()
                epoch_train_loss += loss.item()

            # 验证阶段
            val_loss = self.validate(X_val, y_val)
            self.scheduler.step(val_loss)

            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(val_loss)

            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epoch': epoch
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f'Epoch {epoch:3d}: Train Loss: {train_losses[-1]:8.4f}, Val Loss: {val_loss:8.4f}, LR: {current_lr:.6f}')

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return train_losses, val_losses

    def validate(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            val_predictions = self.model(torch.FloatTensor(X_val))
            val_loss = self.criterion(val_predictions, torch.FloatTensor(y_val))
        return val_loss.item()

    def evaluate(self, X_test, y_test):
        """模型评估 - 支持标准化和未标准化数据"""
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(torch.FloatTensor(X_test))
            predictions_scaled = predictions_scaled.numpy()

        # 如果有 data_loader 就进行反标准化
        if self.data_loader is not None:
            predictions_actual = self.data_loader.inverse_transform_target(predictions_scaled)
            y_test_actual = self.data_loader.inverse_transform_target(y_test)
        else:
            # 如果没有 data_loader，使用原始数据
            predictions_actual = predictions_scaled
            y_test_actual = y_test

        mae = mean_absolute_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))

        print(f'📊 Model Performance:')
        print(f'   MAE:  {mae:.2f}')
        print(f'   RMSE: {rmse:.2f}')
        if self.data_loader is not None:
            print(f'   Target range: [{y_test_actual.min():.1f}, {y_test_actual.max():.1f}]')

        return predictions_actual, mae, rmse

    def plot_predictions(self, X_test, y_test, city, n_samples=50):
        """绘制预测结果图"""
        predictions, _, _ = self.evaluate(X_test, y_test)

        # 选择部分样本进行可视化
        indices = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)

        plt.figure(figsize=(12, 6))

        if self.data_loader is not None:
            y_test_actual = self.data_loader.inverse_transform_target(y_test)
            predictions_actual = predictions
        else:
            y_test_actual = y_test
            predictions_actual = predictions

        plt.plot(indices, y_test_actual[indices], 'bo-', label='Actual', alpha=0.7, markersize=4)
        plt.plot(indices, predictions_actual[indices], 'ro-', label='Predicted', alpha=0.7, markersize=4)
        plt.title(f'Traffic Flow Prediction - {city.title()}')
        plt.xlabel('Sample Index')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'predictions_{city}.png', dpi=300, bbox_inches='tight')
        plt.show()