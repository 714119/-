# src/utils/train_utils.py
import torch
import numpy as np


class EarlyStopping:
    """早停类"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # 修复：使用 np.inf 而不是 np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """保存检查点"""
        if self.verbose:
            print(f'验证损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss