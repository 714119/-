import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GranularityAnalyzer:
    """多时间粒度分析器"""

    def __init__(self, fused_data):
        self.data = fused_data

    def compare_granularities(self):
        """比较不同时间粒度的流量模式"""
        # 5分钟粒度
        flow_5min = self.data.set_index('timestamp')['flow'].resample('5min').mean()

        # 1小时粒度
        flow_1h = self.data.set_index('timestamp')['flow'].resample('1h').mean()

        # 绘制对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 5分钟粒度
        flow_5min.head(288).plot(ax=ax1, title='5分钟粒度交通流量', color='blue')
        ax1.set_ylabel('流量')

        # 1小时粒度
        flow_1h.head(24).plot(ax=ax2, title='1小时粒度交通流量', color='red')
        ax2.set_ylabel('流量')
        ax2.set_xlabel('时间')

        plt.tight_layout()
        plt.savefig('granularity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return flow_5min, flow_1h

    def analyze_patterns(self, granularity='1h'):
        """分析不同时间粒度的流量规律"""
        if granularity == '1h':
            data = self.data.set_index('timestamp').resample('1h').mean()
        else:
            data = self.data.set_index('timestamp').resample('5min').mean()

        # 时间特征分析
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek

        # 绘制小时模式
        hourly_pattern = data.groupby('hour')['flow'].mean()

        plt.figure(figsize=(10, 6))
        hourly_pattern.plot(kind='bar', color='skyblue')
        plt.title(f'{granularity}粒度 - 小时流量模式')
        plt.xlabel('小时')
        plt.ylabel('平均流量')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'hourly_pattern_{granularity}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return hourly_pattern