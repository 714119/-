import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加src到路径并获取项目根目录
current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")
sys.path.append(str(current_dir / 'src'))


def analyze_real_data_insights():
    """分析真实加州交通数据的洞察"""
    print("=== Real California Traffic Data Insights ===")

    # 修复：使用绝对路径导入
    from utils.data_loader import TrafficDataLoader

    # 修复：传递绝对路径给DataLoader
    config_path = current_dir / 'config.yaml'
    data_loader = TrafficDataLoader(config_path=config_path)

    try:
        # 加载处理后的真实数据
        df = data_loader.load_processed_traffic_data('california')

        print(f"📊 Dataset Overview:")
        print(f"   Total records: {len(df):,}")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Unique sensors: {df['sensor_id'].nunique()}")
        print(f"   Features: {list(df.columns)}")

        # 基本统计
        print(f"\n📈 Traffic Flow Statistics:")
        print(f"   Mean: {df['flow'].mean():.1f} vehicles/5min")
        print(f"   Std: {df['flow'].std():.1f}")
        print(f"   Min: {df['flow'].min():.1f}")
        print(f"   Max: {df['flow'].max():.1f}")

        # 创建可视化
        plt.figure(figsize=(15, 10))

        # 1. 流量分布
        plt.subplot(2, 3, 1)
        plt.hist(df['flow'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Traffic Flow Distribution')
        plt.xlabel('Flow (vehicles/5min)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # 2. 小时模式
        plt.subplot(2, 3, 2)
        hourly_flow = df.groupby('hour')['flow'].mean()
        hourly_flow.plot(kind='bar', color='lightcoral', alpha=0.7)
        plt.title('Average Flow by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Flow')
        plt.grid(True, alpha=0.3)

        # 3. 周模式 - 修复：根据实际天数动态设置标签
        plt.subplot(2, 3, 3)
        daily_flow = df.groupby('day_of_week')['flow'].mean()

        # 动态设置天数标签
        days_available = len(daily_flow)
        if days_available == 7:
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        elif days_available == 2:
            # 我们只有2天的数据（1月1日和2日）
            day_labels = ['Tue', 'Wed']  # 2019-01-01是周二，01-02是周三
        else:
            day_labels = [f'Day {i}' for i in range(days_available)]

        daily_flow.index = day_labels
        daily_flow.plot(kind='bar', color='lightgreen', alpha=0.7)
        plt.title('Average Flow by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Flow')
        plt.grid(True, alpha=0.3)

        # 4. 传感器地理位置（如果有坐标）
        if 'latitude' in df.columns and 'longitude' in df.columns:
            plt.subplot(2, 3, 4)
            sensor_locations = df.groupby('sensor_id')[['latitude', 'longitude']].first()
            plt.scatter(sensor_locations['longitude'], sensor_locations['latitude'],
                        alpha=0.6, s=10, c='blue')
            plt.title('Sensor Locations')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)

        # 5. 时间序列样本
        plt.subplot(2, 3, 5)
        sample_sensor = df['sensor_id'].iloc[0]
        sensor_data = df[df['sensor_id'] == sample_sensor].head(200)
        plt.plot(sensor_data['timestamp'], sensor_data['flow'], 'b-', linewidth=1)
        plt.title(f'Flow Time Series (Sensor: {sample_sensor})')
        plt.xlabel('Time')
        plt.ylabel('Flow')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 6. 相关性热力图
        plt.subplot(2, 3, 6)
        numeric_cols = ['flow', 'hour', 'day_of_week', 'is_weekend']
        if 'latitude' in df.columns:
            numeric_cols.extend(['latitude', 'longitude'])

        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f')
        plt.title('Feature Correlations')

        plt.tight_layout()
        plt.savefig('real_california_insights.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Real data analysis completed!")
        print(f"   Visualization saved to: real_california_insights.png")

        return df

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    analyze_real_data_insights()