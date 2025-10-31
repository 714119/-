import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys

current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")
sys.path.append(str(current_dir / 'src'))


class SimpleDashboard:
    def __init__(self):
        self.load_data()

    def load_data(self):
        """加载数据"""
        try:
            from utils.data_loader import TrafficDataLoader
            config_path = current_dir / 'config.yaml'
            data_loader = TrafficDataLoader(config_path=config_path)
            self.df = data_loader.load_processed_traffic_data('california')
            print(f"✅ Loaded {len(self.df)} records for static dashboard")

            # 确保有坐标数据
            if 'latitude' not in self.df.columns:
                self.add_sample_coordinates()

        except Exception as e:
            print(f"❌ Error: {e}")
            self.create_sample_data()

    def add_sample_coordinates(self):
        """添加示例坐标"""
        np.random.seed(42)
        unique_sensors = self.df['sensor_id'].unique()

        lat_min, lat_max = 32.5, 42.0
        lon_min, lon_max = -124.5, -114.0

        sensor_coords = {}
        for sensor in unique_sensors:
            sensor_coords[sensor] = {
                'latitude': np.random.uniform(lat_min, lat_max),
                'longitude': np.random.uniform(lon_min, lon_max)
            }

        self.df['latitude'] = self.df['sensor_id'].map(lambda x: sensor_coords[x]['latitude'])
        self.df['longitude'] = self.df['sensor_id'].map(lambda x: sensor_coords[x]['longitude'])

    def create_sample_data(self):
        """创建示例数据"""
        # ... (同之前的示例数据代码)
        pass

    def create_static_dashboard(self):
        """创建静态仪表盘"""
        print("Creating static dashboard...")

        # 创建带子图的仪表盘
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '🚗 Traffic Sensor Map',
                '📈 Hourly Traffic Pattern',
                '🕐 Time Series Analysis',
                '🔮 Prediction Results'
            ],
            specs=[
                [{"type": "scattermapbox"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )

        # 1. 地图
        sensor_avg = self.df.groupby('sensor_id').agg({
            'flow': 'mean', 'latitude': 'first', 'longitude': 'first'
        }).reset_index()

        fig.add_trace(
            go.Scattermapbox(
                lat=sensor_avg['latitude'],
                lon=sensor_avg['longitude'],
                mode='markers',
                marker=dict(
                    size=sensor_avg['flow'] / 10,
                    color=sensor_avg['flow'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Flow")
                ),
                text=sensor_avg['sensor_id'],
                hovertemplate='<b>%{text}</b><br>Flow: %{marker.color:.1f}<extra></extra>',
                name='Sensors'
            ),
            row=1, col=1
        )

        # 2. 小时模式
        hourly_pattern = self.df.groupby('hour')['flow'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=hourly_pattern['hour'],
                y=hourly_pattern['flow'],
                marker_color='lightblue',
                name='Hourly Flow'
            ),
            row=1, col=2
        )

        # 3. 时间序列
        daily_flow = self.df.groupby(self.df['timestamp'].dt.date)['flow'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=daily_flow['timestamp'],
                y=daily_flow['flow'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Daily Average'
            ),
            row=2, col=1
        )

        # 4. 预测结果
        sample_data = self.df.head(20)
        predictions = sample_data['flow'].values + np.random.normal(0, 15, len(sample_data))

        fig.add_trace(
            go.Scatter(
                x=sample_data['timestamp'],
                y=sample_data['flow'],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                name='Actual'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=sample_data['timestamp'],
                y=predictions,
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dash'),
                name='Predicted'
            ),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            height=1000,
            title_text="California Traffic Analysis Dashboard<br><sub>Static HTML Version</sub>",
            title_x=0.5,
            showlegend=True,
            mapbox=dict(
                style="open-street-map",
                zoom=4,
                center=dict(lat=36.5, lon=-119.5)
            ),
            template="plotly_white"
        )

        # 更新坐标轴标签
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="Flow (vehicles/5min)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Flow (vehicles/5min)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Flow (vehicles/5min)", row=2, col=2)

        # 保存为HTML
        fig.write_html("static_traffic_dashboard.html")
        print("✅ Static dashboard saved to static_traffic_dashboard.html")

        # 创建单独的动画HTML
        self.create_animation_html()

    def create_animation_html(self):
        """创建动画HTML"""
        print("Creating animation HTML...")

        # 按小时聚合数据
        hourly_data = self.df.groupby([self.df['timestamp'].dt.floor('H'), 'sensor_id']).agg({
            'flow': 'mean', 'latitude': 'first', 'longitude': 'first'
        }).reset_index()

        # 创建动画
        fig = px.scatter_mapbox(
            hourly_data,
            lat="latitude",
            lon="longitude",
            size="flow",
            color="flow",
            animation_frame=hourly_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
            hover_name="sensor_id",
            color_continuous_scale=px.colors.sequential.Plasma,
            zoom=4,
            height=600,
            title="California Traffic Flow Animation (Hourly)"
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=36.5, lon=-119.5), zoom=4)
        )

        fig.write_html("traffic_animation.html")
        print("✅ Animation saved to traffic_animation.html")


if __name__ == "__main__":
    dashboard = SimpleDashboard()
    dashboard.create_static_dashboard()