import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
current_dir = Path(r"E:\NJU\大数据系统原理与运用\traffic-analysis-project\traffic-analysis-project")
sys.path.append(str(current_dir / 'src'))


class TrafficDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.load_data()
        self.setup_layout()
        self.setup_callbacks()

    def load_data(self):
        """加载处理后的数据"""
        try:
            from utils.data_loader import TrafficDataLoader
            config_path = current_dir / 'config.yaml'
            data_loader = TrafficDataLoader(config_path=config_path)
            self.df = data_loader.load_processed_traffic_data('california')
            print(f"✅ Loaded {len(self.df)} records for dashboard")

            # 确保数据有地理位置信息
            if 'latitude' not in self.df.columns or 'longitude' not in self.df.columns:
                print("⚠️ Adding sample coordinates...")
                self.add_sample_coordinates()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.create_sample_data()

    def add_sample_coordinates(self):
        """添加示例坐标（加州范围）"""
        np.random.seed(42)
        unique_sensors = self.df['sensor_id'].unique()

        # 加州大致坐标范围
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
        print("Creating sample data for dashboard...")
        dates = pd.date_range('2019-01-01', '2019-01-07', freq='5T')
        sensors = [f'sensor_{i:03d}' for i in range(1, 11)]

        # 加州大致坐标范围
        lat_min, lat_max = 32.5, 42.0
        lon_min, lon_max = -124.5, -114.0

        data = []
        for sensor in sensors:
            sensor_lat = np.random.uniform(lat_min, lat_max)
            sensor_lon = np.random.uniform(lon_min, lon_max)

            for date in dates:
                # 模拟早晚高峰模式
                hour = date.hour
                if 7 <= hour < 9 or 17 <= hour < 19:
                    flow = np.random.randint(200, 600)
                else:
                    flow = np.random.randint(50, 200)

                data.append({
                    'sensor_id': sensor,
                    'timestamp': date,
                    'flow': flow,
                    'latitude': sensor_lat,
                    'longitude': sensor_lon,
                    'hour': hour,
                    'day_of_week': date.dayofweek
                })

        self.df = pd.DataFrame(data)
        print("✅ Created sample data for dashboard")

    def setup_layout(self):
        """设置仪表盘布局"""
        self.app.layout = html.Div([
            # 标题
            html.H1("🚗 California Traffic Flow Analysis Dashboard",
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

            # 控制面板
            html.Div([
                html.Div([
                    html.Label("📅 Date Range:"),
                    dcc.DatePickerRange(
                        id='date-picker',
                        start_date=self.df['timestamp'].min().date(),
                        end_date=self.df['timestamp'].max().date(),
                        display_format='YYYY-MM-DD'
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("🕒 Time of Day:"),
                    dcc.RangeSlider(
                        id='hour-slider',
                        min=0, max=23, value=[7, 19],
                        marks={i: f'{i:02d}:00' for i in range(0, 24, 3)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
            ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

            # 图表区域
            html.Div([
                # 地图
                html.Div([
                    html.H3("🗺️ Traffic Sensor Map", style={'textAlign': 'center'}),
                    dcc.Graph(id='traffic-map')
                ], style={'width': '48%', 'display': 'inline-block'}),

                # 时间序列
                html.Div([
                    html.H3("📈 Traffic Flow Over Time", style={'textAlign': 'center'}),
                    dcc.Graph(id='time-series-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
            ]),

            # 第二行图表
            html.Div([
                # 小时模式
                html.Div([
                    html.H3("🕐 Hourly Traffic Pattern", style={'textAlign': 'center'}),
                    dcc.Graph(id='hourly-pattern')
                ], style={'width': '48%', 'display': 'inline-block'}),

                # 预测结果
                html.Div([
                    html.H3("🔮 Traffic Flow Prediction", style={'textAlign': 'center'}),
                    dcc.Graph(id='prediction-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
            ], style={'marginTop': 30}),

            # 动画控制
            html.Div([
                html.H3("🎬 Traffic Flow Animation", style={'textAlign': 'center'}),
                html.Div([
                    html.Button('▶️ Play Animation', id='play-button', n_clicks=0,
                                style={'marginRight': '10px'}),
                    dcc.Slider(
                        id='animation-slider',
                        min=0,
                        max=287,  # 24小时 * 12个5分钟间隔 - 1
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ]),
                dcc.Graph(id='animation-chart'),
                dcc.Interval(
                    id='animation-interval',
                    interval=1000,  # 1秒更新
                    n_intervals=0,
                    disabled=True
                ),
            ], style={'marginTop': 30, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

            # 数据导出
            html.Div([
                html.Hr(),
                html.Button('📥 Export Filtered Data', id='export-button', n_clicks=0,
                            style={'padding': '10px 20px', 'fontSize': '16px'}),
                dcc.Download(id="download-dataframe-csv")
            ], style={'marginTop': 30, 'textAlign': 'center'})
        ], style={'padding': '20px'})

    def setup_callbacks(self):
        """设置交互回调"""

        @self.app.callback(
            [Output('traffic-map', 'figure'),
             Output('time-series-chart', 'figure'),
             Output('hourly-pattern', 'figure'),
             Output('prediction-chart', 'figure')],
            [Input('date-picker', 'start_date'),
             Input('date-picker', 'end_date'),
             Input('hour-slider', 'value')]
        )
        def update_charts(start_date, end_date, hour_range):
            """更新主要图表"""
            filtered_df = self.filter_data(start_date, end_date, hour_range)

            # 1. 地图
            map_fig = self.create_map(filtered_df)

            # 2. 时间序列
            time_fig = self.create_time_series(filtered_df)

            # 3. 小时模式
            hourly_fig = self.create_hourly_pattern(filtered_df)

            # 4. 预测图表
            prediction_fig = self.create_prediction_chart(filtered_df)

            return map_fig, time_fig, hourly_fig, prediction_fig

        @self.app.callback(
            Output('animation-chart', 'figure'),
            [Input('animation-slider', 'value')]
        )
        def update_animation(frame):
            """更新动画帧"""
            return self.create_animation_frame(frame)

        @self.app.callback(
            [Output('animation-interval', 'disabled'),
             Output('play-button', 'children')],
            [Input('play-button', 'n_clicks')],
            [dash.dependencies.State('animation-interval', 'disabled')]
        )
        def toggle_animation(n_clicks, is_disabled):
            """播放/暂停动画"""
            if n_clicks % 2 == 1:
                return False, '⏸️ Pause Animation'
            else:
                return True, '▶️ Play Animation'

        @self.app.callback(
            Output('animation-slider', 'value'),
            [Input('animation-interval', 'n_intervals')],
            [dash.dependencies.State('animation-slider', 'value')]
        )
        def update_slider(n_intervals, current_value):
            """自动更新滑块"""
            if n_intervals is None:
                return current_value
            return (current_value + 1) % 288

        @self.app.callback(
            Output("download-dataframe-csv", "data"),
            [Input("export-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def export_data(n_clicks):
            """导出数据"""
            return dcc.send_data_frame(self.df.to_csv, "california_traffic_data.csv")

    def filter_data(self, start_date, end_date, hour_range):
        """过滤数据"""
        filtered_df = self.df.copy()

        if start_date:
            filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(end_date)]

        if hour_range:
            filtered_df = filtered_df[
                (filtered_df['hour'] >= hour_range[0]) &
                (filtered_df['hour'] <= hour_range[1])
                ]

        return filtered_df

    def create_map(self, df):
        """创建地图"""
        if df.empty:
            return go.Figure()

        # 计算每个传感器的平均流量
        sensor_avg = df.groupby('sensor_id').agg({
            'flow': 'mean',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()

        fig = px.scatter_mapbox(
            sensor_avg,
            lat="latitude",
            lon="longitude",
            size="flow",
            color="flow",
            hover_name="sensor_id",
            hover_data={"flow": ":.1f"},
            color_continuous_scale=px.colors.sequential.Viridis,
            zoom=5,
            height=400,
            title="Traffic Sensor Locations and Flow Intensity"
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=36.5, lon=-119.5),  # 加州中心
                zoom=5
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        return fig

    def create_time_series(self, df):
        """创建时间序列图"""
        if df.empty:
            return go.Figure()

        # 按时间聚合
        hourly_flow = df.groupby(df['timestamp'].dt.floor('H'))['flow'].mean().reset_index()

        fig = px.line(
            hourly_flow,
            x='timestamp',
            y='flow',
            title='Average Traffic Flow Over Time'
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Flow (vehicles/5min)",
            height=400
        )

        return fig

    def create_hourly_pattern(self, df):
        """创建小时模式图"""
        if df.empty:
            return go.Figure()

        hourly_pattern = df.groupby('hour')['flow'].mean().reset_index()

        fig = px.bar(
            hourly_pattern,
            x='hour',
            y='flow',
            title='Average Traffic Flow by Hour of Day'
        )

        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Flow",
            height=400
        )

        return fig

    def create_prediction_chart(self, df):
        """创建预测图表"""
        # 使用真实数据的前面部分作为"预测"展示
        if len(df) < 10:
            return go.Figure()

        sample_data = df.head(50)
        # 模拟预测结果
        predictions = sample_data['flow'].values + np.random.normal(0, 10, len(sample_data))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_data['timestamp'],
            y=sample_data['flow'],
            mode='lines+markers',
            name='Actual Flow',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=sample_data['timestamp'],
            y=predictions,
            mode='lines+markers',
            name='Predicted Flow',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='Traffic Flow Prediction vs Actual',
            xaxis_title="Time",
            yaxis_title="Flow (vehicles/5min)",
            height=400
        )

        return fig

    def create_animation_frame(self, frame):
        """创建动画帧"""
        # 计算当前时间点（从数据开始时间）
        base_time = self.df['timestamp'].min()
        current_time = base_time + pd.Timedelta(minutes=5 * frame)

        # 过滤当前时间点的数据
        current_data = self.df[
            self.df['timestamp'].dt.floor('5T') == current_time.floor('5T')
            ]

        if current_data.empty:
            # 如果没有精确匹配，找最接近的
            time_diff = abs(self.df['timestamp'] - current_time)
            closest_idx = time_diff.idxmin()
            current_data = self.df.loc[[closest_idx]]

        fig = px.scatter_mapbox(
            current_data,
            lat="latitude",
            lon="longitude",
            size="flow",
            color="flow",
            hover_name="sensor_id",
            hover_data={"flow": ":.1f"},
            color_continuous_scale=px.colors.sequential.Plasma,
            zoom=5,
            height=400,
            title=f"Traffic Flow at {current_time.strftime('%Y-%m-%d %H:%M')}"
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=36.5, lon=-119.5),
                zoom=5
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        return fig

    def run(self, debug=True, port=8050):
        """运行仪表盘"""
        print(f"🚀 Starting dashboard on http://localhost:{port}")
        print("📊 Dashboard features:")
        print("   - Interactive map with traffic sensors")
        print("   - Time series analysis")
        print("   - Hourly pattern visualization")
        print("   - Traffic flow animation")
        print("   - Data export functionality")

        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    dashboard = TrafficDashboard()
    dashboard.run()