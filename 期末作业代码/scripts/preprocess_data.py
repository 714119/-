# scripts/preprocess_data.py
import sys
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录

# 添加项目根目录到Python路径
sys.path.append(project_root)

from src.data.preprocessor import TennisDataPreprocessor


def main():
    """主函数：运行数据预处理"""

    # 初始化预处理器
    preprocessor = TennisDataPreprocessor()

    # 正确的文件路径（相对于项目根目录）
    input_file = os.path.join(project_root, 'data', 'raw', '2024_Wimbledon_featured_matches.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    output_file = os.path.join(output_dir, 'tennis_matches_processed.csv')

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 未找到数据文件: {input_file}")
        print("请检查文件是否存在。")
        return

    print(f"✅ 找到数据文件: {input_file}")
    print(f"📂 输出目录: {output_dir}")
    print(f"💾 输出文件: {output_file}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 运行预处理
    try:
        print("\n" + "=" * 60)
        print("开始数据预处理...")
        print("=" * 60)

        processed_df = preprocessor.preprocess(
            input_path=input_file,
            output_path=output_file
        )

        print("\n" + "=" * 60)
        print("✅ 预处理成功完成！")
        print("=" * 60)

        # 显示一些统计信息
        print(f"\n📊 处理结果统计:")
        print(f"  原始文件: {input_file}")
        print(f"  处理后文件: {output_file}")
        print(f"  数据形状: {processed_df.shape}")
        print(f"  比赛数量: {processed_df['match_id'].nunique()}")
        print(f"  总点数: {len(processed_df)}")
        print(f"  特征数量: {len(processed_df.columns)}")

        # 目标变量分布
        target_mean = processed_df['target'].mean()
        print(f"\n🎯 预测目标分布:")
        print(f"  Player1赢下一分: {processed_df['target'].sum():,}次 ({target_mean:.1%})")
        print(f"  Player2赢下一分: {(1 - processed_df['target']).sum():,}次 ({(1 - target_mean):.1%})")

        # 检查一些重要特征
        print(f"\n🔍 重要特征示例:")
        important_features = [
            'score_diff', 'is_server_p1', 'is_break_point',
            'p1_win_rate_5', 'match_progress', 'games_diff'
        ]
        for feat in important_features:
            if feat in processed_df.columns:
                print(
                    f"  {feat}: 均值={processed_df[feat].mean():.3f}, 范围=[{processed_df[feat].min():.3f}, {processed_df[feat].max():.3f}]")

    except Exception as e:
        print(f"\n❌ 预处理失败!")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()