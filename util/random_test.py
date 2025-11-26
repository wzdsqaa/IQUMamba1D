import pandas as pd
import numpy as np
import glob
import os

def combine_experiment_results(csv_files, output_file='combined_results.csv'):
    """
    综合多次随机种子实验的结果
    
    Parameters:
    csv_files: list of str, CSV文件路径列表
    output_file: str, 输出文件名
    
    Returns:
    pd.DataFrame: 综合结果
    """
    
    # 存储所有实验的数据
    all_data = []
    
    # 读取所有CSV文件
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            df['experiment'] = i + 1  # 添加实验编号
            all_data.append(df)
            print(f"成功读取实验 {i+1}: {file_path}")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    if not all_data:
        raise ValueError("没有成功读取任何文件")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按SNR分组计算统计量
    metrics = ['Loss', 'Correlation', 'SI-SDR', 'SI-SIR', 'SI-SAR']
    
    # 计算均值和标准差
    stats_mean = combined_df.groupby('SNR')[metrics].mean()
    stats_std = combined_df.groupby('SNR')[metrics].std()
    
    # 创建结果DataFrame
    result_df = pd.DataFrame()
    result_df['SNR'] = stats_mean.index
    
    # 添加均值列
    for metric in metrics:
        result_df[f'{metric}_mean'] = stats_mean[metric].values
        result_df[f'{metric}_std'] = stats_std[metric].values
        # 计算95%置信区间的半宽度 (假设正态分布)
        n_experiments = len(csv_files)
        sem = stats_std[metric] / np.sqrt(n_experiments)  # 标准误
        ci_half_width = 1.96 * sem  # 95%置信区间
        result_df[f'{metric}_ci95'] = ci_half_width.values
    
    # 重置索引
    result_df = result_df.reset_index(drop=True)
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"\n综合结果已保存到: {output_file}")
    
    return result_df

def display_summary_table(result_df):
    """显示结果摘要表"""
    print("\n=== 实验结果摘要 ===")
    print("格式: 均值 ± 标准差 [95%置信区间]")
    print("-" * 80)
    
    metrics = ['Loss', 'Correlation', 'SI-SDR', 'SI-SIR', 'SI-SAR']
    
    for _, row in result_df.iterrows():
        print(f"\nSNR = {row['SNR']:.1f} dB:")
        for metric in metrics:
            mean_val = row[f'{metric}_mean']
            std_val = row[f'{metric}_std']
            ci_val = row[f'{metric}_ci95']
            print(f"  {metric:12}: {mean_val:8.4f} ± {std_val:6.4f} [±{ci_val:6.4f}]")


def plot_results(result_df, save_plots=True):
    """
    绘制结果图表
    """
    import matplotlib.pyplot as plt
    
    metrics = ['Loss', 'Correlation', 'SI-SDR', 'SI-SIR', 'SI-SAR']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        x = result_df['SNR']
        y = result_df[f'{metric}_mean']
        yerr = result_df[f'{metric}_std']
        
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs SNR')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
        print("图表已保存到: experiment_results.png")
    
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 方法1: 手动指定文件列表
    csv_files = [
        '/root/IQUMamba1D/results/results_0_seed_42/metrics_summary.csv',
        '/root/IQUMamba1D/results/results_0_seed_43/metrics_summary.csv', 
        '/root/IQUMamba1D/results/results_0_seed_44/metrics_summary.csv',
        '/root/IQUMamba1D/results/results_0_seed_45/metrics_summary.csv',
        '/root/IQUMamba1D/results/results_0_seed_46/metrics_summary.csv'
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if len(existing_files) < 2:
        print("错误: 找不到足够的CSV文件")
        print("请确保CSV文件存在于当前目录中")
        print(f"当前查找的文件: {csv_files}")
    else:
        print(f"找到 {len(existing_files)} 个实验文件:")
        for f in existing_files:
            print(f"  - {f}")
        
        # 综合结果
        try:
            result_df = combine_experiment_results(existing_files)
            
            # 显示摘要
            display_summary_table(result_df)
            
            # 可选: 创建详细的统计报告
            print(f"\n详细结果已保存到 combined_results.csv")
            
        except Exception as e:
            print(f"处理过程中出错: {e}")

    # 创建可视化图表，可以添加以下函数
    plot_results(result_df)