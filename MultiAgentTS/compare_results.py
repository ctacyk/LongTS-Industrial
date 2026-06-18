#!/usr/bin/env python3
"""
对比原始版和优化版的数据生成效果
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

print("="*80)
print("磨煤机数据生成对比分析")
print("="*80)

# 读取原始生成数据
print("\n[1] 读取原始生成数据...")
original_data = pd.read_csv("Data/CoalMill/synthetic_coal_mill_30days.csv")
print(f"    原始数据: {len(original_data)} 条记录")

# 读取优化生成数据
print("[2] 读取优化生成数据...")
optimized_data = pd.read_csv("MultiAgentTS/visualizations/optimized_data_sample.csv")
print(f"    优化数据: {len(optimized_data)} 条记录")

# 只对比前1000个样本
sample_size = 1000
orig_sample = original_data.head(sample_size).reset_index(drop=True)
opt_sample = optimized_data.head(sample_size).reset_index(drop=True)

# 创建对比图
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

columns_map = [
    ('Motor_Current_A', 'Current (A)'),
    ('Winding_Temp_1_C', 'Winding Temp 1 (°C)'),
    ('Winding_Temp_2_C', 'Winding Temp 2 (°C)'),
    ('Bearing_Temp_NDS_C', 'Bearing NDS (°C)'),
    ('Bearing_Temp_DS_C', 'Bearing DS (°C)')
]

x_orig = np.arange(sample_size)

print("\n[3] 绘制对比图表...")
print("\n    数据统计对比:")
print("    " + "-"*75)
print(f"    {'指标':<30} {'原始生成':<20} {'优化生成':<20}")
print("    " + "-"*75)

for idx, (col, label) in enumerate(columns_map):
    # 原始版
    ax1 = fig.add_subplot(gs[idx, 0])
    ax1.plot(x_orig, orig_sample[col], linewidth=1, color='#d62728', alpha=0.7)
    ax1.set_ylabel(label, fontsize=9)
    ax1.set_title(f"原始版 - {label}", fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, sample_size)
    
    # 优化版
    ax2 = fig.add_subplot(gs[idx, 1])
    ax2.plot(x_orig, opt_sample[col], linewidth=1, color='#1f77b4', alpha=0.7)
    ax2.set_ylabel(label, fontsize=9)
    ax2.set_title(f"优化版 - {label}", fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, sample_size)
    
    # 统计对比
    orig_mean = orig_sample[col].mean()
    orig_std = orig_sample[col].std()
    opt_mean = opt_sample[col].mean()
    opt_std = opt_sample[col].std()
    
    print(f"    {col:<30} Mean:{orig_mean:7.2f}±{orig_std:5.2f}  Mean:{opt_mean:7.2f}±{opt_std:5.2f}")

print("    " + "-"*75)

fig.suptitle('磨煤机数据生成对比 - 原始版 vs 优化版 (前1000样本)', 
             fontsize=14, fontweight='bold', y=0.995)

out_dir = "MultiAgentTS/visualizations"
os.makedirs(out_dir, exist_ok=True)

fig.savefig(os.path.join(out_dir, "original_vs_optimized.png"), dpi=150, bbox_inches='tight')
print(f"\n[4] 已保存对比图: {os.path.join(out_dir, 'original_vs_optimized.png')}")

# 计算平滑度指标 - 使用一阶差分的方差来衡量
print("\n[5] 平滑度分析 (一阶差分标准差):")
print("    " + "-"*60)
print(f"    {'指标':<30} {'原始版':<15} {'优化版':<15} {'改进':<10}")
print("    " + "-"*60)

for col, label in columns_map:
    orig_diff_std = np.diff(orig_sample[col]).std()
    opt_diff_std = np.diff(opt_sample[col]).std()
    improve_pct = ((orig_diff_std - opt_diff_std) / orig_diff_std * 100) if orig_diff_std > 0 else 0
    
    print(f"    {col:<30} {orig_diff_std:>14.4f} {opt_diff_std:>14.4f} {improve_pct:>8.1f}%")

print("    " + "-"*60)
print("\n    注: 数值越小表示越平滑,改进百分比越高表示优化效果越好")

print("\n" + "="*80)
print("[完成] 对比分析完成!")
print("="*80)

plt.show()

