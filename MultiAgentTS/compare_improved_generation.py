"""
Comparison visualization between real and improved generated data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# Set Chinese font
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# Load data
print("Loading data...")
df_real = pd.read_csv('../aligned_output/merged.csv')
df_gen = pd.read_csv('Data/CoalMill/synthetic_coal_mill_improved.csv')

# Convert timestamps
df_real['timestamp'] = pd.to_datetime(df_real['timestamp'])
df_gen['timestamp'] = pd.to_datetime(df_gen['timestamp'])

# Resample for cleaner visualization (every 12 minutes instead of 30s)
sample_rate = 24  # Every 24 points = 12 minutes
df_real_plot = df_real.iloc[::sample_rate].reset_index(drop=True)
df_gen_plot = df_gen.iloc[::sample_rate].reset_index(drop=True)

print(f"Real data: {len(df_real)} points → {len(df_real_plot)} sampled")
print(f"Generated data: {len(df_gen)} points → {len(df_gen_plot)} sampled")

# Create comparison plot
fig, axes = plt.subplots(7, 1, figsize=(16, 14))
fig.suptitle('Coal Mill Data Comparison: Real vs Generated (Improved)', fontsize=16, fontweight='bold')

columns = ['电机电流', '电机绕组温度1', '电机绕组温度2', 
           '电机非驱动侧轴承温度', '电机驱动侧轴承温度', '磨机入口温度', '磨机入口调节阀开度']

for idx, col in enumerate(columns):
    ax = axes[idx]
    
    # Plot both signals
    ax.plot(df_real_plot.index, df_real_plot[col], label='Real Data', 
            color='blue', linewidth=1, alpha=0.8)
    ax.plot(df_gen_plot.index, df_gen_plot[col], label='Generated (Improved)', 
            color='red', linewidth=1, alpha=0.7)
    
    ax.set_ylabel(col, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add statistics to title
    real_mean = df_real[col].mean()
    real_std = df_real[col].std()
    gen_mean = df_gen[col].mean()
    gen_std = df_gen[col].std()
    
    stats_text = f"Real: μ={real_mean:.1f}±{real_std:.1f} | Gen: μ={gen_mean:.1f}±{gen_std:.1f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/improved_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison plot: visualizations/improved_comparison.png")

# Create detailed analysis plot
fig, axes = plt.subplots(7, 2, figsize=(18, 14))
fig.suptitle('Detailed Statistical Comparison', fontsize=16, fontweight='bold')

for idx, col in enumerate(columns):
    # Distribution comparison
    ax1 = axes[idx, 0]
    ax1.hist(df_real[col], bins=50, alpha=0.6, label='Real', color='blue', density=True)
    ax1.hist(df_gen[col], bins=50, alpha=0.6, label='Generated', color='red', density=True)
    ax1.set_xlabel(col, fontsize=9)
    ax1.set_ylabel('Density', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time series autocorrelation
    ax2 = axes[idx, 1]
    real_sample = df_real[col].iloc[::100]
    gen_sample = df_gen[col].iloc[::100]
    
    # Compute autocorrelation
    from scipy.stats import pearsonr
    real_ac = [1] + [pearsonr(real_sample.iloc[:-i], real_sample.iloc[i:])[0] for i in range(1, 51)]
    gen_ac = [1] + [pearsonr(gen_sample.iloc[:-i], gen_sample.iloc[i:])[0] for i in range(1, 51)]
    
    ax2.plot(real_ac, label='Real', marker='o', color='blue', markersize=3)
    ax2.plot(gen_ac, label='Generated', marker='s', color='red', markersize=3)
    ax2.set_xlabel('Lag', fontsize=9)
    ax2.set_ylabel('Autocorrelation', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/detailed_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved detailed analysis: visualizations/detailed_analysis.png")

print("\n=== Statistical Comparison ===")
for col in columns:
    real_mean = df_real[col].mean()
    real_std = df_real[col].std()
    gen_mean = df_gen[col].mean()
    gen_std = df_gen[col].std()
    
    print(f"\n{col}:")
    print(f"  Real:      Mean={real_mean:.2f}, Std={real_std:.2f}, Range=[{df_real[col].min():.2f}, {df_real[col].max():.2f}]")
    print(f"  Generated: Mean={gen_mean:.2f}, Std={gen_std:.2f}, Range=[{df_gen[col].min():.2f}, {df_gen[col].max():.2f}]")

