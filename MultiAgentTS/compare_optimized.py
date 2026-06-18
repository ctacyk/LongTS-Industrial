#!/usr/bin/env python3
"""
磨煤机数据生成对比脚本 - 优化版
对比原始生成和优化生成的效果
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from time_blender.core import Event, LambdaEvent, ConstantEvent, Generator
from time_blender.random_events import NormalEvent
from time_blender.coordination_events import SeasonalEvent
from time_blender.util import set_random_seed

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

set_random_seed(42)

# ============= 优化事件类 =============
class ExpoSmooth(Event):
    def __init__(self, src, alpha=0.12, name=None, parallel_events=None):
        super().__init__(name, parallel_events, push_down=True, allow_learning=False)
        self.source = src
        self.alpha = alpha
        self.val = None
        self._causal_parameters.append(src)
    
    def _execute(self, t, i):
        v = self.source.execute(t)
        if self.val is None: self.val = v
        else: self.val = self.alpha * v + (1 - self.alpha) * self.val
        return self.val
    
    def reset(self):
        self.val = None
        super().reset()

class StrongAR(Event):
    def __init__(self, coef=0.94, scale=0.3, name=None, parallel_events=None):
        super().__init__(name, parallel_events, push_down=True, allow_learning=False)
        self.coef = coef
        self.scale = scale
        self.v = 0.0
    
    def _execute(self, t, i):
        self.v = self.coef * self.v + np.random.normal(0, self.scale)
        return self.v
    
    def reset(self):
        self.v = 0.0
        super().reset()

class HardClip(Event):
    def __init__(self, src, mn, mx, name=None, parallel_events=None):
        super().__init__(name, parallel_events, push_down=True, allow_learning=False)
        self.source = src
        self.mn, self.mx = mn, mx
        self._causal_parameters.append(src)
    
    def _execute(self, t, i):
        return np.clip(self.source.execute(t), self.mn, self.mx)
    
    def reset(self):
        super().reset()

def make_optimized_model():
    """创建优化的磨煤机模型"""
    
    # 电机电流
    current_base = ConstantEvent(50.0)
    def daily_cycle(t, i, m, s):
        phase = (i % 2880) / 2880.0 * 2 * np.pi
        return 3.5 * np.sin(phase)
    daily = LambdaEvent(daily_cycle, sub_events={})
    current_ar = StrongAR(0.95, 0.35)
    weekend = SeasonalEvent(ConstantEvent(-2.5), is_weekend=True, default=0)
    noise = NormalEvent(0, 0.3)
    
    def combine_cur(t, i, m, s):
        return (s['base'].execute(t) + s['daily'].execute(t) + s['ar'].execute(t) + 
                s['weekend'].execute(t) + s['noise'].execute(t))
    cur_comb = LambdaEvent(combine_cur, sub_events={
        'base': current_base, 'daily': daily, 'ar': current_ar,
        'weekend': weekend, 'noise': noise})
    motor_current = HardClip(cur_comb, 40.0, 62.0)
    
    # 绕组温度1
    def wind1_base(t, i, m, s):
        return 38.0 + (s['current'].execute(t) - 40.0) * 0.35
    wind1_base = LambdaEvent(wind1_base, sub_events={'current': motor_current})
    wind1_lag = ExpoSmooth(wind1_base, 0.10)
    wind1_ar = StrongAR(0.93, 0.1)
    wind1_noise = NormalEvent(0, 0.08)
    
    def comb_wind(t, i, m, s):
        return s['lag'].execute(t) + s['ar'].execute(t) + s['noise'].execute(t)
    wind1_comb = LambdaEvent(comb_wind, sub_events={'lag': wind1_lag, 'ar': wind1_ar, 'noise': wind1_noise})
    winding_temp_1 = HardClip(wind1_comb, 38.0, 58.0)
    
    # 绕组温度2
    wind2_ar = StrongAR(0.92, 0.1)
    wind2_noise = NormalEvent(0, 0.08)
    def comb_wind2(t, i, m, s):
        return s['lag'].execute(t) + 1.5 + s['ar'].execute(t) + s['noise'].execute(t)
    wind2_comb = LambdaEvent(comb_wind2, sub_events={'lag': wind1_lag, 'ar': wind2_ar, 'noise': wind2_noise})
    winding_temp_2 = HardClip(wind2_comb, 39.0, 60.0)
    
    # NDS轴承温度
    def nds_base(t, i, m, s):
        return 42.5 + (s['current'].execute(t) - 40.0) * 0.25
    nds_base = LambdaEvent(nds_base, sub_events={'current': motor_current})
    nds_lag = ExpoSmooth(nds_base, 0.08)
    nds_ar = StrongAR(0.92, 0.12)
    nds_noise = NormalEvent(0, 0.1)
    nds_comb = LambdaEvent(comb_wind, sub_events={'lag': nds_lag, 'ar': nds_ar, 'noise': nds_noise})
    nds_temp = HardClip(nds_comb, 43.0, 58.0)
    
    # DS轴承温度
    def ds_base(t, i, m, s):
        return 50.0 + (s['current'].execute(t) - 40.0) * 0.3
    ds_base = LambdaEvent(ds_base, sub_events={'current': motor_current})
    ds_lag = ExpoSmooth(ds_base, 0.09)
    ds_ar = StrongAR(0.91, 0.12)
    ds_noise = NormalEvent(0, 0.1)
    ds_comb = LambdaEvent(comb_wind, sub_events={'lag': ds_lag, 'ar': ds_ar, 'noise': ds_noise})
    ds_temp = HardClip(ds_comb, 50.0, 65.0)
    
    return {
        'Motor_Current_A': motor_current,
        'Winding_Temp_1_C': winding_temp_1,
        'Winding_Temp_2_C': winding_temp_2,
        'Bearing_Temp_NDS_C': nds_temp,
        'Bearing_Temp_DS_C': ds_temp,
    }

def generate_data(model, days=30):
    """生成数据"""
    start = pd.Timestamp('2024-01-01 00:00:00')
    end = start + pd.Timedelta(days=days)
    
    gen = Generator(start_date=start, end_date=end, freq='30S')
    data = gen.generate(model, n=1)
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Timestamp'}, inplace=True)
    return data

def plot_comparison(data, title="磨煤机数据 - 优化版"):
    """绘制对比图"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    columns = ['Motor_Current_A', 'Winding_Temp_1_C', 'Bearing_Temp_NDS_C', 'Bearing_Temp_DS_C', 'Winding_Temp_2_C']
    
    for idx, (ax, col) in enumerate(zip(axes, columns)):
        ax.plot(data['Timestamp'], data[col], label=col, linewidth=1.5, color='#1f77b4')
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(f"{col} - 时间序列", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

print("="*70)
print("磨煤机数据生成 - 优化版")
print("="*70)

print("\n[1] 创建优化模型...")
model = make_optimized_model()

print("[2] 生成30天数据...")
data = generate_data(model, days=30)

print(f"[3] 数据统计:")
print(f"  总记录数: {len(data)}")
print(f"  时间范围: {data['Timestamp'].min()} 到 {data['Timestamp'].max()}")

print("\n  基本统计:")
for col in data.columns[1:]:
    print(f"  {col:25s} Mean:{data[col].mean():8.2f} Std:{data[col].std():6.2f} Min:{data[col].min():8.2f} Max:{data[col].max():8.2f}")

print("\n[4] 生成可视化...")
fig = plot_comparison(data, "磨煤机数据生成 - 优化版 (30天)")

out_dir = "MultiAgentTS/visualizations"
os.makedirs(out_dir, exist_ok=True)

fig.savefig(os.path.join(out_dir, "optimized_generation.png"), dpi=150, bbox_inches='tight')
print(f"  已保存: {os.path.join(out_dir, 'optimized_generation.png')}")

data.to_csv(os.path.join(out_dir, "optimized_data_sample.csv"), index=False)
print(f"  已保存: {os.path.join(out_dir, 'optimized_data_sample.csv')}")

print("\n[完成] 优化数据生成成功！")
print("="*70)

plt.show()

