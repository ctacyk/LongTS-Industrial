"""
TimeBlender-based Coal Mill Synthesis — 基于真实数据分析的仿真脚本

真实数据关键特征（从 chunk_1.png 观察）：
1. 电流 40-65A 连续波动，频繁短暂跌落（停机/减载）
2. 绕组温度 35-55°C 缓慢波动，2-3天长周期
3. 轴承温度有明显日夜温差
4. 入口温度大部分260-300°C，停机时骤降到0
5. 阀门开度80-100%，停机时跌到0
6. 所有通道高度自相关（物理热惯性）
7. 运行/停机交替出现

核心建模思路：
- 用 TimeBlender 的 LambdaEvent + memory 实现热惯性递推
- 用负荷曲线（含随机停机事件）驱动所有通道
- 通道间通过 sub_events 依赖图自然耦合
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_blender.core import Generator, LambdaEvent, ConstantEvent


# ============================================================
# 物理模型：热惯性递推
# ============================================================
def thermal_inertia(t, i, memory, sub_events):
    """一阶低通滤波器模拟热惯性: dT/dt = alpha*(T_target - T) + noise"""
    target = sub_events['target'].execute(t)
    alpha = sub_events['alpha'].execute(t)
    noise_std = sub_events['noise'].execute(t)

    if 'prev' not in memory:
        memory['prev'] = target

    prev = memory['prev']
    noise = np.random.normal(0, noise_std)
    new_val = prev + alpha * (target - prev) + noise
    memory['prev'] = new_val
    return new_val


# ============================================================
# 负荷模型：含随机停机事件
# ============================================================
def load_model(t, i, memory, sub_events):
    """
    模拟真实磨煤机的负荷曲线：
    - 正常运行时负荷在0.6-0.9之间缓慢波动
    - 随机出现停机事件（负荷骤降到0）
    - 停机持续几小时到半天
    - 启动时负荷逐渐爬升
    """
    if 'state' not in memory:
        memory['state'] = 'running'  # running / shutdown / starting
        memory['load'] = 0.7
        memory['shutdown_timer'] = 0
        memory['startup_timer'] = 0
        memory['next_shutdown'] = np.random.randint(2000, 8000)  # 随机停机间隔
        memory['step'] = 0

    memory['step'] += 1
    step = memory['step']

    if memory['state'] == 'running':
        # 缓慢波动
        drift = np.random.normal(0, 0.003)
        memory['load'] = np.clip(memory['load'] + drift, 0.5, 0.95)

        # 检查是否该停机了
        if step > memory['next_shutdown']:
            memory['state'] = 'shutdown'
            memory['shutdown_timer'] = np.random.randint(500, 3000)  # 停机持续步数
            memory['next_shutdown'] = step + memory['shutdown_timer'] + np.random.randint(2000, 8000)

    elif memory['state'] == 'shutdown':
        # 快速降载
        memory['load'] = max(0.0, memory['load'] - 0.05)
        memory['shutdown_timer'] -= 1
        if memory['shutdown_timer'] <= 0:
            memory['state'] = 'starting'
            memory['startup_timer'] = np.random.randint(100, 400)

    elif memory['state'] == 'starting':
        # 缓慢升载
        memory['load'] = min(0.7 + np.random.normal(0, 0.02), memory['load'] + 0.005)
        memory['startup_timer'] -= 1
        if memory['startup_timer'] <= 0:
            memory['state'] = 'running'

    return memory['load']


# ============================================================
# 各通道的目标值模型
# ============================================================
def current_target(t, i, memory, sub_events):
    """电机电流目标值 = f(负荷)，含测量噪声"""
    load = sub_events['load'].execute(t)
    base = 10 + 55 * load  # 0负荷≈10A(停机), 满负荷≈65A
    noise = np.random.normal(0, 1.2)  # 电流测量噪声较大
    return base + noise


def winding_target(t, i, memory, sub_events):
    """绕组温度目标值 = f(电流²)，I²R损耗模型"""
    curr = sub_events['current'].execute(t)
    ambient = 20 + 3 * np.sin(2 * np.pi * i / 2880)  # 日夜环境温差
    heat_rise = (curr ** 2) * 0.012  # I²R系数
    return ambient + heat_rise


def bearing_ds_target(t, i, memory, sub_events):
    """驱动侧轴承温度目标值 = f(电流, 环境)"""
    curr = sub_events['current'].execute(t)
    ambient = 18 + 4 * np.sin(2 * np.pi * i / 2880)
    # 轴承温度 = 环境 + 负荷相关热 + 摩擦基础热
    return ambient + curr * 0.25 + 5.0


def bearing_nds_target(t, i, memory, sub_events):
    """非驱动侧轴承温度 = 驱动侧略低"""
    curr = sub_events['current'].execute(t)
    ambient = 16 + 4 * np.sin(2 * np.pi * i / 2880)
    return ambient + curr * 0.15


def inlet_temp_target(t, i, memory, sub_events):
    """入口温度目标值：运行时260-300°C，停机时降到环境温度"""
    load = sub_events['load'].execute(t)
    if load > 0.1:
        return 250 + 50 * load + np.random.normal(0, 2)
    else:
        return 20 + np.random.normal(0, 1)  # 停机


def valve_target(t, i, memory, sub_events):
    """阀门开度：运行时80-100%，停机时0"""
    load = sub_events['load'].execute(t)
    if load > 0.1:
        return 75 + 25 * load + np.random.normal(0, 1.5)
    else:
        return np.random.normal(0, 0.5)  # 停机时关闭


# ============================================================
# 构建完整模型
# ============================================================
def build_coal_mill_model():
    """构建磨煤机多通道仿真模型"""

    # 负荷曲线（驱动所有通道的核心信号）
    load = LambdaEvent(load_model, sub_events={})

    # 电机电流（直接跟随负荷，噪声大）
    current_tgt = LambdaEvent(current_target, sub_events={'load': load})
    motor_current = LambdaEvent(
        thermal_inertia,
        sub_events={
            'target': current_tgt,
            'alpha': ConstantEvent(0.3),     # 电流响应快
            'noise': ConstantEvent(0.8),
        }
    )

    # 绕组温度1（依赖电流，热惯性大）
    winding_tgt = LambdaEvent(winding_target, sub_events={'current': motor_current})
    winding_temp_1 = LambdaEvent(
        thermal_inertia,
        sub_events={
            'target': winding_tgt,
            'alpha': ConstantEvent(0.008),   # 热时间常数 ~125步 ≈ 1小时
            'noise': ConstantEvent(0.08),
        }
    )

    # 绕组温度2（跟随温度1，有小偏差）
    def winding2_func(t, i, mem, sub):
        w1 = sub['w1'].execute(t)
        if 'prev' not in mem:
            mem['prev'] = w1 - 4.0
        target = w1 - 4.0 + np.random.normal(0, 0.15)  # 固定偏差-4°C
        mem['prev'] = mem['prev'] + 0.01 * (target - mem['prev'])
        return mem['prev']

    winding_temp_2 = LambdaEvent(winding2_func, sub_events={'w1': winding_temp_1})

    # 非驱动侧轴承温度
    bearing_nds_tgt = LambdaEvent(bearing_nds_target, sub_events={'current': motor_current})
    bearing_temp_nds = LambdaEvent(
        thermal_inertia,
        sub_events={
            'target': bearing_nds_tgt,
            'alpha': ConstantEvent(0.005),   # 轴承热惯性更大
            'noise': ConstantEvent(0.06),
        }
    )

    # 驱动侧轴承温度
    bearing_ds_tgt = LambdaEvent(bearing_ds_target, sub_events={'current': motor_current})
    bearing_temp_ds = LambdaEvent(
        thermal_inertia,
        sub_events={
            'target': bearing_ds_tgt,
            'alpha': ConstantEvent(0.005),
            'noise': ConstantEvent(0.08),
        }
    )

    # 入口温度（响应较快，但有一定惯性）
    inlet_tgt = LambdaEvent(inlet_temp_target, sub_events={'load': load})
    inlet_temp = LambdaEvent(
        thermal_inertia,
        sub_events={
            'target': inlet_tgt,
            'alpha': ConstantEvent(0.05),    # 热风响应中等
            'noise': ConstantEvent(1.5),
        }
    )

    # 阀门开度（响应快）
    valve_tgt = LambdaEvent(valve_target, sub_events={'load': load})
    valve_opening = LambdaEvent(
        thermal_inertia,
        sub_events={
            'target': valve_tgt,
            'alpha': ConstantEvent(0.15),    # 阀门执行器响应快
            'noise': ConstantEvent(0.3),
        }
    )

    return {
        'motor_current': motor_current,
        'winding_temp_1': winding_temp_1,
        'winding_temp_2': winding_temp_2,
        'bearing_temp_nondrive': bearing_temp_nds,
        'bearing_temp_drive': bearing_temp_ds,
        'inlet_temp': inlet_temp,
        'inlet_valve_opening': valve_opening,
    }


# ============================================================
# 生成 + 可视化对比
# ============================================================
def generate_and_compare():
    np.random.seed(42)

    # 生成 15 天数据（30s 采样 = 43200 步）
    total_steps = 43200
    start_date = pd.Timestamp('2024-02-19')
    end_date = start_date + pd.Timedelta(seconds=30 * total_steps)

    print("Building TimeBlender model...")
    events = build_coal_mill_model()

    print(f"Generating {total_steps} steps (~15 days at 30s)...")
    gen = Generator(start_date=start_date, end_date=end_date, freq='30S')
    df_synth = gen.generate(events, n=1)

    # 保存合成数据
    os.makedirs('Data/CoalMill_TB', exist_ok=True)
    df_synth.to_csv('Data/CoalMill_TB/synth_timeblender.csv')
    print(f"Saved: Data/CoalMill_TB/synth_timeblender.csv ({df_synth.shape})")

    # 加载真实数据
    df_real = pd.read_csv('Dataset/CoalMill/merged.csv', index_col=0, parse_dates=[0])
    # 截取前15天对齐
    df_real = df_real.iloc[:total_steps]

    # 列名映射
    real_cols = list(df_real.columns)
    synth_cols = list(df_synth.columns)
    col_pairs = list(zip(real_cols, synth_cols))

    # 可视化对比
    fig, axes = plt.subplots(len(col_pairs), 1, figsize=(18, 3 * len(col_pairs)), sharex=False)

    for idx, (real_col, synth_col) in enumerate(col_pairs):
        ax = axes[idx]
        # 真实数据（蓝色）
        ax.plot(df_real[real_col].values, color='#1f77b4', linewidth=0.4, alpha=0.8, label='Real')
        # 合成数据（红色）
        ax.plot(df_synth[synth_col].values, color='#d62728', linewidth=0.4, alpha=0.8, label='TimeBlender Synth')

        ax.set_ylabel(synth_col, fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.set_title('Real (Blue) vs TimeBlender Synthetic (Red) — Coal Mill 15 Days',
                        fontsize=11, fontweight='bold')

    axes[-1].set_xlabel('Time Step (30s intervals)')
    plt.tight_layout()
    plt.savefig('Data/CoalMill_TB/comparison_real_vs_tb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: Data/CoalMill_TB/comparison_real_vs_tb.png")

    # 统计对比
    print("\n=== Statistical Comparison ===")
    print(f"{'Channel':<25} {'Real Mean':>10} {'Synth Mean':>10} {'Real Std':>10} {'Synth Std':>10}")
    for real_col, synth_col in col_pairs:
        rm = df_real[real_col].mean()
        sm = df_synth[synth_col].mean()
        rs = df_real[real_col].std()
        ss = df_synth[synth_col].std()
        print(f"{synth_col:<25} {rm:>10.2f} {sm:>10.2f} {rs:>10.2f} {ss:>10.2f}")


if __name__ == '__main__':
    generate_and_compare()
