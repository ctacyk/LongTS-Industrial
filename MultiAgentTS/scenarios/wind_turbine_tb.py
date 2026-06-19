"""
风力发电机组 TimeBlender 场景库 — 10种故障 + 3种正常工况

遵循 data-synthesis-loop rule：
- 所有温度/慢响应通道使用 LambdaEvent + memory 热惯性递推
- 通道间通过 sub_events 依赖图自然耦合
- 参数基于 Kelmarsh SCADA 真实数据统计校准

真实数据校准目标（Kelmarsh Turbine 1, 2019, Senvion MM92 2.05MW）：
  wind_speed:       mean=6.07, std=2.71, AC1=0.966, range=[0.1, 22.3]
  rotor_speed:      mean=10.45, std=3.99, AC1=0.953, max=15.3
  generator_speed:  mean=1240, std=472, AC1=0.952 (gear_ratio≈118.7)
  power_output:     mean=604, std=603, AC1=0.968, rated=2050kW
  pitch_angle:      mean=4.95, std=14.1, AC1=0.896, mostly 0° (q75=0.91)
  nacelle_temp:     mean=21.8, std=5.3, AC1=0.956
  gearbox_oil_temp: mean=53.4, std=5.4, AC1=0.980
  gen_bearing_temp: mean=40.1, std=5.6, AC1=0.972
  停机占9.6%, 额定以下88.1%, 额定以上2.3%

故障谱（10种）：
  机械类: 齿轮箱轴承磨损, 发电机轴承磨损
  变桨类: 变桨卡死
  热力类: 发电机过热, 齿轮箱冷却故障
  气动类: 偏航误差, 叶片结冰, 功率曲线偏差
  传感器: 风速仪漂移
  电气类: 电网扰动
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from time_blender.core import Generator, LambdaEvent, ConstantEvent
from MultiAgentTS.scenarios.base import (
    GenerationMetadata, EventMetadata, ChannelInfo,
    compute_difficulty, save_scenario_output, plot_eval_chart,
    register_scenario, get_all_scenarios, compute_fault_delta, compute_power_envelope,
)

# ============================================================
# 通道定义（基于 Kelmarsh 校准）
# ============================================================
WIND_TURBINE_CHANNELS = [
    ChannelInfo("wind_speed", "风速", "m/s",
                normal_range=(3.0, 15.0), alarm_threshold=25.0,
                physical_meaning="机舱风速仪测量的10分钟平均风速，25m/s为切出风速"),
    ChannelInfo("rotor_speed", "转子转速", "RPM",
                normal_range=(6.0, 15.3), alarm_threshold=16.5, alarm_low=5.0,
                physical_meaning="风轮旋转速度，额定约14.5RPM，低于5RPM疑似卡涩"),
    ChannelInfo("generator_speed", "发电机转速", "RPM",
                normal_range=(700.0, 1800.0), alarm_threshold=1900.0, alarm_low=500.0,
                physical_meaning="发电机转速，齿轮比约118.7"),
    ChannelInfo("power_output", "有功功率", "kW",
                normal_range=(0.0, 2050.0), alarm_threshold=2100.0,
                physical_meaning="并网有功功率输出，额定2050kW"),
    ChannelInfo("pitch_angle", "桨距角", "°",
                normal_range=(0.0, 25.0), alarm_threshold=None,
                physical_meaning="叶片变桨角度，额定以下约0°，额定以上逐渐增大"),
    ChannelInfo("nacelle_temp", "机舱温度", "°C",
                normal_range=(10.0, 35.0), alarm_threshold=50.0,
                physical_meaning="机舱内部环境温度，受发电机散热影响"),
    ChannelInfo("gearbox_oil_temp", "齿轮箱油温", "°C",
                normal_range=(45.0, 60.0), alarm_threshold=70.0, alarm_low=35.0,
                physical_meaning="齿轮箱润滑油温度，低于35°C润滑不良需预热"),
    ChannelInfo("gen_bearing_temp", "发电机轴承温度", "°C",
                normal_range=(30.0, 55.0), alarm_threshold=80.0,
                physical_meaning="发电机前端轴承温度"),
]

DOMAIN = "wind_turbine"
DOMAIN_DISPLAY = "风力发电机组"
SAMPLING_RATE = "10min"
SAMPLING_SECONDS = 600

_CH = {ch.name: ch for ch in WIND_TURBINE_CHANNELS}

GEAR_RATIO = 118.7
RATED_POWER = 2050.0
CUT_IN_WIND = 3.0
RATED_WIND = 12.5
CUT_OUT_WIND = 25.0
RATED_ROTOR_RPM = 14.5
AMBIENT_TEMP_BASE = 20.0


# ============================================================
# TimeBlender 基础构建块
# ============================================================
def thermal_inertia(t, i, memory, sub_events):
    """一阶低通递推（温度/慢响应通道通用）"""
    target = sub_events['target'].execute(t)
    alpha = sub_events['alpha'].execute(t)
    noise_std = sub_events['noise'].execute(t)
    if 'prev' not in memory:
        memory['prev'] = target
    prev = memory['prev']
    memory['prev'] = prev + alpha * (target - prev) + np.random.normal(0, noise_std)
    return memory['prev']


# ============================================================
# 风速模型（驱动整个系统的非平稳随机过程）
# ============================================================
def _wind_steady(t, i, memory, sub_events):
    """稳定风况：均值~6m/s, AC1≈0.96（均值回复 + 白噪声）
    使用 Ornstein-Uhlenbeck 过程匹配真实 SCADA 统计"""
    if 'ws' not in memory:
        memory['ws'] = 6.0
    alpha_w = 0.035  # → AC1 ≈ 1 - alpha ≈ 0.965
    mu = 6.0
    memory['ws'] += alpha_w * (mu - memory['ws']) + np.random.normal(0, 0.70)
    memory['ws'] = np.clip(memory['ws'], 0.3, 22.0)
    return memory['ws']


def _wind_varying(t, i, memory, sub_events):
    """变化风况：含日变化调制 + 慢趋势 + 阵风"""
    if 'ws' not in memory:
        memory['ws'] = 6.0
        memory['step'] = 0
    memory['step'] += 1
    s = memory['step']
    mu = 6.0 + 1.5 * np.sin(2 * np.pi * s / 144) + 1.0 * np.sin(2 * np.pi * s / 4320)
    alpha_w = 0.04
    gust = 0.0
    if np.random.random() < 0.005:
        gust = np.random.uniform(2.0, 6.0) * np.random.choice([-1, 1])
    memory['ws'] += alpha_w * (mu - memory['ws']) + np.random.normal(0, 0.55) + gust * 0.1
    memory['ws'] = np.clip(memory['ws'], 0.3, 23.0)
    return memory['ws']


def _wind_turbulent(t, i, memory, sub_events):
    """强湍流：大幅波动 + 频繁低风速"""
    if 'ws' not in memory:
        memory['ws'] = 5.5
    alpha_w = 0.03
    mu = 5.5
    memory['ws'] += alpha_w * (mu - memory['ws']) + np.random.normal(0, 0.75)
    memory['ws'] = np.clip(memory['ws'], 0.3, 23.0)
    return memory['ws']


# ============================================================
# 通道目标值函数
# ============================================================
def _rotor_speed_tgt(t, i, m, s):
    """风速 → 转子转速（空气动力学映射 + 额定限制）"""
    ws = s['wind'].execute(t)
    if ws < CUT_IN_WIND:
        return 0.0
    if ws >= CUT_OUT_WIND:
        return 0.0
    rpm = min(RATED_ROTOR_RPM, ws * 1.5)
    return rpm + np.random.normal(0, 0.15)


def _gen_speed_tgt(t, i, m, s):
    """转子转速 × 齿轮比 → 发电机转速"""
    rotor = s['rotor'].execute(t)
    return rotor * GEAR_RATIO + np.random.normal(0, 5.0)


def _power_tgt(t, i, m, s):
    """风速 → 功率（立方曲线 + 额定限制 + 变桨修正）"""
    ws = s['wind'].execute(t)
    pitch = s['pitch'].execute(t)
    if ws < CUT_IN_WIND or ws >= CUT_OUT_WIND:
        return 0.0
    Cp = 0.45 * max(0.0, 1.0 - pitch / 90.0)
    rho = 1.225
    A = np.pi * (46.0 ** 2)  # rotor radius 46m
    P = 0.5 * rho * A * Cp * (ws ** 3) / 1000.0
    P = min(P, RATED_POWER)
    return P + np.random.normal(0, max(10.0, P * 0.02))


def _pitch_tgt(t, i, m, s):
    """风速 → 桨距角（控制逻辑）
    Kelmarsh 实测: mean=4.95, q75=0.91 → 大部分时间≈0°
    仅停机(feathered)和额定以上才>0°"""
    ws = s['wind'].execute(t)
    if ws < CUT_IN_WIND - 0.5:
        return np.random.uniform(75.0, 90.0)  # feathered (standstill)
    if ws >= CUT_OUT_WIND:
        return 90.0
    if ws < RATED_WIND:
        return max(0.0, np.random.normal(0.2, 0.3))
    excess = ws - RATED_WIND
    return min(25.0, excess * 2.0 + np.random.normal(0, 0.5))


def _nacelle_temp_tgt(t, i, m, s):
    """机舱温度目标：环境(可变) + 设备散热
    Kelmarsh实测: mean=21.8, std=5.3, range=[5.2, 39.7]"""
    power = s['power'].execute(t)
    heat_from_equip = max(0, power) / RATED_POWER * 10.0
    if 'amb' not in m:
        m['amb'] = AMBIENT_TEMP_BASE
    m['amb'] += np.random.normal(0, 0.3)
    m['amb'] = np.clip(m['amb'], 5.0, 32.0)
    diurnal = 3.5 * np.sin(2 * np.pi * i / 144)
    return m['amb'] + diurnal + heat_from_equip + np.random.normal(0, 0.8)


def _gearbox_oil_tgt(t, i, m, s):
    """齿轮箱油温目标：受转速（摩擦热）+ 环境温度影响
    Kelmarsh实测: mean=53.4, range=[14, 60]"""
    rotor = s['rotor'].execute(t)
    operating = 1.0 if rotor > 1.0 else 0.0
    friction_heat = (rotor / RATED_ROTOR_RPM) ** 2 * 18.0
    return 38.0 + friction_heat * operating + AMBIENT_TEMP_BASE * 0.35 + np.random.normal(0, 0.4)


def _gen_bearing_tgt(t, i, m, s):
    """发电机轴承温度目标：受功率（电流热）+ 环境温度影响
    Kelmarsh实测: mean=40.1, range=[11, 74]"""
    power = s['power'].execute(t)
    current_heat = (max(0, power) / RATED_POWER) ** 2 * 25.0
    operating = 1.0 if power > 10.0 else 0.0
    return 28.0 + current_heat * operating + AMBIENT_TEMP_BASE * 0.35 + np.random.normal(0, 0.4)


# ============================================================
# 故障目标值修改器
# ============================================================
def _fault_ramp(t, i, m, s):
    """渐变故障：故障区间内 target 线性→指数增加 delta"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    delta = s['delta'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        return base + delta * (np.exp(2 * p) - 1) / (np.exp(2) - 1)
    elif i >= fe:
        return base + delta
    return base


def _fault_step(t, i, m, s):
    """阶跃故障：故障开始后 target 直接偏移"""
    base = s['base'].execute(t)
    fs = int(s['fs'].execute(t))
    delta = s['delta'].execute(t)
    return base + delta if i >= fs else base


def _fault_oscillation(t, i, m, s):
    """振荡故障：叠加周期性波动"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    amp = s['amp'].execute(t)
    period = s['period'].execute(t)
    if fs <= i < fe:
        return base + amp * np.sin(2 * np.pi * (i - fs) / period) * (1 + np.random.normal(0, 0.2))
    return base


def _fault_alpha_reduce(t, i, m, s):
    """降低散热系数（冷却故障用）"""
    normal_alpha = s['normal_alpha'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    reduction = s['reduction'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        return normal_alpha * (1 - reduction * p)
    elif i >= fe:
        return normal_alpha * (1 - reduction)
    return normal_alpha


def _fault_stuck(t, i, m, s):
    """卡死故障：故障区间内输出固定值"""
    base = s['base'].execute(t)
    fs = int(s['fs'].execute(t))
    stuck_val = s['stuck_val'].execute(t)
    return stuck_val if i >= fs else base


def _fault_scale(t, i, m, s):
    """缩放故障：故障区间内 output 乘以系数（全窗口线性渐进）"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    factor = s['factor'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        return base * (1.0 - p * (1.0 - factor))
    elif i >= fe:
        return base * factor
    return base


def _fault_scale_fast(t, i, m, s):
    """快速缩放：前 10% 窗口线性过渡，之后保持 factor"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    factor = s['factor'].execute(t)
    if fs <= i < fe:
        ramp_end = fs + max(int((fe - fs) * 0.10), 1)
        if i < ramp_end:
            p = (i - fs) / (ramp_end - fs)
        else:
            p = 1.0
        return base * (1.0 - p * (1.0 - factor))
    elif i >= fe:
        return base * factor
    return base


def _fault_power_envelope(t, i, m, s):
    """功率包络限制：降低最大可达功率而非按比例缩放
    物理依据：叶片结冰/偏航误差/功率曲线偏差等故障降低气动效率Cp，
    等效于降低功率曲线的额定功率上限。
    低风速时功率本就低于上限，影响不明显；
    高风速时功率被限制在新上限，产生明显的功率损失。
    比例缩放的问题：当 power=100kW 时缩放 12% 仅损失 12kW，对比 std=603 完全不可见。
    """
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    max_power = s['max_power'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        current_cap = RATED_POWER - p * (RATED_POWER - max_power)
        return min(base, current_cap)
    elif i >= fe:
        return min(base, max_power)
    return base


def _fault_drift(t, i, m, s):
    """漂移故障：传感器读数渐进偏移"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    drift = s['drift'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        return base + drift * p
    elif i >= fe:
        return base + drift
    return base


# ============================================================
# 模型构建工厂
# ============================================================
def _build_base_model(wind_fn):
    """构建基础模型（正常运行），返回事件字典 + 关键中间引用"""
    wind = LambdaEvent(wind_fn, sub_events={})

    pitch_tgt = LambdaEvent(_pitch_tgt, sub_events={'wind': wind})
    pitch = LambdaEvent(thermal_inertia, sub_events={
        'target': pitch_tgt, 'alpha': ConstantEvent(0.6), 'noise': ConstantEvent(0.3)})

    rotor_tgt = LambdaEvent(_rotor_speed_tgt, sub_events={'wind': wind})
    rotor = LambdaEvent(thermal_inertia, sub_events={
        'target': rotor_tgt, 'alpha': ConstantEvent(0.5), 'noise': ConstantEvent(0.20)})

    gen_spd_tgt = LambdaEvent(_gen_speed_tgt, sub_events={'rotor': rotor})
    gen_spd = LambdaEvent(thermal_inertia, sub_events={
        'target': gen_spd_tgt, 'alpha': ConstantEvent(0.5), 'noise': ConstantEvent(12.0)})

    power_tgt = LambdaEvent(_power_tgt, sub_events={'wind': wind, 'pitch': pitch})
    power = LambdaEvent(thermal_inertia, sub_events={
        'target': power_tgt, 'alpha': ConstantEvent(0.5), 'noise': ConstantEvent(18.0)})

    nacelle_tgt = LambdaEvent(_nacelle_temp_tgt, sub_events={'power': power})
    nacelle = LambdaEvent(thermal_inertia, sub_events={
        'target': nacelle_tgt, 'alpha': ConstantEvent(0.06), 'noise': ConstantEvent(0.35)})

    gearbox_tgt = LambdaEvent(_gearbox_oil_tgt, sub_events={'rotor': rotor})
    gearbox = LambdaEvent(thermal_inertia, sub_events={
        'target': gearbox_tgt, 'alpha': ConstantEvent(0.025), 'noise': ConstantEvent(0.25)})

    gen_bear_tgt = LambdaEvent(_gen_bearing_tgt, sub_events={'power': power})
    gen_bear = LambdaEvent(thermal_inertia, sub_events={
        'target': gen_bear_tgt, 'alpha': ConstantEvent(0.04), 'noise': ConstantEvent(0.30)})

    events = {
        'wind_speed': wind, 'rotor_speed': rotor, 'generator_speed': gen_spd,
        'power_output': power, 'pitch_angle': pitch,
        'nacelle_temp': nacelle, 'gearbox_oil_temp': gearbox,
        'gen_bearing_temp': gen_bear,
    }
    refs = {
        'wind': wind, 'rotor': rotor, 'gen_spd': gen_spd,
        'power': power, 'pitch': pitch, 'pitch_tgt': pitch_tgt,
        'rotor_tgt': rotor_tgt, 'power_tgt': power_tgt,
        'nacelle_tgt': nacelle_tgt, 'gearbox_tgt': gearbox_tgt,
        'gen_bear_tgt': gen_bear_tgt,
    }
    return events, refs


def _inject_fault(wind_fn, fault_type, fs, fe, severity, params):
    """构建带故障注入的模型"""
    sev_map = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}
    sev_factor = sev_map.get(severity, 0.6)

    wind = LambdaEvent(wind_fn, sub_events={})
    fsc, fec = ConstantEvent(float(fs)), ConstantEvent(float(fe))

    # defaults
    pitch_tgt_ev = LambdaEvent(_pitch_tgt, sub_events={'wind': wind})
    rotor_tgt_ev = LambdaEvent(_rotor_speed_tgt, sub_events={'wind': wind})
    power_tgt_ev = None  # rebuilt after pitch
    nacelle_tgt_ev = None  # rebuilt after power
    gearbox_tgt_ev = None  # rebuilt after rotor
    gen_bear_tgt_ev = None  # rebuilt after power

    gearbox_alpha_ev = ConstantEvent(0.025)

    wind_final = wind

    if fault_type == "gearbox_bearing_wear":
        # 齿轮箱轴承磨损 → 油温渐升 + 传动效率↓ → 功率↓
        pass  # handled via gearbox_tgt modification below

    elif fault_type == "pitch_stuck":
        # 变桨卡死 → 桨距角固定在故障时刻值
        stuck_val = sev_factor * 3.0  # stuck at a small angle
        pitch_tgt_ev = LambdaEvent(_fault_stuck, sub_events={
            'base': LambdaEvent(_pitch_tgt, sub_events={'wind': wind}),
            'fs': fsc, 'stuck_val': ConstantEvent(stuck_val)})

    elif fault_type == "generator_overheat":
        # 发电机过热 → 轴承温度↑ + 降额
        pass  # handled via gen_bear_tgt modification below

    elif fault_type == "yaw_misalignment":
        # 偏航误差 → 等效风速降低 → 功率↓
        pass  # handled via power scaling

    elif fault_type == "blade_icing":
        # 叶片结冰 → Cp降低 → 转速和功率同时↓
        pass  # handled via power and rotor scaling

    elif fault_type == "gearbox_cooling_fault":
        # 齿轮箱冷却故障 → alpha降低 → 油温持续升高
        reduction = sev_factor * 0.85
        gearbox_alpha_ev = LambdaEvent(_fault_alpha_reduce, sub_events={
            'normal_alpha': ConstantEvent(0.025), 'fs': fsc, 'fe': fec,
            'reduction': ConstantEvent(reduction)})

    elif fault_type == "power_curve_deviation":
        # 功率曲线偏差 → 同风速下功率偏低
        pass  # handled via power scaling

    elif fault_type == "anemometer_drift":
        # 风速仪漂移 → 风速读数渐变偏移（但实际物理不变）
        d_ws = compute_fault_delta(_CH['wind_speed'], severity, "increase") * 0.5
        wind_final = LambdaEvent(_fault_drift, sub_events={
            'base': wind, 'fs': fsc, 'fe': fec,
            'drift': ConstantEvent(d_ws)})

    elif fault_type == "gen_bearing_wear":
        # 发电机轴承磨损 → 轴承温度渐升
        pass  # handled via gen_bear_tgt modification below

    elif fault_type == "grid_disturbance":
        # 电网扰动 → 功率振荡
        pass  # handled via power oscillation

    # Build pitch
    pitch = LambdaEvent(thermal_inertia, sub_events={
        'target': pitch_tgt_ev, 'alpha': ConstantEvent(0.6), 'noise': ConstantEvent(0.3)})

    # Build rotor (some faults scale rotor)
    if fault_type == "blade_icing":
        factor = 1.0 - sev_factor * 0.55  # 结冰导致气动升力大幅下降
        rotor_tgt_ev = LambdaEvent(_fault_scale, sub_events={
            'base': LambdaEvent(_rotor_speed_tgt, sub_events={'wind': wind}),
            'fs': fsc, 'fe': fec, 'factor': ConstantEvent(factor)})
    elif fault_type == "pitch_stuck":
        # 桨距角卡死 → 无法变桨限速 → 高风速时转子过速
        factor = 1.0 + sev_factor * 0.25
        rotor_tgt_ev = LambdaEvent(_fault_scale, sub_events={
            'base': LambdaEvent(_rotor_speed_tgt, sub_events={'wind': wind}),
            'fs': fsc, 'fe': fec, 'factor': ConstantEvent(factor)})

    rotor = LambdaEvent(thermal_inertia, sub_events={
        'target': rotor_tgt_ev, 'alpha': ConstantEvent(0.5), 'noise': ConstantEvent(0.20)})

    gen_spd_base_tgt = LambdaEvent(_gen_speed_tgt, sub_events={'rotor': rotor})
    if fault_type == "grid_disturbance":
        # A-class fix: 电网扰动 → 发电机转速轻微波动
        d_gs = compute_fault_delta(_CH['generator_speed'], severity, "increase") * 0.3
        gen_spd_tgt_ev = LambdaEvent(_fault_oscillation, sub_events={
            'base': gen_spd_base_tgt, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_gs), 'period': ConstantEvent(15.0)})
    else:
        gen_spd_tgt_ev = gen_spd_base_tgt
    gen_spd = LambdaEvent(thermal_inertia, sub_events={
        'target': gen_spd_tgt_ev, 'alpha': ConstantEvent(0.5), 'noise': ConstantEvent(12.0)})

    # Build power (multiple faults affect power)
    power_base_tgt = LambdaEvent(_power_tgt, sub_events={'wind': wind, 'pitch': pitch})

    # 功率故障策略：使用乘法衰减（_fault_scale），物理含义为效率Cp/η降低
    # power_output CV≈1.0（std≈mean），mean-based SNR 结构性偏低，
    # 但乘法衰减在整个信号上压缩振幅包络，视觉上可辨识
    if fault_type == "yaw_misalignment":
        # 偏航误差：P ∝ cos³(θ)，快速过渡（偏航通常是突发机械故障）
        theta = sev_factor * 55.0  # mild=16.5°, moderate=33°, severe=55°
        factor = np.cos(np.radians(theta)) ** 3
        power_tgt_ev = LambdaEvent(_fault_scale_fast, sub_events={
            'base': power_base_tgt, 'fs': fsc, 'fe': fec,
            'factor': ConstantEvent(factor)})
    elif fault_type == "blade_icing":
        # Cp 大幅下降（冰层逐渐累积，使用普通 scale 渐进但加大幅度）
        factor = 1.0 - sev_factor * 0.65  # mild=0.805, moderate=0.61, severe=0.35
        power_tgt_ev = LambdaEvent(_fault_scale, sub_events={
            'base': power_base_tgt, 'fs': fsc, 'fe': fec,
            'factor': ConstantEvent(factor)})
    elif fault_type == "power_curve_deviation":
        # 叶片侵蚀 → Cp 渐进下降（快速过渡：已运行一段时间后突然劣化到临界点）
        factor = 1.0 - sev_factor * 0.50  # mild=0.85, moderate=0.70, severe=0.50
        power_tgt_ev = LambdaEvent(_fault_scale_fast, sub_events={
            'base': power_base_tgt, 'fs': fsc, 'fe': fec,
            'factor': ConstantEvent(factor)})
    elif fault_type == "gearbox_bearing_wear":
        # 传动效率降低
        factor = 1.0 - sev_factor * 0.30  # mild=0.91, moderate=0.82, severe=0.70
        power_tgt_ev = LambdaEvent(_fault_scale, sub_events={
            'base': power_base_tgt, 'fs': fsc, 'fe': fec,
            'factor': ConstantEvent(factor)})
    elif fault_type == "generator_overheat":
        # 降额保护限制最大输出
        factor = 1.0 - sev_factor * 0.35
        power_tgt_ev = LambdaEvent(_fault_scale, sub_events={
            'base': power_base_tgt, 'fs': fsc, 'fe': fec,
            'factor': ConstantEvent(factor)})
    elif fault_type == "grid_disturbance":
        # 电网扰动：功率振荡 + 均值偏移
        d_pw = compute_fault_delta(_CH['power_output'], severity, "increase") * 0.5
        power_tgt_ev = LambdaEvent(_fault_oscillation, sub_events={
            'base': power_base_tgt, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_pw), 'period': ConstantEvent(15.0)})
    else:
        power_tgt_ev = power_base_tgt

    power = LambdaEvent(thermal_inertia, sub_events={
        'target': power_tgt_ev, 'alpha': ConstantEvent(0.5), 'noise': ConstantEvent(18.0)})

    # Build gearbox oil temp (delta calibrated via compute_fault_delta)
    gearbox_base_tgt = LambdaEvent(_gearbox_oil_tgt, sub_events={'rotor': rotor})
    if fault_type == "gearbox_bearing_wear":
        d_gb = compute_fault_delta(_CH['gearbox_oil_temp'], severity, "increase")
        gearbox_tgt_ev = LambdaEvent(_fault_ramp, sub_events={
            'base': gearbox_base_tgt, 'fs': fsc, 'fe': fec,
            'delta': ConstantEvent(d_gb)})
    elif fault_type == "gearbox_cooling_fault":
        # alpha 降低 + 直接温升 delta 双管齐下确保可见
        d_gb = compute_fault_delta(_CH['gearbox_oil_temp'], severity, "increase") * 0.7
        gearbox_tgt_ev = LambdaEvent(_fault_ramp, sub_events={
            'base': gearbox_base_tgt, 'fs': fsc, 'fe': fec,
            'delta': ConstantEvent(d_gb)})
    else:
        gearbox_tgt_ev = gearbox_base_tgt

    gearbox = LambdaEvent(thermal_inertia, sub_events={
        'target': gearbox_tgt_ev, 'alpha': gearbox_alpha_ev, 'noise': ConstantEvent(0.25)})

    # Build gen bearing temp (delta calibrated via compute_fault_delta)
    gen_bear_base_tgt = LambdaEvent(_gen_bearing_tgt, sub_events={'power': power})
    if fault_type == "generator_overheat":
        d_gb_t = compute_fault_delta(_CH['gen_bearing_temp'], severity, "increase")
        gen_bear_tgt_ev = LambdaEvent(_fault_ramp, sub_events={
            'base': gen_bear_base_tgt, 'fs': fsc, 'fe': fec,
            'delta': ConstantEvent(d_gb_t)})
    elif fault_type == "gen_bearing_wear":
        d_gb_t = compute_fault_delta(_CH['gen_bearing_temp'], severity, "increase")
        gen_bear_tgt_ev = LambdaEvent(_fault_ramp, sub_events={
            'base': gen_bear_base_tgt, 'fs': fsc, 'fe': fec,
            'delta': ConstantEvent(d_gb_t)})
    else:
        gen_bear_tgt_ev = gen_bear_base_tgt

    gen_bear = LambdaEvent(thermal_inertia, sub_events={
        'target': gen_bear_tgt_ev, 'alpha': ConstantEvent(0.04), 'noise': ConstantEvent(0.30)})

    # Build nacelle temp
    nacelle_base_tgt = LambdaEvent(_nacelle_temp_tgt, sub_events={'power': power})
    if fault_type == "gearbox_cooling_fault":
        # A-class fix: 齿轮箱冷却故障 → 热量向机舱扩散 → nacelle_temp 上升
        d_nac = compute_fault_delta(_CH['nacelle_temp'], severity, "increase") * 0.5
        nacelle_tgt_ev = LambdaEvent(_fault_ramp, sub_events={
            'base': nacelle_base_tgt, 'fs': fsc, 'fe': fec,
            'delta': ConstantEvent(d_nac)})
    else:
        nacelle_tgt_ev = nacelle_base_tgt
    nacelle = LambdaEvent(thermal_inertia, sub_events={
        'target': nacelle_tgt_ev, 'alpha': ConstantEvent(0.06), 'noise': ConstantEvent(0.35)})

    return {
        'wind_speed': wind_final, 'rotor_speed': rotor, 'generator_speed': gen_spd,
        'power_output': power, 'pitch_angle': pitch,
        'nacelle_temp': nacelle, 'gearbox_oil_temp': gearbox,
        'gen_bearing_temp': gen_bear,
    }


# ============================================================
# 故障定义表
# ============================================================
WIND_TURBINE_FAULTS = {
    "gearbox_bearing_wear": (
        ["gearbox_oil_temp", "power_output"],
        ["齿轮箱轴承磨损加剧", "摩擦系数增大", "油温渐进式上升",
         "润滑性能下降", "传动效率降低", "功率输出下降"],
        "风力发电机齿轮箱轴承磨损故障"),
    "pitch_stuck": (
        ["pitch_angle", "rotor_speed", "power_output", "generator_speed"],
        ["变桨液压系统故障", "桨距角卡在固定位置",
         "高风速时无法限速", "转子转速异常", "功率偏离正常曲线"],
        "风力发电机变桨系统卡死故障"),
    "generator_overheat": (
        ["gen_bearing_temp", "power_output"],
        ["发电机绝缘老化", "等效电阻增大", "铜损增加",
         "轴承温度渐进上升", "触发降额保护", "功率限制输出"],
        "风力发电机发电机过热故障"),
    "yaw_misalignment": (
        ["power_output"],
        ["偏航驱动系统故障", "机舱朝向偏离风向",
         "迎风角偏差导致能量捕获效率下降", "风速正常但功率偏低"],
        "风力发电机偏航误差故障"),
    "blade_icing": (
        ["power_output", "rotor_speed", "generator_speed"],
        ["叶片表面结冰", "气动外形改变", "升力系数下降",
         "转子转速降低", "功率输出同步下降"],
        "风力发电机叶片结冰"),
    "gearbox_cooling_fault": (
        ["gearbox_oil_temp", "nacelle_temp"],
        ["齿轮箱冷却系统散热器堵塞或风扇故障", "冷却能力下降",
         "油温散热变慢导致持续升高", "热量向机舱扩散"],
        "风力发电机齿轮箱冷却系统故障"),
    "power_curve_deviation": (
        ["power_output"],
        ["叶片表面污染或前缘侵蚀", "气动性能劣化",
         "功率系数Cp下降", "同等风速下功率输出偏低"],
        "风力发电机功率曲线偏差"),
    "anemometer_drift": (
        ["wind_speed"],
        ["机舱风速仪标定漂移", "测量值渐进式偏离真实值",
         "其他通道无异常但风速-功率关系偏移"],
        "风力发电机风速仪漂移"),
    "gen_bearing_wear": (
        ["gen_bearing_temp"],
        ["发电机前端轴承磨损", "局部摩擦增大",
         "轴承温度渐进式升高", "电流和功率暂无明显变化"],
        "风力发电机发电机轴承磨损"),
    "grid_disturbance": (
        ["power_output", "generator_speed"],
        ["电网频率或电压波动", "并网逆变器调节受扰",
         "有功功率出现周期性振荡", "发电机转速轻微波动"],
        "电网扰动导致功率振荡"),
}


# ============================================================
# 统一生成接口
# ============================================================
def generate_wind_turbine(total_length: int = 5000, seed: int = 42,
                           fault_type: str = "normal",
                           severity: str = "moderate",
                           wind_mode: str = "steady",
                           randomize_fault_window: bool = False,
                           fault_start_range: Tuple[float, float] = (0.10, 0.80),
                           fault_duration_range: Tuple[float, float] = (0.08, 0.25),
                           severity_jitter: float = 0.0,
                           ) -> Tuple[pd.DataFrame, GenerationMetadata]:
    np.random.seed(seed)

    start_date = pd.Timestamp('2024-01-15')
    end_date = start_date + pd.Timedelta(seconds=SAMPLING_SECONDS * total_length)

    wind_fn = {"steady": _wind_steady, "varying": _wind_varying,
               "turbulent": _wind_turbulent}.get(wind_mode, _wind_steady)

    # Fault window
    if randomize_fault_window:
        _rng = np.random.RandomState(seed + 10007)
        dur_ratio = _rng.uniform(*fault_duration_range)
        fault_dur = max(20, int(total_length * dur_ratio))
        earliest = int(total_length * fault_start_range[0])
        latest = max(earliest, int(total_length * fault_start_range[1]) - fault_dur)
        fault_start = _rng.randint(earliest, latest + 1)
        fault_end = min(fault_start + fault_dur, total_length - 1)
    else:
        fault_start = int(total_length * 0.55)
        fault_end = min(int(fault_start + total_length * 0.2), total_length - 1)

    events_meta = []

    if fault_type == "normal":
        model, _ = _build_base_model(wind_fn)
    elif fault_type in WIND_TURBINE_FAULTS:
        model = _inject_fault(wind_fn, fault_type, fault_start, fault_end, severity, {})
        fdef = WIND_TURBINE_FAULTS[fault_type]
        sev_desc = {"mild": "轻微", "moderate": "中等", "severe": "严重"}[severity]
        sev_map = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}
        events_meta.append(EventMetadata(
            event_type="fault", fault_name=fault_type,
            time_range=(fault_start, fault_end), severity=severity,
            parameters={
                "severity_factor": round(sev_map.get(severity, 0.6), 3),
                "randomized": randomize_fault_window,
            },
            affected_channels=fdef[0],
            causal_chain=fdef[1],
            description=f"{fdef[2]}（{sev_desc}级），第{fault_start}步至第{fault_end}步"))
    else:
        model, _ = _build_base_model(wind_fn)

    gen = Generator(start_date=start_date, end_date=end_date, freq='10T')
    df = gen.generate(model, n=1)

    el = [e.time_range[1] - e.time_range[0] for e in events_meta]
    difficulty = compute_difficulty(total_length, el)
    nr = sum(el) / total_length if events_meta else 0.0

    scenario_id = (f"wind_turbine_{fault_type}_{severity}"
                   if fault_type != "normal"
                   else f"wind_turbine_normal_{wind_mode}")
    metadata = GenerationMetadata(
        scenario_id=scenario_id,
        scenario_name=f"风力发电机-{fault_type}-{severity}",
        domain=DOMAIN, domain_display=DOMAIN_DISPLAY,
        total_length=total_length,
        sampling_rate=SAMPLING_RATE, sampling_seconds=SAMPLING_SECONDS,
        channels=WIND_TURBINE_CHANNELS,
        base_description=f"风力发电机组{wind_mode}风况运行。Senvion MM92, 2.05MW额定。",
        injected_events=events_meta,
        overall_description=(events_meta[0].description if events_meta
                             else f"风力发电机组正常{wind_mode}风况运行。"),
        generation_seed=seed, difficulty=difficulty, needle_ratio=round(nr, 4),
    )
    return df, metadata


# ============================================================
# 场景注册
# ============================================================
def register_all_wind_turbine_tb():
    for mode in ["steady", "varying", "turbulent"]:
        sid = f"normal_{mode}"
        def gen(total_length=5000, seed=42, _m=mode, **kwargs):
            return generate_wind_turbine(total_length, seed, "normal", "moderate", _m, **kwargs)
        register_scenario(DOMAIN, sid, f"正常-{mode}", gen)

    for ft in WIND_TURBINE_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            sid = f"{ft}_{sev}"
            def gen(total_length=5000, seed=42, _ft=ft, _sev=sev, **kwargs):
                return generate_wind_turbine(total_length, seed, _ft, _sev, "steady", **kwargs)
            register_scenario(DOMAIN, sid, f"{ft}-{sev}", gen)


register_all_wind_turbine_tb()


# ============================================================
# 验证脚本
# ============================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== Wind Turbine TB — Fault Verification ===")
    os.makedirs('Data/WindTurbine_TB/fault_samples', exist_ok=True)

    test_faults = ["normal"] + list(WIND_TURBINE_FAULTS.keys())

    for ft in test_faults:
        sev = "moderate" if ft != "normal" else "moderate"
        df, meta = generate_wind_turbine(5000, 42, ft, sev, "steady")
        print(f"  {ft}: shape={df.shape}, events={len(meta.injected_events)}")

        fig, axes = plt.subplots(8, 1, figsize=(16, 20), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for idx, col in enumerate(df.columns):
            axes[idx].plot(df[col].values, color=colors[idx], linewidth=0.6)
            axes[idx].set_ylabel(col, fontsize=8)
            axes[idx].grid(True, alpha=0.2)
            for ev in meta.injected_events:
                if col in ev.affected_channels:
                    axes[idx].axvspan(ev.time_range[0], ev.time_range[1],
                                      alpha=0.15, color='red')
        axes[0].set_title(f"Wind Turbine: {ft} ({sev})", fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'Data/WindTurbine_TB/fault_samples/{ft}_{sev}.png', dpi=100)
        plt.close()

    print(f"\nAll {len(test_faults)} fault types verified.")
    print(f"Total registered scenarios: {len(get_all_scenarios(DOMAIN))}")
