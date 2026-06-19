"""
变压器 TimeBlender 场景库 — 基于 ETTh2 真实数据校准

真实数据校准目标（Dataset/ETT/ETTh2.csv, 17420行, 1h采样, ~2年）:
  HUFL: mean=37.2, std=10.2, AC1=0.950, 日周期(峰21h,谷8h)
  HULL: mean=8.5,  std=6.0,  AC1=0.955
  MUFL: mean=43.8, std=13.1, AC1=0.976
  MULL: mean=8.3,  std=4.4,  AC1=0.946
  LUFL: mean=-3.4, std=6.1,  AC1=0.985
  LULL: mean=-2.1, std=6.0,  AC1=0.995
  OT:   mean=26.6, std=11.9, AC1=0.994, 季节性趋势

关键：负荷通道AC1较低(0.95)→用较高alpha；OT的AC1高(0.994)→用低alpha
"""

import os, sys, numpy as np, pandas as pd
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from time_blender.core import Generator, LambdaEvent, ConstantEvent
from MultiAgentTS.scenarios.base import (
    GenerationMetadata, EventMetadata, ChannelInfo,
    compute_difficulty, register_scenario, get_all_scenarios, compute_fault_delta,
)

TRANSFORMER_CHANNELS = [
    ChannelInfo("HUFL", "高压侧有功负载", "MW", normal_range=(20.0, 55.0),
                alarm_threshold=90.0, alarm_low=5.0,
                physical_meaning="高压侧有功功率，低于5MW表示失负荷/解列"),
    ChannelInfo("HULL", "高压侧无功负载", "MVar", normal_range=(0.0, 20.0),
                alarm_threshold=35.0,
                physical_meaning="高压侧无功功率"),
    ChannelInfo("MUFL", "中压侧有功负载", "MW", normal_range=(25.0, 65.0),
                alarm_threshold=85.0, alarm_low=5.0,
                physical_meaning="中压侧有功功率，低于5MW表示失负荷"),
    ChannelInfo("MULL", "中压侧无功负载", "MVar", normal_range=(2.0, 15.0),
                alarm_threshold=25.0,
                physical_meaning="中压侧无功功率"),
    ChannelInfo("LUFL", "低压侧有功负载", "MW", normal_range=(-10.0, 10.0),
                alarm_threshold=15.0,
                physical_meaning="低压侧有功功率，可反向潮流"),
    ChannelInfo("LULL", "低压侧无功负载", "MVar", normal_range=(-15.0, 2.0),
                alarm_threshold=None,
                physical_meaning="低压侧无功功率"),
    ChannelInfo("OT", "油温", "°C", normal_range=(15.0, 45.0),
                alarm_threshold=65.0,
                physical_meaning="变压器油顶层温度，IEC 60076限值65°C"),
]

DOMAIN = "transformer"
DOMAIN_DISPLAY = "电力变压器"
SAMPLING_RATE = "1h"
SAMPLING_SECONDS = 3600

_CH = {ch.name: ch for ch in TRANSFORMER_CHANNELS}


# ============================================================
# TimeBlender 构建块
# ============================================================
def thermal_inertia(t, i, memory, sub_events):
    target = sub_events['target'].execute(t)
    alpha = sub_events['alpha'].execute(t)
    noise_std = sub_events['noise'].execute(t)
    if 'prev' not in memory:
        memory['prev'] = target
    memory['prev'] += alpha * (target - memory['prev']) + np.random.normal(0, noise_std)
    return memory['prev']


# ============================================================
# 日周期负荷模型
# ============================================================
def _daily_load(t, i, mem, sub):
    """24h日负荷曲线: 8-9h低谷, 12-14h午高峰, 19-22h晚高峰"""
    if 'base' not in mem:
        mem['base'] = 0.6
    hour = i % 24
    # 典型日负荷曲线
    if hour < 6:
        pattern = 0.4 + 0.05 * hour  # 凌晨低
    elif hour < 9:
        pattern = 0.65 - 0.05 * (hour - 6)  # 早晨谷
    elif hour < 12:
        pattern = 0.55 + 0.12 * (hour - 9)  # 上午爬升
    elif hour < 14:
        pattern = 0.85  # 午高峰
    elif hour < 17:
        pattern = 0.75 + 0.03 * (hour - 14)  # 下午平台
    elif hour < 22:
        pattern = 0.8 + 0.04 * (hour - 17)  # 晚高峰
    else:
        pattern = 0.9 - 0.15 * (hour - 22)  # 夜间下降

    # 缓慢趋势（模拟周/季节变化）
    mem['base'] = np.clip(mem['base'] + np.random.normal(0, 0.003), 0.4, 0.85)
    return pattern * mem['base']


# ============================================================
# 各通道目标值
# ============================================================
def _hufl_tgt(t, i, m, s):
    """HUFL = 20 + 45*load + noise，校准到 mean~37, std~10"""
    ld = s['load'].execute(t)
    return 20 + 50 * ld + np.random.normal(0, 4)

def _hull_tgt(t, i, m, s):
    """HULL与HUFL相关(0.67) + 独立波动"""
    hufl = s['hufl'].execute(t)
    return 0.3 * hufl + np.random.normal(0, 2)

def _mufl_tgt(t, i, m, s):
    """MUFL相对独立，mean~44, std~13"""
    ld = s['load'].execute(t)
    return 22 + 55 * ld + np.random.normal(0, 5)

def _mull_tgt(t, i, m, s):
    """MULL与HULL强相关(0.915)"""
    hull = s['hull'].execute(t)
    return 0.7 * hull + np.random.normal(0, 1.2)

def _lufl_tgt(t, i, m, s):
    """LUFL可为负，mean~-3.4"""
    ld = s['load'].execute(t)
    return -8 + 12 * ld + np.random.normal(0, 2)

def _lull_tgt(t, i, m, s):
    """LULL最稳定，mean~-2.1, std~6"""
    if 'val' not in m: m['val'] = -2.0
    m['val'] += np.random.normal(0, 0.08)
    m['val'] = np.clip(m['val'], -15, 3)
    return m['val']

def _ot_target(t, i, m, s):
    """油温 = f(负荷², 环境温度)，热惯性最大"""
    ld = s['load'].execute(t)
    # 环境温度：日温差(±5°C) + 季节趋势(±12°C)
    ambient = 15 + 6 * np.sin(2 * np.pi * i / 24 - np.pi / 2)  # 日温差
    ambient += 12 * np.sin(2 * np.pi * i / (24 * 365))  # 季节性幅度增大
    thermal_load = (ld ** 1.5) * 35  # I²R 损耗
    return ambient + thermal_load + np.random.normal(0, 1.0)


# ============================================================
# 故障修改器
# ============================================================
def _fault_ramp(t, i, m, s):
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    delta = s['delta'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        return base + delta * (np.exp(2 * p) - 1) / (np.exp(2) - 1)
    elif i >= fe: return base + delta
    return base

def _fault_step(t, i, m, s):
    base = s['base'].execute(t)
    return base + s['delta'].execute(t) if i >= int(s['fs'].execute(t)) else base

def _fault_oscillation(t, i, m, s):
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    if fs <= i < fe:
        return base + s['amp'].execute(t) * np.sin(2*np.pi*(i-fs)/s['period'].execute(t))
    return base

def _fault_spike(t, i, m, s):
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    if fs <= i < fe and np.random.random() < s['prob'].execute(t):
        return base + np.random.uniform(0.5, 1.0) * s['amp'].execute(t)
    return base


# ============================================================
# 模型构建
# ============================================================
def _build_base():
    load = LambdaEvent(_daily_load, sub_events={})
    hufl_t = LambdaEvent(_hufl_tgt, sub_events={'load': load})
    hufl = LambdaEvent(thermal_inertia, sub_events={
        'target': hufl_t, 'alpha': ConstantEvent(0.4), 'noise': ConstantEvent(1.5)})
    hull_t = LambdaEvent(_hull_tgt, sub_events={'hufl': hufl})
    hull = LambdaEvent(thermal_inertia, sub_events={
        'target': hull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(1.0)})
    mufl_t = LambdaEvent(_mufl_tgt, sub_events={'load': load})
    mufl = LambdaEvent(thermal_inertia, sub_events={
        'target': mufl_t, 'alpha': ConstantEvent(0.25), 'noise': ConstantEvent(2.0)})
    mull_t = LambdaEvent(_mull_tgt, sub_events={'hull': hull})
    mull = LambdaEvent(thermal_inertia, sub_events={
        'target': mull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(0.8)})
    lufl_t = LambdaEvent(_lufl_tgt, sub_events={'load': load})
    lufl = LambdaEvent(thermal_inertia, sub_events={
        'target': lufl_t, 'alpha': ConstantEvent(0.15), 'noise': ConstantEvent(1.0)})
    lull_t = LambdaEvent(_lull_tgt, sub_events={})
    lull = LambdaEvent(thermal_inertia, sub_events={
        'target': lull_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.3)})
    ot_t = LambdaEvent(_ot_target, sub_events={'load': load})
    ot = LambdaEvent(thermal_inertia, sub_events={
        'target': ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.3)})

    events = {'HUFL': hufl, 'HULL': hull, 'MUFL': mufl, 'MULL': mull,
              'LUFL': lufl, 'LULL': lull, 'OT': ot}
    refs = {'load': load, 'hufl': hufl, 'hull': hull,
            'hufl_t': hufl_t, 'hull_t': hull_t, 'mufl_t': mufl_t,
            'mull_t': mull_t, 'ot_t': ot_t}
    return events, refs


def _inject_transformer_fault(fault_type, fs, fe, severity):
    """所有 delta 基于 compute_fault_delta()，消除幽灵通道。"""
    fsc, fec = ConstantEvent(float(fs)), ConstantEvent(float(fe))

    events, refs = _build_base()

    # ==== A-class: 修复幽灵通道 ====

    if fault_type == "overload":
        # 根因: 超容量运行 → HUFL + MUFL 阶跃上升 + OT 热响应
        d_hufl = compute_fault_delta(_CH['HUFL'], severity, "increase")
        new_hufl_t = LambdaEvent(_fault_step, sub_events={
            'base': refs['hufl_t'], 'fs': fsc, 'delta': ConstantEvent(d_hufl)})
        events['HUFL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hufl_t, 'alpha': ConstantEvent(0.4), 'noise': ConstantEvent(1.5)})
        # 补充: MUFL 同步增加 (~80%)
        d_mufl = compute_fault_delta(_CH['MUFL'], severity, "increase") * 0.8
        new_mufl_t = LambdaEvent(_fault_step, sub_events={
            'base': refs['mufl_t'], 'fs': fsc, 'delta': ConstantEvent(d_mufl)})
        events['MUFL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_mufl_t, 'alpha': ConstantEvent(0.25), 'noise': ConstantEvent(2.0)})
        # 补充: I²R → OT 渐升 (~60%)
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase") * 0.6
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.3)})

    elif fault_type == "insulation_aging":
        # 根因: 绝缘劣化 → 介质损耗增大
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase")
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.3)})
        # 补充: 绝缘劣化 → HULL 无功损耗增加
        d_hull = compute_fault_delta(_CH['HULL'], severity, "increase") * 0.4
        new_hull_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['hull_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_hull)})
        events['HULL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(1.0)})
        # 补充: MULL 跟随 HULL 增加 (~50%)
        d_mull = compute_fault_delta(_CH['MULL'], severity, "increase") * 0.5
        new_mull_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['mull_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_mull)})
        events['MULL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_mull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(0.8)})

    elif fault_type == "load_surge":
        # 根因: 大型设备启动 → HUFL + MUFL 阶跃 + OT 滞后
        d_hufl = compute_fault_delta(_CH['HUFL'], severity, "increase")
        new_hufl_t = LambdaEvent(_fault_step, sub_events={
            'base': refs['hufl_t'], 'fs': fsc, 'delta': ConstantEvent(d_hufl)})
        events['HUFL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hufl_t, 'alpha': ConstantEvent(0.4), 'noise': ConstantEvent(1.5)})
        d_mufl = compute_fault_delta(_CH['MUFL'], severity, "increase") * 0.8
        new_mufl_t = LambdaEvent(_fault_step, sub_events={
            'base': refs['mufl_t'], 'fs': fsc, 'delta': ConstantEvent(d_mufl)})
        events['MUFL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_mufl_t, 'alpha': ConstantEvent(0.25), 'noise': ConstantEvent(2.0)})
        # 补充: OT 热滞后响应 (~40%)
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase") * 0.4
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.3)})

    elif fault_type == "partial_discharge":
        # 根因: 绝缘缺陷 → HULL 间歇性脉冲
        d_hull = compute_fault_delta(_CH['HULL'], severity, "increase")
        new_hull_t = LambdaEvent(_fault_spike, sub_events={
            'base': refs['hull_t'],
            'fs': fsc, 'fe': fec, 'amp': ConstantEvent(d_hull), 'prob': ConstantEvent(0.05)})
        events['HULL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(1.0)})
        # 补充: 局部热点 → OT 小幅上升 (~20%)
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase") * 0.2
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.3)})

    elif fault_type == "winding_deformation":
        # 根因: 绕组变形 → 阻抗变化 → 损耗增大
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase")
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.5)})
        # 补充: 阻抗变化 → HULL 无功波动 (~30%)
        d_hull = compute_fault_delta(_CH['HULL'], severity, "increase") * 0.3
        new_hull_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['hull_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_hull)})
        events['HULL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(1.0)})

    elif fault_type == "harmonic_distortion":
        # 根因: 电网谐波注入 → MUFL 振荡
        d_mufl = compute_fault_delta(_CH['MUFL'], severity, "increase") * 0.5
        new_mufl_t = LambdaEvent(_fault_oscillation, sub_events={
            'base': refs['mufl_t'], 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_mufl), 'period': ConstantEvent(6.0)})
        events['MUFL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_mufl_t, 'alpha': ConstantEvent(0.25), 'noise': ConstantEvent(2.0)})
        # 补充: 谐波 → HULL 无功波动
        d_hull = compute_fault_delta(_CH['HULL'], severity, "increase") * 0.4
        new_hull_t = LambdaEvent(_fault_oscillation, sub_events={
            'base': refs['hull_t'], 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_hull), 'period': ConstantEvent(6.0)})
        events['HULL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hull_t, 'alpha': ConstantEvent(0.35), 'noise': ConstantEvent(1.0)})

    # ==== C-class: 仅 delta 校准 ====

    elif fault_type == "oil_temp_anomaly":
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase")
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.3)})

    elif fault_type == "cooling_fan_fault":
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase")
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.015), 'noise': ConstantEvent(0.3)})

    elif fault_type == "tap_changer_fault":
        d_hufl = compute_fault_delta(_CH['HUFL'], severity, "increase") * 0.8
        new_hufl_t = LambdaEvent(_fault_oscillation, sub_events={
            'base': refs['hufl_t'], 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_hufl), 'period': ConstantEvent(8.0)})
        events['HUFL'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_hufl_t, 'alpha': ConstantEvent(0.4), 'noise': ConstantEvent(1.5)})

    elif fault_type == "oil_leak":
        d_ot = compute_fault_delta(_CH['OT'], severity, "increase")
        new_ot_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['ot_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ot)})
        events['OT'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_ot_t, 'alpha': ConstantEvent(0.03), 'noise': ConstantEvent(0.8)})

    return events


# ============================================================
# 故障定义表
# ============================================================
TRANSFORMER_FAULTS = {
    "overload": (["HUFL", "MUFL", "OT"], ["电网调度超容量运行", "负荷升高", "I²R损耗增加", "油温上升"],
                 "变压器过载运行"),
    "oil_temp_anomaly": (["OT"], ["冷却系统效率下降", "散热不足", "油温异常升高", "负荷不变"],
                         "变压器油温异常"),
    "insulation_aging": (["OT", "HULL", "MULL"], ["绝缘材料劣化", "介质损耗增大", "油温基线缓慢爬升"],
                         "变压器绝缘老化"),
    "load_surge": (["HUFL", "MUFL", "OT"], ["大型用电设备启动", "负荷阶跃变化", "油温滞后响应"],
                   "变压器负载突变"),
    "partial_discharge": (["HULL", "OT"], ["绝缘缺陷处局部放电", "无功间歇性脉冲", "局部热点"],
                          "变压器局部放电"),
    "cooling_fan_fault": (["OT"], ["冷却风扇转速下降", "散热能力降低", "油温渐升", "负荷不变"],
                          "变压器冷却风扇故障"),
    "tap_changer_fault": (["HUFL"], ["有载调压开关异常", "负荷周期性振荡"],
                          "变压器有载调压故障"),
    "winding_deformation": (["OT", "HULL"], ["绕组变形", "阻抗变化", "损耗增大", "油温渐升"],
                            "变压器绕组变形"),
    "oil_leak": (["OT"], ["油箱密封老化", "油位下降", "散热面积减小", "油温升高"],
                 "变压器漏油"),
    "harmonic_distortion": (["MUFL", "HULL"], ["电网谐波注入", "中压侧功率振荡", "无功波动"],
                            "电网谐波畸变"),
}


# ============================================================
# 统一生成接口
# ============================================================
def generate_transformer(total_length=5000, seed=42, fault_type="normal",
                          severity="moderate",
                          randomize_fault_window: bool = False,
                          fault_start_range=(0.10, 0.80),
                          fault_duration_range=(0.08, 0.25),
                          severity_jitter: float = 0.0,
                          ) -> Tuple[pd.DataFrame, GenerationMetadata]:
    np.random.seed(seed)
    start = pd.Timestamp('2016-07-01')
    end = start + pd.Timedelta(hours=total_length)

    # --- 故障窗口计算 ---
    if randomize_fault_window:
        _rng = np.random.RandomState(seed + 10007)
        dur_ratio = _rng.uniform(*fault_duration_range)
        fault_dur = max(20, int(total_length * dur_ratio))
        earliest = int(total_length * fault_start_range[0])
        latest = max(earliest, int(total_length * fault_start_range[1]) - fault_dur)
        fs = _rng.randint(earliest, latest + 1)
        fe = min(fs + fault_dur, total_length - 1)
    else:
        fs = int(total_length * 0.55)
        fe = min(int(fs + total_length * 0.2), total_length - 1)

    base_sev_factor = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}.get(severity, 0.6)
    if severity_jitter > 0:
        _rng_sev = np.random.RandomState(seed + 20013)
        sev_factor = base_sev_factor * (1 + _rng_sev.uniform(-severity_jitter, severity_jitter))
    else:
        sev_factor = base_sev_factor

    events_meta = []

    if fault_type == "normal":
        model, _ = _build_base()
    elif fault_type in TRANSFORMER_FAULTS:
        model = _inject_transformer_fault(fault_type, fs, fe, severity)
        fdef = TRANSFORMER_FAULTS[fault_type]
        sev_desc = {"mild": "轻微", "moderate": "中等", "severe": "严重"}[severity]
        events_meta.append(EventMetadata(
            event_type="fault", fault_name=fault_type,
            time_range=(fs, fe), severity=severity,
            parameters={"severity_factor": round(sev_factor, 3), "randomized": randomize_fault_window},
            affected_channels=fdef[0], causal_chain=fdef[1],
            description=f"{fdef[2]}（{sev_desc}级），第{fs}步至第{fe}步"))
    else:
        model, _ = _build_base()

    gen = Generator(start_date=start, end_date=end, freq='1H')
    df = gen.generate(model, n=1)

    el = [e.time_range[1] - e.time_range[0] for e in events_meta]
    sid = f"transformer_{fault_type}_{severity}" if fault_type != "normal" else "transformer_normal"

    metadata = GenerationMetadata(
        scenario_id=sid, scenario_name=f"变压器-{fault_type}-{severity}",
        domain=DOMAIN, domain_display=DOMAIN_DISPLAY,
        total_length=total_length, sampling_rate=SAMPLING_RATE,
        sampling_seconds=SAMPLING_SECONDS, channels=TRANSFORMER_CHANNELS,
        base_description="变压器正常运行，含日周期和季节性。",
        injected_events=events_meta,
        overall_description=events_meta[0].description if events_meta else "变压器正常运行。",
        generation_seed=seed, difficulty=compute_difficulty(total_length, el),
        needle_ratio=round(sum(el)/total_length, 4) if el else 0.0,
    )
    return df, metadata


# ============================================================
# 场景注册
# ============================================================
def register_all_transformer_tb():
    def gen(total_length=5000, seed=42, **kwargs):
        return generate_transformer(total_length, seed, "normal", **kwargs)
    register_scenario(DOMAIN, "normal", "正常运行", gen)

    for ft in TRANSFORMER_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            sid = f"{ft}_{sev}"
            def gen(total_length=5000, seed=42, _ft=ft, _sev=sev, **kwargs):
                return generate_transformer(total_length, seed, _ft, _sev, **kwargs)
            register_scenario(DOMAIN, sid, f"{ft}-{sev}", gen)

register_all_transformer_tb()


# ============================================================
# 验证
# ============================================================
if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== Transformer TB Verification ===")

    # 加载真实数据对比
    df_real = pd.read_csv('Dataset/ETT/ETTh2.csv').iloc[:5001, 1:]

    # 正常样本
    df_s, meta = generate_transformer(5000, 42, "normal")
    print(f"Normal: {df_s.shape}")

    print("\n{:<12} {:>10} {:>10} {:>10} {:>10}".format("Channel", "Real Mean", "Synth Mean", "Real Std", "Synth Std"))
    for rc, sc in zip(df_real.columns, df_s.columns):
        print("{:<12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            sc, df_real[rc].mean(), df_s[sc].mean(), df_real[rc].std(), df_s[sc].std()))

    # 对比图
    fig, axes = plt.subplots(7, 1, figsize=(18, 21), sharex=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for idx, (rc, sc) in enumerate(zip(df_real.columns, df_s.columns)):
        axes[idx].plot(df_real[rc].values, color='#1f77b4', linewidth=0.5, alpha=0.8, label='Real')
        axes[idx].plot(df_s[sc].values, color='#d62728', linewidth=0.5, alpha=0.8, label='TB Synth')
        axes[idx].set_ylabel(sc, fontsize=8)
        axes[idx].legend(fontsize=7)
        axes[idx].grid(True, alpha=0.2)
    axes[0].set_title('Transformer: Real vs TB Synth', fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs('Data/Transformer_TB', exist_ok=True)
    plt.savefig('Data/Transformer_TB/comparison.png', dpi=150)
    plt.close()
    print("\nSaved: Data/Transformer_TB/comparison.png")
    print(f"Registered: {len(get_all_scenarios(DOMAIN))} scenarios")
