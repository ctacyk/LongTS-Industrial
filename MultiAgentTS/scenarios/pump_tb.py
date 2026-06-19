"""
水循环泵 TimeBlender 场景库 — 基于 SKAB 真实数据校准

真实数据校准目标（SKAB 0.csv, 1125行, 1s采样）:
  Accelerometer1RMS: mean=0.028, std=0.001, AC1=0.28 (快变)
  Accelerometer2RMS: mean=0.039, std=0.001, AC1=0.53
  Current:           mean=0.97,  std=0.27,  AC1=0.40 (近随机)
  Pressure:          mean=0.04,  std=0.26,  AC1=0.05 (白噪声)
  Temperature:       mean=68.76, std=1.44,  AC1=0.995 (慢变，热惯性)
  Thermocouple:      mean=24.33, std=0.03,  AC1=0.976
  Voltage:           mean=230.9, std=10.8,  AC1=-0.01 (随机)
  Flow Rate:         mean=32.04, std=0.58,  AC1=0.19 (离散)

关键：大部分通道AC1很低→用高alpha或直接随机；只有Temperature/Thermocouple用低alpha热惯性
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

PUMP_CHANNELS = [
    ChannelInfo("vibration_pump", "泵侧振动RMS", "g",
                normal_range=(0.025, 0.035), alarm_threshold=0.08,
                physical_meaning="泵体振动加速度RMS，ISO 10816 Good级"),
    ChannelInfo("vibration_motor", "电机侧振动RMS", "g",
                normal_range=(0.035, 0.045), alarm_threshold=0.10,
                physical_meaning="电机侧振动加速度RMS，ISO 10816 Good级"),
    ChannelInfo("motor_current", "电机电流", "A",
                normal_range=(0.5, 1.5), alarm_threshold=2.0, alarm_low=0.2,
                physical_meaning="小型水泵电机电流，低于0.2A表示干转/断相"),
    ChannelInfo("pressure", "泵出口压力", "Bar",
                normal_range=(2.0, 3.0), alarm_threshold=4.0, alarm_low=1.0,
                physical_meaning="循环管路出口表压，典型工业循环泵2-3Bar"),
    ChannelInfo("motor_temp", "电机体温度", "°C",
                normal_range=(65.0, 72.0), alarm_threshold=85.0,
                physical_meaning="电机壳体温度，B级绝缘限值130°C"),
    ChannelInfo("fluid_temp", "循环液温度", "°C",
                normal_range=(24.2, 24.5), alarm_threshold=30.0,
                physical_meaning="循环管路流体温度"),
    ChannelInfo("motor_voltage", "电机电压", "V",
                normal_range=(225.0, 235.0), alarm_threshold=253.0, alarm_low=198.0,
                physical_meaning="电机供电电压，IEC标准230V +10%/-14%"),
    ChannelInfo("flow_rate", "循环流量", "L/min",
                normal_range=(31.0, 33.0), alarm_threshold=35.0, alarm_low=28.0,
                physical_meaning="循环管路流量，额定32L/min ±10%报警"),
]

DOMAIN = "pump"
DOMAIN_DISPLAY = "水循环泵系统"
SAMPLING_RATE = "1s"
SAMPLING_SECONDS = 1

_CH = {ch.name: ch for ch in PUMP_CHANNELS}


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
# 通道目标值（正常运行）
# ============================================================
def _vib_pump(t, i, m, s):
    """泵振动：低AC，近白噪声"""
    return 0.028 + np.random.normal(0, 0.0008)

def _vib_motor(t, i, m, s):
    """电机振动：略高AC"""
    if 'p' not in m: m['p'] = 0.039
    m['p'] = 0.039 + 0.5 * (m['p'] - 0.039) + np.random.normal(0, 0.0006)
    return m['p']

def _current(t, i, m, s):
    """电流：低AC，宽幅随机"""
    return 0.97 + np.random.normal(0, 0.25) * (0.5 + 0.5 * abs(np.sin(i / 50)))

def _pressure_target(t, i, m, s):
    """泵出口压力目标：典型工业循环泵 2-3 Bar，缓慢漂移
    物理依据：管路阻力和泵转速决定出口压力，正常运行时缓慢波动
    """
    if 'base' not in m:
        m['base'] = 2.5
    m['base'] += np.random.normal(0, 0.008)
    m['base'] = np.clip(m['base'], 2.0, 3.0)
    return m['base']

def _temp_target(t, i, m, s):
    """电机温度目标：缓慢变化，std~1.4°C"""
    if 'base' not in m: m['base'] = 69.0
    # 增大漂移幅度以匹配真实std=1.44
    m['base'] += np.random.normal(0, 0.05)
    m['base'] = np.clip(m['base'], 64, 73)
    return m['base']

def _thermo_target(t, i, m, s):
    """热电偶温度目标，std~0.03"""
    if 'base' not in m: m['base'] = 24.33
    m['base'] += np.random.normal(0, 0.003)
    m['base'] = np.clip(m['base'], 24.15, 24.55)
    return m['base']

def _voltage_target(t, i, m, s):
    """电机电压目标：230V ±5V 缓慢漂移
    物理依据：工业电网电压受负荷和变压器调压器影响，呈缓慢变化
    IEC标准230V +10%/-14%，正常波动远小于此范围
    """
    if 'base' not in m:
        m['base'] = 230.0
    m['base'] += np.random.normal(0, 0.15)
    m['base'] = np.clip(m['base'], 225.0, 235.0)
    return m['base']

def _flow_target(t, i, m, s):
    """循环流量目标：32 L/min 连续平滑变化
    物理依据：管路流量由泵转速和阀门开度决定，变化缓慢
    额定32 L/min ±1 L/min 为正常波动
    """
    if 'base' not in m:
        m['base'] = 32.0
    m['base'] += np.random.normal(0, 0.02)
    m['base'] = np.clip(m['base'], 31.0, 33.0)
    return m['base']


# ============================================================
# 故障修改器
# ============================================================
def _fault_ramp(t, i, m, s):
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    delta = s['delta'].execute(t)
    if fs <= i < fe:
        p = (i - fs) / max(fe - fs, 1)
        return base + delta * (np.exp(2*p)-1)/(np.exp(2)-1)
    elif i >= fe: return base + delta
    return base

def _fault_step(t, i, m, s):
    base = s['base'].execute(t)
    return base + s['delta'].execute(t) if i >= int(s['fs'].execute(t)) else base

def _fault_burst(t, i, m, s):
    """间歇性突发（气蚀用）"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    amp = s['amp'].execute(t)
    if fs <= i < fe:
        # 间歇性：70%时间有突发，30%正常
        burst = np.random.choice([0, 1], p=[0.3, 0.7])
        if burst:
            return base + np.random.exponential(amp)
        return base
    return base


# ============================================================
# 模型构建
# ============================================================
def _build_base():
    vp = LambdaEvent(_vib_pump, sub_events={})
    vm = LambdaEvent(_vib_motor, sub_events={})
    cur = LambdaEvent(_current, sub_events={})

    # 压力：thermal_inertia 平滑，mean~2.5 Bar, std~0.15, AC1>0.95
    pres_t = LambdaEvent(_pressure_target, sub_events={})
    pres = LambdaEvent(thermal_inertia, sub_events={
        'target': pres_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.02)})

    tt = LambdaEvent(_temp_target, sub_events={})
    temp = LambdaEvent(thermal_inertia, sub_events={
        'target': tt, 'alpha': ConstantEvent(0.005), 'noise': ConstantEvent(0.05)})
    tht = LambdaEvent(_thermo_target, sub_events={})
    thermo = LambdaEvent(thermal_inertia, sub_events={
        'target': tht, 'alpha': ConstantEvent(0.02), 'noise': ConstantEvent(0.003)})

    # 电压：thermal_inertia 平滑，mean~230V, std~2-3V, AC1>0.95
    volt_t = LambdaEvent(_voltage_target, sub_events={})
    volt = LambdaEvent(thermal_inertia, sub_events={
        'target': volt_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.3)})

    # 流量：thermal_inertia 极平滑，mean~32 L/min, std~0.3, AC1>0.99
    flow_t = LambdaEvent(_flow_target, sub_events={})
    flow = LambdaEvent(thermal_inertia, sub_events={
        'target': flow_t, 'alpha': ConstantEvent(0.01), 'noise': ConstantEvent(0.02)})

    events = {
        'vibration_pump': vp, 'vibration_motor': vm, 'motor_current': cur,
        'pressure': pres, 'motor_temp': temp, 'fluid_temp': thermo,
        'motor_voltage': volt, 'flow_rate': flow,
    }
    refs = {
        'vp': vp, 'vm': vm, 'cur': cur,
        'pres': pres_t, 'tt': tt, 'tht': tht,
        'volt_t': volt_t, 'flow_t': flow_t,
    }
    return events, refs


def _inject_pump_fault(fault_type, fs, fe, severity):
    """所有 delta 基于 compute_fault_delta() 从报警阈值自动校准，消除幽灵通道。"""
    fsc, fec = ConstantEvent(float(fs)), ConstantEvent(float(fe))
    events, refs = _build_base()

    # ==== A-class: 修复幽灵通道 + 根因驱动耦合 ====

    if fault_type == "cavitation":
        # 根因: 入口压力低于饱和蒸汽压 → 气泡破裂
        d_vp = compute_fault_delta(_CH['vibration_pump'], severity, "increase")
        events['vibration_pump'] = LambdaEvent(_fault_burst, sub_events={
            'base': refs['vp'], 'fs': fsc, 'fe': fec, 'amp': ConstantEvent(d_vp)})
        events['vibration_motor'] = LambdaEvent(_fault_burst, sub_events={
            'base': refs['vm'], 'fs': fsc, 'fe': fec, 'amp': ConstantEvent(d_vp * 0.5)})
        # 补充: 气蚀 → 压力波动增大 + 平均压力下降
        d_p = compute_fault_delta(_CH['pressure'], severity, "decrease") * 0.5
        new_pres_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['pres'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_p)})
        events['pressure'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_pres_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.02)})

    elif fault_type == "seal_leak":
        # 根因: 机械密封老化 → 泄漏通道形成
        d_p = compute_fault_delta(_CH['pressure'], severity, "decrease")
        new_pres_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['pres'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_p)})
        events['pressure'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_pres_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.02)})
        # 补充: 泄漏 → 流量减少
        d_f = compute_fault_delta(_CH['flow_rate'], severity, "decrease")
        new_flow_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['flow_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_f)})
        events['flow_rate'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_flow_t, 'alpha': ConstantEvent(0.01), 'noise': ConstantEvent(0.02)})

    elif fault_type == "pipe_blockage":
        # 根因: 管道沉积物 → 过流面积减小
        d_p = compute_fault_delta(_CH['pressure'], severity, "increase")
        new_pres_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['pres'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_p)})
        events['pressure'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_pres_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.02)})
        # 补充: 阻力增加 → 恒功率负载 → 电流上升
        d_c = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.4
        events['motor_current'] = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['cur'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_c)})

    elif fault_type == "impeller_damage":
        # 根因: 叶轮叶片断裂/腐蚀 → 质量不平衡
        d_vp = compute_fault_delta(_CH['vibration_pump'], severity, "increase")
        events['vibration_pump'] = LambdaEvent(_fault_step, sub_events={
            'base': refs['vp'], 'fs': fsc, 'delta': ConstantEvent(d_vp)})
        # 补充: 效率下降 → 流量减少
        d_f = compute_fault_delta(_CH['flow_rate'], severity, "decrease") * 0.5
        new_flow_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['flow_t'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_f)})
        events['flow_rate'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_flow_t, 'alpha': ConstantEvent(0.01), 'noise': ConstantEvent(0.02)})
        # 补充: 效率下降 → 压力下降
        d_p = compute_fault_delta(_CH['pressure'], severity, "decrease") * 0.3
        new_pres_t = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['pres'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_p)})
        events['pressure'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_pres_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.02)})

    elif fault_type == "dry_run":
        # 根因: 液位过低 → 泵体失去液封
        d_c = compute_fault_delta(_CH['motor_current'], severity, "decrease")
        events['motor_current'] = LambdaEvent(_fault_step, sub_events={
            'base': refs['cur'], 'fs': fsc, 'delta': ConstantEvent(d_c)})
        d_vp = compute_fault_delta(_CH['vibration_pump'], severity, "increase")
        events['vibration_pump'] = LambdaEvent(_fault_step, sub_events={
            'base': refs['vp'], 'fs': fsc, 'delta': ConstantEvent(d_vp)})
        # 补充: 压力趋零
        d_p = compute_fault_delta(_CH['pressure'], severity, "decrease")
        new_pres_t = LambdaEvent(_fault_step, sub_events={
            'base': refs['pres'], 'fs': fsc, 'delta': ConstantEvent(d_p)})
        events['pressure'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_pres_t, 'alpha': ConstantEvent(0.05), 'noise': ConstantEvent(0.02)})

    elif fault_type == "voltage_sag":
        # 根因: 供电电压骤降
        d_v = compute_fault_delta(_CH['motor_voltage'], severity, "decrease")
        events['motor_voltage'] = LambdaEvent(_fault_step, sub_events={
            'base': refs['volt_t'], 'fs': fsc, 'delta': ConstantEvent(d_v)})
        # 补充: 恒功率负载 → voltage↓ → current↑
        d_c = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.3
        events['motor_current'] = LambdaEvent(_fault_step, sub_events={
            'base': refs['cur'], 'fs': fsc, 'delta': ConstantEvent(d_c)})

    # ==== B-class: 补充缺失耦合 ====

    elif fault_type == "flow_oscillation":
        # 管路共振 → 流量振荡 + 压力联动
        d_f = compute_fault_delta(_CH['flow_rate'], severity, "increase")
        def _flow_osc(t, i, m, s):
            fs_val = int(s['fs'].execute(t))
            fe_val = int(s['fe'].execute(t))
            amp = s['amp'].execute(t)
            base = s['base'].execute(t)
            if fs_val <= i < fe_val:
                return base + amp * np.sin(2 * np.pi * i / 30)
            return base
        events['flow_rate'] = LambdaEvent(_flow_osc, sub_events={
            'base': refs['flow_t'], 'fs': fsc, 'fe': fec, 'amp': ConstantEvent(d_f)})
        # 补充: 流量振荡 → 压力联动振荡
        d_p = compute_fault_delta(_CH['pressure'], severity, "increase") * 0.3
        def _pres_osc(t, i, m, s):
            fs_val = int(s['fs'].execute(t))
            fe_val = int(s['fe'].execute(t))
            amp = s['amp'].execute(t)
            base = s['base'].execute(t)
            if fs_val <= i < fe_val:
                return base + amp * np.sin(2 * np.pi * i / 30 + np.pi / 4)
            return base
        events['pressure'] = LambdaEvent(_pres_osc, sub_events={
            'base': refs['pres'], 'fs': fsc, 'fe': fec, 'amp': ConstantEvent(d_p)})

    # ==== C-class: 仅 delta 校准 ====

    elif fault_type == "bearing_wear":
        # 轴承磨损: 振动渐增 + 温度渐升
        d_vp = compute_fault_delta(_CH['vibration_pump'], severity, "increase")
        events['vibration_pump'] = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['vp'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_vp)})
        d_t = compute_fault_delta(_CH['motor_temp'], severity, "increase")
        new_tt = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['tt'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_t)})
        events['motor_temp'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_tt, 'alpha': ConstantEvent(0.005), 'noise': ConstantEvent(0.05)})

    elif fault_type == "motor_overload":
        # 电机过载: 电流升高 + 温度升高
        d_c = compute_fault_delta(_CH['motor_current'], severity, "increase")
        events['motor_current'] = LambdaEvent(_fault_step, sub_events={
            'base': refs['cur'], 'fs': fsc, 'delta': ConstantEvent(d_c)})
        d_t = compute_fault_delta(_CH['motor_temp'], severity, "increase")
        new_tt = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['tt'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_t)})
        events['motor_temp'] = LambdaEvent(thermal_inertia, sub_events={
            'target': new_tt, 'alpha': ConstantEvent(0.005), 'noise': ConstantEvent(0.05)})

    elif fault_type == "coupling_misalignment":
        # 联轴器不对中: 两侧振动渐增
        d_vp = compute_fault_delta(_CH['vibration_pump'], severity, "increase")
        d_vm = compute_fault_delta(_CH['vibration_motor'], severity, "increase")
        events['vibration_pump'] = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['vp'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_vp)})
        events['vibration_motor'] = LambdaEvent(_fault_ramp, sub_events={
            'base': refs['vm'], 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_vm)})

    return events


# ============================================================
# 故障定义表
# ============================================================
PUMP_FAULTS = {
    "cavitation": (["vibration_pump", "vibration_motor", "pressure"],
                   ["入口压力低于饱和蒸汽压", "气泡产生并破裂", "振动间歇突发", "压力波动增大"],
                   "水泵气蚀故障"),
    "bearing_wear": (["vibration_pump", "motor_temp"],
                     ["轴承滚道点蚀", "振动缓慢增大", "摩擦热累积", "温度渐升"],
                     "水泵轴承磨损"),
    "seal_leak": (["pressure", "flow_rate"],
                  ["机械密封老化", "泄漏通道形成", "系统压力下降", "流量减少"],
                  "水泵密封泄漏"),
    "pipe_blockage": (["pressure", "motor_current"],
                      ["管道沉积物积累", "过流面积减小", "出口压力升高", "电流增加"],
                      "管道堵塞"),
    "impeller_damage": (["vibration_pump", "flow_rate", "pressure"],
                        ["叶轮叶片断裂/腐蚀", "质量不平衡", "振动突增", "效率下降"],
                        "叶轮损伤"),
    "motor_overload": (["motor_current", "motor_temp"],
                       ["液力负荷过大", "电流持续偏高", "I²R发热增加", "温度上升"],
                       "电机过载"),
    "dry_run": (["motor_current", "vibration_pump", "pressure"],
                ["液位过低", "泵体失去液封", "电流下降", "振动剧增", "压力趋零"],
                "水泵干转"),
    "coupling_misalignment": (["vibration_pump", "vibration_motor"],
                              ["联轴器安装不对中", "径向/轴向偏差", "两侧振动同步增大"],
                              "联轴器不对中"),
    "flow_oscillation": (["flow_rate", "pressure"],
                         ["管路系统共振", "流量产生周期性振荡", "压力联动波动"],
                         "管路流量振荡"),
    "voltage_sag": (["motor_voltage", "motor_current"],
                    ["供电电压骤降", "电机功率因数变化", "电流可能增大"],
                    "供电电压跌落"),
}


# ============================================================
# 统一生成接口
# ============================================================
def generate_pump(total_length=5000, seed=42, fault_type="normal",
                   severity="moderate",
                   randomize_fault_window: bool = False,
                   fault_start_range=(0.10, 0.80),
                   fault_duration_range=(0.08, 0.25),
                   severity_jitter: float = 0.0,
                   ) -> Tuple[pd.DataFrame, GenerationMetadata]:
    np.random.seed(seed)
    start = pd.Timestamp('2020-03-09 16:00:00')
    end = start + pd.Timedelta(seconds=total_length)

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
    elif fault_type in PUMP_FAULTS:
        model = _inject_pump_fault(fault_type, fs, fe, severity)
        fdef = PUMP_FAULTS[fault_type]
        sev_desc = {"mild": "轻微", "moderate": "中等", "severe": "严重"}[severity]
        events_meta.append(EventMetadata(
            event_type="fault", fault_name=fault_type,
            time_range=(fs, fe), severity=severity,
            parameters={"severity_factor": round(sev_factor, 3), "randomized": randomize_fault_window},
            affected_channels=fdef[0], causal_chain=fdef[1],
            description=f"{fdef[2]}（{sev_desc}级），第{fs}步至第{fe}步"))
    else:
        model, _ = _build_base()

    gen = Generator(start_date=start, end_date=end, freq='1S')
    df = gen.generate(model, n=1)

    el = [e.time_range[1] - e.time_range[0] for e in events_meta]
    sid = f"pump_{fault_type}_{severity}" if fault_type != "normal" else "pump_normal"

    metadata = GenerationMetadata(
        scenario_id=sid, scenario_name=f"水泵-{fault_type}-{severity}",
        domain=DOMAIN, domain_display=DOMAIN_DISPLAY,
        total_length=total_length, sampling_rate=SAMPLING_RATE,
        sampling_seconds=SAMPLING_SECONDS, channels=PUMP_CHANNELS,
        base_description="水循环泵系统正常运行。",
        injected_events=events_meta,
        overall_description=events_meta[0].description if events_meta else "水泵正常运行。",
        generation_seed=seed, difficulty=compute_difficulty(total_length, el),
        needle_ratio=round(sum(el)/total_length, 4) if el else 0.0,
    )
    return df, metadata


# ============================================================
# 场景注册
# ============================================================
def register_all_pump_tb():
    register_scenario(DOMAIN, "normal", "正常运行",
                      lambda total_length=5000, seed=42, **kwargs: generate_pump(total_length, seed, "normal", **kwargs))
    for ft in PUMP_FAULTS:
        for sev in ["mild", "moderate", "severe"]:
            sid = f"{ft}_{sev}"
            def gen(total_length=5000, seed=42, _ft=ft, _sev=sev, **kwargs):
                return generate_pump(total_length, seed, _ft, _sev, **kwargs)
            register_scenario(DOMAIN, sid, f"{ft}-{sev}", gen)

register_all_pump_tb()


# ============================================================
# 验证
# ============================================================
if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== Pump TB Verification ===")

    # 正常样本 vs 真实
    df_real = pd.read_csv('Dataset/SKAB/0.csv', sep=';')
    sensor_cols = [c for c in df_real.columns if c not in ['datetime', 'anomaly', 'changepoint']]

    df_s, meta = generate_pump(1125, 42, "normal")
    print(f"Normal: {df_s.shape}")

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format("Channel", "Real Mean", "Synth Mean", "Real Std", "Synth Std"))
    for rc, sc in zip(sensor_cols, df_s.columns):
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            sc, df_real[rc].mean(), df_s[sc].mean(), df_real[rc].std(), df_s[sc].std()))

    # Comparison plot
    fig, axes = plt.subplots(8, 1, figsize=(16, 20), sharex=False)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']
    for idx, (rc, sc) in enumerate(zip(sensor_cols, df_s.columns)):
        axes[idx].plot(df_real[rc].values, color='#1f77b4', linewidth=0.4, alpha=0.8, label='Real')
        axes[idx].plot(df_s[sc].values, color='#d62728', linewidth=0.4, alpha=0.8, label='TB Synth')
        axes[idx].set_ylabel(sc, fontsize=7)
        axes[idx].legend(fontsize=6)
        axes[idx].grid(True, alpha=0.2)
    axes[0].set_title('Pump: Real SKAB vs TB Synth', fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs('Data/Pump_TB', exist_ok=True)
    plt.savefig('Data/Pump_TB/comparison.png', dpi=150)
    plt.close()
    print("\nSaved: Data/Pump_TB/comparison.png")
    print(f"Registered: {len(get_all_scenarios(DOMAIN))} scenarios")
