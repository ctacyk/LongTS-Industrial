"""
磨煤机（Coal Mill）合成场景库

物理背景（基于真实 merged.csv 数据分析 + 领域知识）：
- 磨煤机是火力发电厂燃料制备系统的核心设备，将原煤研磨成煤粉
- 主要监测参数：电机电流(负载指标)、绕组温度(电机健康)、轴承温度(机械健康)、
  入口温度(热风干燥)、阀门开度(进风量控制)

真实数据统计特征（来自 Dataset/CoalMill/merged.csv, 85860行, 30s采样, ~30天）：
- 电机电流: mean=47.3A, std=13.3, range=[~0, 72]A, autocorr=0.998
- 绕组温度1: mean=50.4°C, std=7.1, range=[28, 73]°C, autocorr=0.9999
- 绕组温度2: mean=45.1°C, std=6.8, range=[24, 66]°C, autocorr=0.9999
- 非驱动侧轴承温度: mean=23.9°C, std=5.9, range=[11, 41]°C, autocorr=0.9999
- 驱动侧轴承温度: mean=31.8°C, std=6.2, range=[16, 53]°C, autocorr=0.9994
- 入口温度: mean=270.8°C, std=54.9, range=[7, 318]°C, autocorr=0.9997
- 阀门开度: mean=92.6%, std=22.8, range=[~0, 101]%, autocorr=0.999

关键相关性（必须在合成中保持）：
- 绕组温度1 ↔ 绕组温度2: 0.991（双传感器，几乎相同）
- 两个轴承温度: 0.966（同一轴系两端）
- 绕组温度 ↔ 轴承温度: 0.88-0.93（热传导）
- 入口温度 ↔ 阀门开度: 0.703（热风控制关系）
- 电流 ↔ 入口温度/阀门: 0.47-0.53（负载关联）
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

from MultiAgentTS.scenarios.base import (
    GenerationMetadata, EventMetadata, ChannelInfo,
    register_scenario, generate_base_signal,
    inject_gradual_change, inject_spike, inject_intermittent_fault,
    apply_thermal_lag, compute_difficulty
)

# ============================================================================
# 磨煤机通道定义
# ============================================================================

COAL_MILL_CHANNELS = [
    ChannelInfo("motor_current", "电机电流", "A",
                normal_range=(35.0, 60.0), alarm_threshold=70.0,
                physical_meaning="反映磨煤机研磨负荷，正常运行35-60A"),
    ChannelInfo("winding_temp_1", "电机绕组温度1", "°C",
                normal_range=(42.0, 62.0), alarm_threshold=85.0,
                physical_meaning="电机定子绕组温度，反映电机热状态"),
    ChannelInfo("winding_temp_2", "电机绕组温度2", "°C",
                normal_range=(38.0, 58.0), alarm_threshold=80.0,
                physical_meaning="同一绕组的冗余温度传感器"),
    ChannelInfo("bearing_temp_nondrive", "非驱动侧轴承温度", "°C",
                normal_range=(18.0, 35.0), alarm_threshold=55.0,
                physical_meaning="远离负载端的轴承温度，反映轴承润滑和磨损状态"),
    ChannelInfo("bearing_temp_drive", "驱动侧轴承温度", "°C",
                normal_range=(25.0, 42.0), alarm_threshold=60.0,
                physical_meaning="承载端轴承温度，通常略高于非驱动侧"),
    ChannelInfo("inlet_temp", "磨机入口温度", "°C",
                normal_range=(240.0, 310.0), alarm_threshold=330.0,
                physical_meaning="热风入口温度，用于干燥煤粉，受锅炉烟气温度控制"),
    ChannelInfo("inlet_valve_opening", "入口调节阀开度", "%",
                normal_range=(75.0, 100.0), alarm_threshold=None,
                physical_meaning="控制热风进入量，与入口温度强相关"),
]

DOMAIN = "coal_mill"
DOMAIN_DISPLAY = "磨煤机"
SAMPLING_RATE = "30s"
SAMPLING_SECONDS = 30


# ============================================================================
# 基础信号生成：生成正常运行状态的多通道时序
# ============================================================================

def _generate_normal_coal_mill(length: int, rng: np.random.Generator,
                                operation_mode: str = "steady",
                                load_level: float = 0.7) -> Dict[str, np.ndarray]:
    """
    生成正常运行的磨煤机多通道数据

    operation_mode: "steady" | "startup" | "load_varying"
    load_level: 0.0-1.0, 表示负荷水平
    """
    signals = {}

    # --- 负荷基准信号 ---
    if operation_mode == "steady":
        # 稳态运行：负荷围绕设定值小幅波动
        load = generate_base_signal(
            length, mean=load_level, noise_std=0.02,
            seasonal_period=int(2880 * (length / 5000)),  # 约24小时周期（2880个30s步）
            seasonal_amplitude=0.03,
            ar_coef=0.995, rng=rng
        )
    elif operation_mode == "startup":
        # 启动过程：从0逐渐升至运行负荷（前15%为升载阶段）
        ramp_len = int(length * 0.15)
        load = np.zeros(length)
        load[:ramp_len] = np.linspace(0, load_level, ramp_len)
        load[ramp_len:] = generate_base_signal(
            length - ramp_len, mean=load_level, noise_std=0.02,
            ar_coef=0.995, rng=rng
        )
    elif operation_mode == "load_varying":
        # 变负荷运行：负荷在一定范围内波动（模拟电网调度）
        load = generate_base_signal(
            length, mean=load_level, noise_std=0.05,
            seasonal_period=int(1440 * (length / 5000)),  # 12小时波动
            seasonal_amplitude=0.12,
            ar_coef=0.998, rng=rng
        )
    else:
        load = np.ones(length) * load_level

    load = np.clip(load, 0, 1)

    # --- 各通道信号 ---
    # 1. 电机电流：直接反映负荷，range 35-60A at load_level=0.7
    base_current = 15 + 60 * load  # 0负荷≈15A(空转), 满负荷≈75A
    signals["motor_current"] = base_current + rng.normal(0, 0.8, length)

    # 2. 绕组温度1：随电流平方正比（铜损=I²R），有热惯性
    thermal_base = 30 + 40 * (load ** 1.3)  # 基温30°C + 负荷相关热
    wt1_raw = generate_base_signal(length, mean=0, noise_std=0.15, ar_coef=0.9998, rng=rng)
    signals["winding_temp_1"] = thermal_base + wt1_raw

    # 3. 绕组温度2：与温度1高度相关，加小偏差（冗余传感器，安装位置略不同）
    wt2_offset = -5.0 + rng.normal(0, 0.3)  # 固定偏差约-5°C
    wt2_noise = rng.normal(0, 0.08, length)  # 独立噪声
    signals["winding_temp_2"] = signals["winding_temp_1"] + wt2_offset + wt2_noise

    # 4. 非驱动侧轴承温度：与环境温度和负荷有关，响应较慢
    ambient_sim = 20 + 3 * np.sin(2 * np.pi * np.arange(length) / max(length, 2880))
    bearing_nd_base = ambient_sim + 8 * load
    bt_nd_raw = generate_base_signal(length, mean=0, noise_std=0.1, ar_coef=0.9999, rng=rng)
    signals["bearing_temp_nondrive"] = bearing_nd_base + bt_nd_raw

    # 5. 驱动侧轴承温度：比非驱动侧高约7-10°C（承受更大径向载荷）
    drive_offset = 7 + 3 * load + rng.normal(0, 0.2)
    bt_d_noise = rng.normal(0, 0.12, length)
    signals["bearing_temp_drive"] = signals["bearing_temp_nondrive"] + drive_offset + bt_d_noise

    # 6. 入口温度：由锅炉侧控制，与磨煤机负荷有一定关联
    inlet_base = 250 + 40 * load  # 低负荷≈250°C, 高负荷≈290°C
    inlet_noise = generate_base_signal(length, mean=0, noise_std=2.5, ar_coef=0.9995, rng=rng)
    signals["inlet_temp"] = inlet_base + inlet_noise

    # 7. 阀门开度：与入口温度联动，PID控制关系
    valve_base = 80 + 15 * load  # 负荷越高开度越大
    valve_noise = generate_base_signal(length, mean=0, noise_std=1.0, ar_coef=0.998, rng=rng)
    signals["inlet_valve_opening"] = np.clip(valve_base + valve_noise, 0, 100)

    return signals


# ============================================================================
# 故障注入函数
# ============================================================================

def _inject_bearing_overheat(signals: Dict[str, np.ndarray],
                              start: int, end: int,
                              severity: str, drive_side: bool = True,
                              rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    轴承过热故障

    物理机理：润滑不足或轴承磨损 → 摩擦增大 → 轴承温度渐升
    → 热传导至绕组(滞后) → 电流可能略增(负载增大)

    severity:
    - mild: 温升5-8°C, 不触发报警
    - moderate: 温升10-18°C, 接近报警线
    - severe: 温升20-30°C, 超过报警线
    """
    delta_map = {"mild": (5, 8), "moderate": (10, 18), "severe": (20, 30)}
    delta_range = delta_map.get(severity, (5, 8))
    delta = rng.uniform(*delta_range)

    bearing_key = "bearing_temp_drive" if drive_side else "bearing_temp_nondrive"
    side_name = "驱动侧" if drive_side else "非驱动侧"

    # 轴承温度渐升（指数型，模拟摩擦热累积）
    signals[bearing_key] = inject_gradual_change(
        signals[bearing_key], start, end, delta, shape="exponential"
    )

    # 另一侧轴承也会受影响，但幅度较小（轴传导）
    other_key = "bearing_temp_nondrive" if drive_side else "bearing_temp_drive"
    signals[other_key] = inject_gradual_change(
        signals[other_key], start + (end - start) // 4, end,
        delta * 0.3, shape="linear"
    )

    # 绕组温度：通过热传导滞后响应
    lag = max(20, (end - start) // 5)
    signals["winding_temp_1"] = apply_thermal_lag(
        signals[bearing_key], signals["winding_temp_1"],
        coupling_coef=0.15, lag_steps=lag, rng=rng
    )
    signals["winding_temp_2"] = apply_thermal_lag(
        signals[bearing_key], signals["winding_temp_2"],
        coupling_coef=0.12, lag_steps=lag, rng=rng
    )

    # 电流：因摩擦增大，负载略增
    current_delta = delta * 0.15  # 温升每度约增加0.15A
    signals["motor_current"] = inject_gradual_change(
        signals["motor_current"], start + lag, end,
        current_delta, shape="sigmoid"
    )

    event = EventMetadata(
        event_type="fault",
        fault_name="bearing_overheat",
        time_range=(start, end),
        severity=severity,
        parameters={"delta_temp": round(delta, 2), "drive_side": drive_side,
                     "thermal_lag_steps": lag},
        affected_channels=[bearing_key, other_key, "winding_temp_1",
                           "winding_temp_2", "motor_current"],
        causal_chain=[
            f"轴承润滑油膜破坏或油质劣化",
            f"{side_name}轴承摩擦系数增大",
            f"轴承温度渐进式上升约{delta:.1f}°C（指数型升温曲线）",
            f"热量通过轴传导至另一侧轴承（幅度约{delta*0.3:.1f}°C）",
            f"经{lag}步热滞后，绕组温度开始跟随上升",
            f"机械摩擦增大导致电机负载增加，电流上升约{current_delta:.1f}A",
        ],
        description=(
            f"磨煤机{side_name}轴承出现{severity}级过热故障。"
            f"温度在第{start}步开始渐进上升，到第{end}步升高约{delta:.1f}°C。"
            f"故障通过热传导影响了绕组温度和电机电流。"
        )
    )
    return signals, event


def _inject_current_anomaly(signals: Dict[str, np.ndarray],
                             start: int, end: int,
                             severity: str, anomaly_type: str = "spike",
                             rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    电机电流异常

    物理机理：
    - spike: 突然性负载冲击（如大块煤/异物进入磨盘）
    - sustained: 持续性过载（如煤质变化导致研磨阻力增大）
    - oscillating: 振荡性电流（如给煤量不均匀）
    """
    amp_map = {
        "mild": {"spike": (8, 12), "sustained": (3, 5), "oscillating": (4, 6)},
        "moderate": {"spike": (15, 22), "sustained": (6, 10), "oscillating": (7, 10)},
        "severe": {"spike": (25, 35), "sustained": (12, 18), "oscillating": (12, 16)},
    }
    amp_range = amp_map.get(severity, amp_map["mild"]).get(anomaly_type, (8, 12))
    amplitude = rng.uniform(*amp_range)

    if anomaly_type == "spike":
        # 尖峰：瞬时冲击
        n_spikes = rng.integers(1, 4)
        for _ in range(n_spikes):
            pos = rng.integers(start, end)
            width = rng.integers(3, 15)
            signals["motor_current"] = inject_spike(
                signals["motor_current"], pos, amplitude, width, "gaussian", rng
            )
        # 电流冲击引起轻微温升
        signals["winding_temp_1"] = inject_gradual_change(
            signals["winding_temp_1"], start, end, amplitude * 0.08, "sigmoid"
        )
        signals["winding_temp_2"] = inject_gradual_change(
            signals["winding_temp_2"], start, end, amplitude * 0.06, "sigmoid"
        )

    elif anomaly_type == "sustained":
        # 持续过载
        signals["motor_current"] = inject_gradual_change(
            signals["motor_current"], start, end, amplitude, "sigmoid"
        )
        # 持续过载导致温升更明显
        temp_rise = amplitude * 0.25
        lag = max(30, (end - start) // 4)
        signals["winding_temp_1"] = inject_gradual_change(
            signals["winding_temp_1"], start + lag, end, temp_rise, "linear"
        )
        signals["winding_temp_2"] = inject_gradual_change(
            signals["winding_temp_2"], start + lag, end, temp_rise * 0.9, "linear"
        )

    elif anomaly_type == "oscillating":
        # 振荡
        duration = end - start
        t = np.arange(duration)
        osc_period = rng.integers(20, 60)
        osc = amplitude * np.sin(2 * np.pi * t / osc_period)
        osc *= np.linspace(0.3, 1.0, duration)  # 渐增包络
        signals["motor_current"][start:end] += osc

    type_desc_map = {
        "spike": "突发性冲击",
        "sustained": "持续性过载",
        "oscillating": "振荡性波动"
    }

    event = EventMetadata(
        event_type="fault",
        fault_name=f"current_{anomaly_type}",
        time_range=(start, end),
        severity=severity,
        parameters={"amplitude": round(amplitude, 2), "anomaly_type": anomaly_type},
        affected_channels=["motor_current", "winding_temp_1", "winding_temp_2"],
        causal_chain=[
            f"{'大块煤/异物进入磨盘' if anomaly_type == 'spike' else '煤质变化或给煤量异常'}",
            f"研磨阻力{'瞬间增大' if anomaly_type == 'spike' else '持续增大'}",
            f"电机电流出现{type_desc_map[anomaly_type]}，幅度约{amplitude:.1f}A",
            f"I²R损耗增加，绕组温度相应上升",
        ],
        description=(
            f"磨煤机电机电流出现{severity}级{type_desc_map[anomaly_type]}。"
            f"在第{start}-{end}步区间，电流幅值变化约{amplitude:.1f}A。"
        )
    )
    return signals, event


def _inject_inlet_blockage(signals: Dict[str, np.ndarray],
                            start: int, end: int,
                            severity: str,
                            rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    入口堵塞/结渣故障

    物理机理：煤粉在入口管道结渣或堵塞 → 风量减少 → 入口温度异常上升
    → 控制系统尝试增大阀门开度补偿 → 如补偿不足，磨内温度异常
    """
    delta_map = {"mild": (8, 15), "moderate": (18, 30), "severe": (35, 55)}
    delta_range = delta_map.get(severity, (8, 15))
    temp_delta = rng.uniform(*delta_range)

    # 入口温度升高（堵塞导致风量减少，热量集中）
    signals["inlet_temp"] = inject_gradual_change(
        signals["inlet_temp"], start, end, temp_delta, "exponential"
    )

    # 阀门开度增大（控制系统响应，尝试补偿）
    valve_response_delay = max(10, (end - start) // 8)
    valve_delta = min(temp_delta * 0.3, 15)  # 阀门补偿，但有上限
    signals["inlet_valve_opening"] = inject_gradual_change(
        signals["inlet_valve_opening"], start + valve_response_delay, end,
        valve_delta, "sigmoid"
    )
    signals["inlet_valve_opening"] = np.clip(signals["inlet_valve_opening"], 0, 100)

    # 电流可能略降（风量不足导致给煤量自动减少）
    if severity in ("moderate", "severe"):
        signals["motor_current"] = inject_gradual_change(
            signals["motor_current"], start + valve_response_delay * 2, end,
            -temp_delta * 0.1, "linear"
        )

    event = EventMetadata(
        event_type="fault",
        fault_name="inlet_blockage",
        time_range=(start, end),
        severity=severity,
        parameters={"temp_delta": round(temp_delta, 2), "valve_delta": round(valve_delta, 2)},
        affected_channels=["inlet_temp", "inlet_valve_opening", "motor_current"],
        causal_chain=[
            "煤粉在入口管道沉积或结渣",
            "通风面积减小，风量下降",
            f"入口区域温度异常上升约{temp_delta:.1f}°C",
            f"DCS控制系统检测到温度偏差，经{valve_response_delay}步后增大阀门开度约{valve_delta:.1f}%",
            "如补偿不足，可能触发保护性减负荷",
        ],
        description=(
            f"磨煤机入口出现{severity}级堵塞/结渣。"
            f"入口温度从第{start}步开始异常上升约{temp_delta:.1f}°C，"
            f"阀门开度自动增大约{valve_delta:.1f}%进行补偿。"
        )
    )
    return signals, event


def _inject_gradual_degradation(signals: Dict[str, np.ndarray],
                                 start: int, end: int,
                                 severity: str,
                                 rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    渐进式性能退化

    物理机理：磨辊/磨盘磨损 → 研磨效率降低 → 电流缓慢上升
    → 同时振动增大（在此场景中体现为轴承温度缓慢爬升）
    """
    rate_map = {"mild": (0.001, 0.003), "moderate": (0.004, 0.008), "severe": (0.01, 0.02)}
    rate_range = rate_map.get(severity, (0.001, 0.003))
    degrad_rate = rng.uniform(*rate_range)  # per step degradation

    duration = end - start
    degrad_profile = degrad_rate * np.arange(duration)

    # 电流缓慢上升
    signals["motor_current"][start:end] += degrad_profile * 3.0
    if end < len(signals["motor_current"]):
        signals["motor_current"][end:] += degrad_profile[-1] * 3.0

    # 轴承温度缓慢上升（磨损导致振动/摩擦增大）
    signals["bearing_temp_drive"][start:end] += degrad_profile * 1.5
    signals["bearing_temp_nondrive"][start:end] += degrad_profile * 0.8
    if end < len(signals["bearing_temp_drive"]):
        signals["bearing_temp_drive"][end:] += degrad_profile[-1] * 1.5
        signals["bearing_temp_nondrive"][end:] += degrad_profile[-1] * 0.8

    # 绕组温度跟随
    signals["winding_temp_1"][start:end] += degrad_profile * 0.6
    signals["winding_temp_2"][start:end] += degrad_profile * 0.5

    total_delta_current = round(degrad_profile[-1] * 3.0, 2)
    total_delta_bearing = round(degrad_profile[-1] * 1.5, 2)

    event = EventMetadata(
        event_type="fault",
        fault_name="gradual_degradation",
        time_range=(start, end),
        severity=severity,
        parameters={"degradation_rate": round(degrad_rate, 5),
                     "total_current_rise": total_delta_current,
                     "total_bearing_rise": total_delta_bearing},
        affected_channels=["motor_current", "bearing_temp_drive",
                           "bearing_temp_nondrive", "winding_temp_1", "winding_temp_2"],
        causal_chain=[
            "磨辊/磨盘表面磨损，研磨面粗糙度增加",
            f"研磨效率逐步降低，退化速率约{degrad_rate:.4f}/步",
            f"电机电流在{duration}步内缓慢上升约{total_delta_current}A",
            f"机械磨损导致轴承负载增加，驱动侧温升约{total_delta_bearing}°C",
            "整体表现为多参数协同缓慢偏移，不存在突变点",
        ],
        description=(
            f"磨煤机出现{severity}级渐进性能退化（磨辊磨损）。"
            f"从第{start}步开始，电流和温度多参数缓慢上升，"
            f"电流累计增加约{total_delta_current}A，"
            f"轴承温度增加约{total_delta_bearing}°C。无明显突变点，需从趋势判断。"
        )
    )
    return signals, event


def _inject_cooling_failure(signals: Dict[str, np.ndarray],
                             start: int, end: int,
                             severity: str,
                             rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    冷却系统故障

    物理机理：冷却风扇故障或冷却通道堵塞 → 散热能力下降
    → 绕组温度上升但轴承温度变化不大（区别于轴承故障）
    → 电流无明显变化（区别于负载增加）
    """
    delta_map = {"mild": (4, 7), "moderate": (8, 14), "severe": (15, 25)}
    delta_range = delta_map.get(severity, (4, 7))
    winding_delta = rng.uniform(*delta_range)

    # 绕组温度上升（散热不良）
    signals["winding_temp_1"] = inject_gradual_change(
        signals["winding_temp_1"], start, end, winding_delta, "sigmoid"
    )
    signals["winding_temp_2"] = inject_gradual_change(
        signals["winding_temp_2"], start, end, winding_delta * 0.95, "sigmoid"
    )

    # 轴承温度几乎不变（冷却系统主要影响绕组，不影响轴承散热路径）
    bearing_delta = winding_delta * 0.05  # 极小影响
    signals["bearing_temp_drive"][start:end] += rng.normal(0, bearing_delta * 0.3, end - start)

    event = EventMetadata(
        event_type="fault",
        fault_name="cooling_failure",
        time_range=(start, end),
        severity=severity,
        parameters={"winding_delta": round(winding_delta, 2)},
        affected_channels=["winding_temp_1", "winding_temp_2"],
        causal_chain=[
            "冷却风扇转速下降或冷却通道积灰堵塞",
            "电机散热能力显著降低",
            f"绕组温度在{end - start}步内上升约{winding_delta:.1f}°C",
            "轴承温度基本不变（不同散热路径）",
            "电流无明显变化（电气负载未改变）",
            "关键鉴别特征：仅绕组温度升高，其他参数正常",
        ],
        description=(
            f"磨煤机冷却系统出现{severity}级故障。"
            f"绕组温度从第{start}步开始上升约{winding_delta:.1f}°C，"
            f"但轴承温度和电流基本不变。这是区分冷却故障和轴承故障的关键特征。"
        )
    )
    return signals, event


# ============================================================================
# 场景生成函数（每个函数 = 一个独立场景）
# ============================================================================

def _make_scenario_generator(operation_mode: str, load_level: float,
                              fault_fn=None, fault_kwargs: dict = None,
                              fault_position_ratio: float = 0.6,
                              fault_duration_ratio: float = 0.15,
                              scenario_id: str = "",
                              scenario_name: str = "",
                              base_desc: str = "",
                              overall_desc: str = ""):
    """
    工厂函数：创建一个场景生成器

    这样做的好处是所有场景共享相同的生成逻辑，
    只通过参数差异化（故障类型、位置、严重度等）
    """
    def generate(total_length: int = 5000,
                 seed: int = 42) -> Tuple[pd.DataFrame, GenerationMetadata]:

        rng = np.random.default_rng(seed)

        # 1. 生成正常运行基线
        signals = _generate_normal_coal_mill(
            total_length, rng, operation_mode, load_level
        )

        events = []

        # 2. 注入故障（如有）
        if fault_fn is not None:
            fault_start = int(total_length * fault_position_ratio)
            fault_end = min(
                int(fault_start + total_length * fault_duration_ratio),
                total_length - 1
            )
            kwargs = dict(fault_kwargs) if fault_kwargs else {}
            signals, event = fault_fn(
                signals, fault_start, fault_end, rng=rng, **kwargs
            )
            events.append(event)

        # 3. 构建 DataFrame
        df = pd.DataFrame(signals)

        # 4. 计算难度
        event_lengths = [e.time_range[1] - e.time_range[0] for e in events]
        difficulty = compute_difficulty(total_length, event_lengths)
        needle_ratio = sum(event_lengths) / total_length if events else 0.0

        # 5. 构建元数据
        metadata = GenerationMetadata(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            domain=DOMAIN,
            domain_display=DOMAIN_DISPLAY,
            total_length=total_length,
            sampling_rate=SAMPLING_RATE,
            sampling_seconds=SAMPLING_SECONDS,
            channels=COAL_MILL_CHANNELS,
            base_description=base_desc,
            injected_events=events,
            overall_description=overall_desc.format(
                length=total_length, **{f"event_{i}": str(e.description) for i, e in enumerate(events)}
            ) if events else base_desc,
            generation_seed=seed,
            difficulty=difficulty,
            needle_ratio=round(needle_ratio, 4),
        )

        return df, metadata

    return generate


# ============================================================================
# 注册所有场景
# ============================================================================

def register_all_coal_mill_scenarios():
    """注册全部磨煤机场景（~28个）"""

    # ---- 正常运行场景 (5个) ----
    for i, (mode, load, name_suffix, desc) in enumerate([
        ("steady", 0.7, "normal_steady_70",
         "磨煤机在70%负荷下稳态运行，各参数平稳波动，无异常。"),
        ("steady", 0.9, "normal_steady_90",
         "磨煤机在90%高负荷下稳态运行，电流和温度偏高但在正常范围内。"),
        ("steady", 0.4, "normal_steady_40",
         "磨煤机在40%低负荷下运行，各参数处于较低水平。"),
        ("startup", 0.7, "normal_startup",
         "磨煤机冷启动过程，前15%时间步为升载阶段，电流从空转逐步升至运行值。"),
        ("load_varying", 0.65, "normal_load_varying",
         "磨煤机在电网调度要求下变负荷运行，各参数随负荷周期性波动。"),
    ]):
        gen = _make_scenario_generator(
            operation_mode=mode, load_level=load,
            scenario_id=name_suffix, scenario_name=f"磨煤机{desc[:6]}",
            base_desc=desc, overall_desc=desc,
        )
        register_scenario(DOMAIN, name_suffix, f"磨煤机正常-{name_suffix}", gen, desc)

    # ---- 轴承过热故障 (6个: 2侧 × 3严重度) ----
    for drive_side, side_id in [(True, "drive"), (False, "nondrive")]:
        for severity in ["mild", "moderate", "severe"]:
            sid = f"fault_bearing_{side_id}_{severity}"
            gen = _make_scenario_generator(
                operation_mode="steady", load_level=0.7,
                fault_fn=_inject_bearing_overheat,
                fault_kwargs={"severity": severity, "drive_side": drive_side},
                fault_position_ratio=0.55, fault_duration_ratio=0.2,
                scenario_id=sid,
                scenario_name=f"磨煤机轴承过热-{side_id}-{severity}",
                base_desc="磨煤机正常运行（70%负荷），突发轴承过热故障。",
            )
            register_scenario(DOMAIN, sid, f"轴承过热-{side_id}-{severity}", gen)

    # ---- 电流异常 (9个: 3类型 × 3严重度) ----
    for anomaly_type in ["spike", "sustained", "oscillating"]:
        for severity in ["mild", "moderate", "severe"]:
            sid = f"fault_current_{anomaly_type}_{severity}"
            dur_ratio = 0.05 if anomaly_type == "spike" else 0.15
            gen = _make_scenario_generator(
                operation_mode="steady", load_level=0.7,
                fault_fn=_inject_current_anomaly,
                fault_kwargs={"severity": severity, "anomaly_type": anomaly_type},
                fault_position_ratio=0.5, fault_duration_ratio=dur_ratio,
                scenario_id=sid,
                scenario_name=f"磨煤机电流{anomaly_type}-{severity}",
                base_desc="磨煤机正常运行，突发电流异常。",
            )
            register_scenario(DOMAIN, sid, f"电流{anomaly_type}-{severity}", gen)

    # ---- 入口堵塞 (3个: 3严重度) ----
    for severity in ["mild", "moderate", "severe"]:
        sid = f"fault_blockage_{severity}"
        gen = _make_scenario_generator(
            operation_mode="steady", load_level=0.7,
            fault_fn=_inject_inlet_blockage,
            fault_kwargs={"severity": severity},
            fault_position_ratio=0.6, fault_duration_ratio=0.18,
            scenario_id=sid,
            scenario_name=f"磨煤机入口堵塞-{severity}",
            base_desc="磨煤机正常运行，入口管道出现结渣堵塞。",
        )
        register_scenario(DOMAIN, sid, f"入口堵塞-{severity}", gen)

    # ---- 渐进退化 (3个: 3严重度) ----
    for severity in ["mild", "moderate", "severe"]:
        sid = f"fault_degradation_{severity}"
        gen = _make_scenario_generator(
            operation_mode="steady", load_level=0.7,
            fault_fn=_inject_gradual_degradation,
            fault_kwargs={"severity": severity},
            fault_position_ratio=0.2, fault_duration_ratio=0.6,
            scenario_id=sid,
            scenario_name=f"磨煤机渐进退化-{severity}",
            base_desc="磨煤机运行中磨辊磨损导致性能渐进退化。",
        )
        register_scenario(DOMAIN, sid, f"渐进退化-{severity}", gen)

    # ---- 冷却故障 (3个: 3严重度) ----
    for severity in ["mild", "moderate", "severe"]:
        sid = f"fault_cooling_{severity}"
        gen = _make_scenario_generator(
            operation_mode="steady", load_level=0.7,
            fault_fn=_inject_cooling_failure,
            fault_kwargs={"severity": severity},
            fault_position_ratio=0.5, fault_duration_ratio=0.2,
            scenario_id=sid,
            scenario_name=f"磨煤机冷却故障-{severity}",
            base_desc="磨煤机运行中冷却系统出现故障。",
        )
        register_scenario(DOMAIN, sid, f"冷却故障-{severity}", gen)


# 模块导入时自动注册
register_all_coal_mill_scenarios()
