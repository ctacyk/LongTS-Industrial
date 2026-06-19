"""
电力变压器（Transformer）合成场景库

物理背景（基于 ETTh2 真实数据 + 电力变压器领域知识）：
- 电力变压器是输配电系统核心设备，将高压变换为低压供电
- 主要监测参数：高/中/低压侧有功无功功率(负载)、油温(绝缘和冷却状态)
- ETT = Electricity Transformer Temperature dataset

真实数据统计特征（来自 Dataset/ETT/ETTh2.csv, 17420行, 1h采样, ~2年）：
- HUFL (High UseFul Load): mean=37.2, std=10.2, range=[0, 107.9]
- HULL (High UseLess Load): mean=8.5, std=6.0, range=[-18.7, 36.4]
- MUFL (Middle UseFul Load): mean=43.8, std=13.1, range=[11.2, 93.2]
- MULL (Middle UseLess Load): mean=8.3, std=4.4, range=[-6.6, 28.7]
- LUFL (Low UseFul Load): mean=-3.4, std=6.1, range=[-14.4, 17.2]
- LULL (Low UseLess Load): mean=-2.1, std=6.0, range=[-31.5, 2.9]
- OT (Oil Temperature): mean=26.6°C, std=11.9, range=[-2.6, 58.9]

领域知识：
- 变压器油温是最关键的健康指标，直接反映绝缘状态
- 负载与油温强相关（I²R损耗 → 热量 → 油温升高）
- 存在明显的日周期（白天用电高峰）和季节性（夏季高温+高负荷）
- 油温过高会加速绝缘老化，缩短变压器寿命
- 常见故障：过载、油温异常、绝缘老化、冷却系统故障、负载突变
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from MultiAgentTS.scenarios.base import (
    GenerationMetadata, EventMetadata, ChannelInfo,
    register_scenario, generate_base_signal,
    inject_gradual_change, inject_spike, inject_intermittent_fault,
    apply_thermal_lag, compute_difficulty
)

# ============================================================================
# 变压器通道定义
# ============================================================================

TRANSFORMER_CHANNELS = [
    ChannelInfo("HUFL", "高压侧有功负载", "MW",
                normal_range=(20.0, 55.0), alarm_threshold=90.0,
                physical_meaning="高压侧有功功率，反映主要电力传输负荷"),
    ChannelInfo("HULL", "高压侧无功负载", "MVar",
                normal_range=(0.0, 20.0), alarm_threshold=35.0,
                physical_meaning="高压侧无功功率，反映电网功率因数"),
    ChannelInfo("MUFL", "中压侧有功负载", "MW",
                normal_range=(25.0, 65.0), alarm_threshold=85.0,
                physical_meaning="中压侧有功功率，通常为主要供电回路"),
    ChannelInfo("MULL", "中压侧无功负载", "MVar",
                normal_range=(2.0, 15.0), alarm_threshold=25.0,
                physical_meaning="中压侧无功功率"),
    ChannelInfo("LUFL", "低压侧有功负载", "MW",
                normal_range=(-10.0, 10.0), alarm_threshold=15.0,
                physical_meaning="低压侧有功功率，可能包含分布式电源反送"),
    ChannelInfo("LULL", "低压侧无功负载", "MVar",
                normal_range=(-15.0, 2.0), alarm_threshold=None,
                physical_meaning="低压侧无功功率"),
    ChannelInfo("OT", "油温", "°C",
                normal_range=(15.0, 45.0), alarm_threshold=65.0,
                physical_meaning="变压器油顶层温度，反映整体热状态和绝缘健康"),
]

DOMAIN = "transformer"
DOMAIN_DISPLAY = "电力变压器"
SAMPLING_RATE = "1h"
SAMPLING_SECONDS = 3600


# ============================================================================
# 基础信号生成
# ============================================================================

def _generate_normal_transformer(length: int, rng: np.random.Generator,
                                  season: str = "mixed",
                                  load_level: float = 0.6) -> Dict[str, np.ndarray]:
    """
    生成正常运行的变压器多通道数据

    电力负荷具有强烈的日周期（白天高、夜间低）和季节性（夏冬高、春秋低）

    season: "summer" | "winter" | "spring" | "mixed"
    load_level: 0.0-1.0 基础负载水平
    """
    signals = {}

    # --- 日周期模型：24小时内负荷先升后降 ---
    hourly_pattern = np.zeros(length)
    for i in range(length):
        hour = i % 24
        # 典型日负荷曲线：凌晨低谷(4-6h)，上午高峰(9-12h)，午后(14-17h)，晚高峰(18-21h)
        if hour < 6:
            hourly_pattern[i] = 0.5 + 0.1 * np.sin(np.pi * hour / 6)  # 低谷
        elif hour < 12:
            hourly_pattern[i] = 0.7 + 0.3 * np.sin(np.pi * (hour - 6) / 6)  # 上午爬坡
        elif hour < 14:
            hourly_pattern[i] = 0.85  # 午后平台
        elif hour < 21:
            hourly_pattern[i] = 0.75 + 0.25 * np.sin(np.pi * (hour - 14) / 7)  # 晚高峰
        else:
            hourly_pattern[i] = 0.6 - 0.1 * (hour - 21) / 3  # 夜间下降

    # --- 季节性调制 ---
    seasonal_factor = np.ones(length)
    if season == "summer":
        base_temp_offset = 15.0
        seasonal_factor *= 1.15  # 夏季高负荷（空调）
    elif season == "winter":
        base_temp_offset = -5.0
        seasonal_factor *= 1.1  # 冬季偏高（采暖）
    elif season == "spring":
        base_temp_offset = 5.0
        seasonal_factor *= 0.85  # 春季低负荷
    else:  # mixed - 模拟跨季节
        cycle_len = max(length, 1)
        seasonal_factor = 0.9 + 0.2 * np.sin(2 * np.pi * np.arange(length) / cycle_len)
        base_temp_offset = 10 * np.sin(2 * np.pi * np.arange(length) / cycle_len)

    load_profile = hourly_pattern * seasonal_factor * load_level

    # --- 各通道 ---
    # HUFL：高压侧有功，跟随负荷曲线
    hufl_noise = generate_base_signal(length, 0, 3.0, ar_coef=0.95, rng=rng)
    signals["HUFL"] = 20 + 50 * load_profile + hufl_noise

    # HULL：高压侧无功，与有功相关但幅度小
    hull_noise = generate_base_signal(length, 0, 2.0, ar_coef=0.92, rng=rng)
    signals["HULL"] = 3 + 18 * load_profile * 0.5 + hull_noise

    # MUFL：中压侧有功，通常是主供电回路，负荷更高
    mufl_noise = generate_base_signal(length, 0, 4.0, ar_coef=0.95, rng=rng)
    signals["MUFL"] = 25 + 55 * load_profile + mufl_noise

    # MULL：中压侧无功
    mull_noise = generate_base_signal(length, 0, 1.5, ar_coef=0.90, rng=rng)
    signals["MULL"] = 3 + 12 * load_profile * 0.4 + mull_noise

    # LUFL：低压侧有功（可能有反送电）
    lufl_noise = generate_base_signal(length, 0, 2.5, ar_coef=0.88, rng=rng)
    signals["LUFL"] = -5 + 15 * load_profile * 0.6 + lufl_noise

    # LULL：低压侧无功
    lull_noise = generate_base_signal(length, 0, 2.0, ar_coef=0.85, rng=rng)
    signals["LULL"] = -8 + 5 * load_profile * 0.3 + lull_noise

    # OT：油温 = f(负荷², 环境温度, 冷却)
    # 油温响应有较大热惯性（时间常数约2-4小时）
    # T_oil ≈ T_ambient + k * Load² (热平衡方程的简化)
    if isinstance(base_temp_offset, np.ndarray):
        ambient = 20 + base_temp_offset
    else:
        ambient = 20 + base_temp_offset + 3 * np.sin(2 * np.pi * np.arange(length) / 24)  # 日温差

    # I²R 损耗正比于负荷平方
    thermal_load = load_profile ** 1.5 * 35  # 满载时温升约35°C
    ot_noise = generate_base_signal(length, 0, 0.8, ar_coef=0.998, rng=rng)

    # 热惯性滤波（一阶低通，时间常数约3小时）
    raw_ot = ambient + thermal_load + ot_noise
    signals["OT"] = np.zeros(length)
    signals["OT"][0] = raw_ot[0]
    tau = 3.0  # 热时间常数（小时）
    alpha = 1 / (1 + tau)
    for i in range(1, length):
        signals["OT"][i] = signals["OT"][i-1] * (1 - alpha) + raw_ot[i] * alpha

    return signals


# ============================================================================
# 故障注入函数
# ============================================================================

def _inject_overload(signals: Dict[str, np.ndarray],
                      start: int, end: int,
                      severity: str,
                      rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    过载故障

    物理机理：电网调度要求超出额定容量运行
    → 所有负荷通道增大 → 油温快速上升（I²R效应）
    → 持续过载导致绝缘加速老化
    """
    overload_map = {"mild": (1.05, 1.15), "moderate": (1.15, 1.3), "severe": (1.3, 1.5)}
    overload_range = overload_map.get(severity, (1.05, 1.15))
    overload_factor = rng.uniform(*overload_range)

    duration = end - start

    # 所有负荷通道按比例增大
    for ch in ["HUFL", "MUFL"]:
        profile = np.linspace(1.0, overload_factor, duration)
        signals[ch][start:end] *= profile
        if end < len(signals[ch]):
            signals[ch][end:] *= overload_factor

    # 无功也相应增大
    for ch in ["HULL", "MULL"]:
        signals[ch][start:end] *= np.linspace(1.0, overload_factor * 0.7, duration)

    # 油温上升：过载导致额外温升 ΔT ∝ (overload_factor² - 1)
    temp_rise = (overload_factor ** 2 - 1) * 25  # 满载温升约25°C的过载部分
    signals["OT"] = inject_gradual_change(
        signals["OT"], start, end, temp_rise, "exponential"
    )

    event = EventMetadata(
        event_type="fault",
        fault_name="overload",
        time_range=(start, end),
        severity=severity,
        parameters={"overload_factor": round(overload_factor, 3),
                     "temp_rise": round(temp_rise, 2)},
        affected_channels=["HUFL", "HULL", "MUFL", "MULL", "OT"],
        causal_chain=[
            "电网调度要求超出额定容量运行",
            f"负荷水平升至额定的{overload_factor:.0%}",
            "有功和无功功率同步增大",
            f"I²R铜损增加，油温上升约{temp_rise:.1f}°C",
            "持续过载将加速变压器绝缘老化",
        ],
        description=(
            f"变压器出现{severity}级过载运行。"
            f"负荷在第{start}步升至额定的{overload_factor:.0%}，"
            f"油温相应上升约{temp_rise:.1f}°C。"
        )
    )
    return signals, event


def _inject_oil_temp_anomaly(signals: Dict[str, np.ndarray],
                              start: int, end: int,
                              severity: str,
                              rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    油温异常（冷却系统故障）

    物理机理：冷却风扇/油泵故障 → 散热能力下降 → 油温上升
    但负荷不变（区别于过载导致的油温升高）
    """
    delta_map = {"mild": (3, 6), "moderate": (7, 12), "severe": (13, 22)}
    delta_range = delta_map.get(severity, (3, 6))
    temp_delta = rng.uniform(*delta_range)

    # 油温升高但负荷不变
    signals["OT"] = inject_gradual_change(
        signals["OT"], start, end, temp_delta, "sigmoid"
    )

    event = EventMetadata(
        event_type="fault",
        fault_name="oil_temp_anomaly",
        time_range=(start, end),
        severity=severity,
        parameters={"temp_delta": round(temp_delta, 2)},
        affected_channels=["OT"],
        causal_chain=[
            "冷却风扇转速下降或冷却油泵流量不足",
            "变压器散热能力降低",
            f"油温在负荷不变的情况下异常上升约{temp_delta:.1f}°C",
            "关键鉴别特征：油温升高但各侧负荷未增加",
            "若不处理，绝缘老化将加速",
        ],
        description=(
            f"变压器冷却系统出现{severity}级故障。"
            f"油温从第{start}步开始异常上升约{temp_delta:.1f}°C，"
            f"但各侧负荷保持正常水平。"
        )
    )
    return signals, event


def _inject_insulation_aging(signals: Dict[str, np.ndarray],
                              start: int, end: int,
                              severity: str,
                              rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    绝缘老化

    物理机理：长期运行导致绝缘材料劣化 → 损耗增大 → 油温基线缓慢爬升
    同时负荷-温度关系的斜率增大（同等负荷下温度更高）
    """
    rate_map = {"mild": (0.002, 0.005), "moderate": (0.006, 0.012), "severe": (0.015, 0.03)}
    rate_range = rate_map.get(severity, (0.002, 0.005))
    aging_rate = rng.uniform(*rate_range)

    duration = end - start
    aging_profile = aging_rate * np.arange(duration)

    # 油温基线缓慢爬升
    signals["OT"][start:end] += aging_profile
    if end < len(signals["OT"]):
        signals["OT"][end:] += aging_profile[-1]

    # 无功损耗略增（绝缘劣化导致介质损耗增大）
    signals["HULL"][start:end] += aging_profile * 0.3
    signals["MULL"][start:end] += aging_profile * 0.2

    total_rise = round(aging_profile[-1], 2)

    event = EventMetadata(
        event_type="fault",
        fault_name="insulation_aging",
        time_range=(start, end),
        severity=severity,
        parameters={"aging_rate": round(aging_rate, 5), "total_rise": total_rise},
        affected_channels=["OT", "HULL", "MULL"],
        causal_chain=[
            "变压器绝缘材料在长期热应力和电应力下老化",
            "绝缘介质损耗角正切值增大",
            f"油温基线在{duration}步内缓慢上升约{total_rise}°C",
            "无功损耗同步小幅增加",
            "无明显突变点，需从长期趋势识别",
        ],
        description=(
            f"变压器出现{severity}级绝缘老化趋势。"
            f"从第{start}步开始，油温以{aging_rate:.4f}°C/步的速率缓慢上升，"
            f"累计上升约{total_rise}°C。变化极其缓慢，难以从短期观察发现。"
        )
    )
    return signals, event


def _inject_load_surge(signals: Dict[str, np.ndarray],
                        start: int, end: int,
                        severity: str,
                        rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    负载突变

    物理机理：大型用电设备启停 → 负荷阶跃变化
    → 油温随之变化但有滞后
    """
    surge_map = {"mild": (10, 18), "moderate": (20, 35), "severe": (40, 55)}
    surge_range = surge_map.get(severity, (10, 18))
    surge_amplitude = rng.uniform(*surge_range)

    # 有功负荷阶跃
    for ch in ["HUFL", "MUFL"]:
        signals[ch] = inject_gradual_change(
            signals[ch], start, end, surge_amplitude, "step"
        )

    # 油温滞后响应
    temp_response = surge_amplitude * 0.4
    lag = max(3, (end - start) // 3)
    signals["OT"] = inject_gradual_change(
        signals["OT"], start + lag, end, temp_response, "exponential"
    )

    event = EventMetadata(
        event_type="fault",
        fault_name="load_surge",
        time_range=(start, end),
        severity=severity,
        parameters={"surge_amplitude": round(surge_amplitude, 2),
                     "temp_response": round(temp_response, 2)},
        affected_channels=["HUFL", "MUFL", "OT"],
        causal_chain=[
            "下游大型用电设备突然启动或负荷转移",
            f"高压侧和中压侧有功负荷突增约{surge_amplitude:.1f}MW",
            f"油温经约{lag}步热滞后上升约{temp_response:.1f}°C",
        ],
        description=(
            f"变压器出现{severity}级负载突变。"
            f"第{start}步有功负荷突增约{surge_amplitude:.1f}MW，"
            f"油温滞后约{lag}步后上升约{temp_response:.1f}°C。"
        )
    )
    return signals, event


def _inject_partial_discharge(signals: Dict[str, np.ndarray],
                               start: int, end: int,
                               severity: str,
                               rng: np.random.Generator = None) -> Tuple[Dict, EventMetadata]:
    """
    局部放电

    物理机理：绝缘缺陷处产生局部放电 → 无功损耗间歇性增大
    → 局部热点导致油温出现小幅高频波动
    """
    amp_map = {"mild": (1.5, 3), "moderate": (3.5, 6), "severe": (7, 12)}
    amp_range = amp_map.get(severity, (1.5, 3))
    pd_amplitude = rng.uniform(*amp_range)

    # 无功损耗间歇性增大
    signals["HULL"] = inject_intermittent_fault(
        signals["HULL"], start, end, pd_amplitude,
        on_duration=rng.integers(5, 15), off_duration=rng.integers(20, 50), rng=rng
    )

    # 油温出现小幅高频波动（局部热点效应）
    osc_duration = end - start
    t = np.arange(osc_duration)
    oil_osc = pd_amplitude * 0.15 * np.sin(2 * np.pi * t / rng.integers(8, 20))
    oil_osc *= rng.choice([0, 1], size=osc_duration, p=[0.3, 0.7])
    signals["OT"][start:end] += oil_osc

    event = EventMetadata(
        event_type="fault",
        fault_name="partial_discharge",
        time_range=(start, end),
        severity=severity,
        parameters={"pd_amplitude": round(pd_amplitude, 2)},
        affected_channels=["HULL", "OT"],
        causal_chain=[
            "变压器绝缘层存在气泡、杂质或裂纹等缺陷",
            "在电场作用下，缺陷处发生局部放电",
            f"无功损耗出现间歇性脉冲，幅值约{pd_amplitude:.1f}MVar",
            "局部热点导致油温出现小幅高频波动",
            "放电具有间歇性特征，不是持续性异常",
        ],
        description=(
            f"变压器出现{severity}级局部放电。"
            f"第{start}-{end}步区间无功损耗出现间歇性脉冲（幅值约{pd_amplitude:.1f}MVar），"
            f"油温伴有小幅波动。特征是间歇性而非持续性。"
        )
    )
    return signals, event


# ============================================================================
# 场景注册工厂
# ============================================================================

def _make_transformer_generator(season: str, load_level: float,
                                 fault_fn=None, fault_kwargs: dict = None,
                                 fault_position_ratio: float = 0.5,
                                 fault_duration_ratio: float = 0.15,
                                 scenario_id: str = "",
                                 scenario_name: str = "",
                                 base_desc: str = ""):

    def generate(total_length: int = 5000, seed: int = 42):
        rng = np.random.default_rng(seed)
        signals = _generate_normal_transformer(total_length, rng, season, load_level)
        events = []

        if fault_fn is not None:
            fs = int(total_length * fault_position_ratio)
            fe = min(int(fs + total_length * fault_duration_ratio), total_length - 1)
            kwargs = dict(fault_kwargs) if fault_kwargs else {}
            signals, event = fault_fn(signals, fs, fe, rng=rng, **kwargs)
            events.append(event)

        df = pd.DataFrame(signals)
        el = [e.time_range[1] - e.time_range[0] for e in events]
        difficulty = compute_difficulty(total_length, el)
        nr = sum(el) / total_length if events else 0.0

        metadata = GenerationMetadata(
            scenario_id=scenario_id, scenario_name=scenario_name,
            domain="transformer", domain_display=DOMAIN_DISPLAY,
            total_length=total_length, sampling_rate=SAMPLING_RATE,
            sampling_seconds=SAMPLING_SECONDS, channels=TRANSFORMER_CHANNELS,
            base_description=base_desc, injected_events=events,
            overall_description=events[0].description if events else base_desc,
            generation_seed=seed, difficulty=difficulty, needle_ratio=round(nr, 4),
        )
        return df, metadata

    return generate


def register_all_transformer_scenarios():
    """注册全部变压器场景（~27个）"""

    # ---- 正常运行 (4个) ----
    for season, load, sid, desc in [
        ("summer", 0.7, "normal_summer_70",
         "变压器夏季70%负荷稳态运行，油温偏高但在正常范围。日负荷曲线明显。"),
        ("winter", 0.65, "normal_winter_65",
         "变压器冬季65%负荷运行，环境温度低，油温整体偏低。"),
        ("spring", 0.5, "normal_spring_50",
         "变压器春季50%轻载运行，各参数处于较低水平。"),
        ("mixed", 0.6, "normal_mixed_60",
         "变压器跨季节运行，负荷和油温呈现季节性变化趋势。"),
    ]:
        gen = _make_transformer_generator(
            season=season, load_level=load,
            scenario_id=sid, scenario_name=f"变压器正常-{sid}",
            base_desc=desc
        )
        register_scenario("transformer", sid, f"变压器正常-{sid}", gen, desc)

    # ---- 过载 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_overload_{sev}"
        gen = _make_transformer_generator(
            season="summer", load_level=0.7,
            fault_fn=_inject_overload, fault_kwargs={"severity": sev},
            fault_position_ratio=0.5, fault_duration_ratio=0.2,
            scenario_id=sid, scenario_name=f"变压器过载-{sev}",
            base_desc="变压器夏季运行中出现过载。"
        )
        register_scenario("transformer", sid, f"过载-{sev}", gen)

    # ---- 油温异常 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_oil_temp_{sev}"
        gen = _make_transformer_generator(
            season="summer", load_level=0.6,
            fault_fn=_inject_oil_temp_anomaly, fault_kwargs={"severity": sev},
            fault_position_ratio=0.5, fault_duration_ratio=0.2,
            scenario_id=sid, scenario_name=f"变压器油温异常-{sev}",
            base_desc="变压器冷却系统故障导致油温异常。"
        )
        register_scenario("transformer", sid, f"油温异常-{sev}", gen)

    # ---- 绝缘老化 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_insulation_{sev}"
        gen = _make_transformer_generator(
            season="mixed", load_level=0.6,
            fault_fn=_inject_insulation_aging, fault_kwargs={"severity": sev},
            fault_position_ratio=0.15, fault_duration_ratio=0.65,
            scenario_id=sid, scenario_name=f"变压器绝缘老化-{sev}",
            base_desc="变压器绝缘材料长期老化。"
        )
        register_scenario("transformer", sid, f"绝缘老化-{sev}", gen)

    # ---- 负载突变 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_load_surge_{sev}"
        gen = _make_transformer_generator(
            season="summer", load_level=0.5,
            fault_fn=_inject_load_surge, fault_kwargs={"severity": sev},
            fault_position_ratio=0.6, fault_duration_ratio=0.1,
            scenario_id=sid, scenario_name=f"变压器负载突变-{sev}",
            base_desc="变压器出现负载突然变化。"
        )
        register_scenario("transformer", sid, f"负载突变-{sev}", gen)

    # ---- 局部放电 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_partial_discharge_{sev}"
        gen = _make_transformer_generator(
            season="summer", load_level=0.65,
            fault_fn=_inject_partial_discharge, fault_kwargs={"severity": sev},
            fault_position_ratio=0.4, fault_duration_ratio=0.25,
            scenario_id=sid, scenario_name=f"变压器局部放电-{sev}",
            base_desc="变压器绝缘缺陷导致局部放电。"
        )
        register_scenario("transformer", sid, f"局部放电-{sev}", gen)

    # ---- 不同季节下的过载（增加多样性）(4个) ----
    for season, load, sev in [
        ("winter", 0.7, "moderate"),
        ("spring", 0.5, "mild"),
        ("mixed", 0.6, "severe"),
        ("summer", 0.8, "severe"),
    ]:
        sid = f"fault_overload_{season}_{sev}"
        gen = _make_transformer_generator(
            season=season, load_level=load,
            fault_fn=_inject_overload, fault_kwargs={"severity": sev},
            fault_position_ratio=0.45, fault_duration_ratio=0.2,
            scenario_id=sid, scenario_name=f"变压器过载-{season}-{sev}",
            base_desc=f"变压器{season}运行中出现{sev}级过载。"
        )
        register_scenario("transformer", sid, f"过载-{season}-{sev}", gen)


register_all_transformer_scenarios()
