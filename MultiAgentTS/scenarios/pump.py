"""
水循环泵系统（Water Pump）合成场景库

物理背景（基于 SKAB 数据集结构 + 泵系统工程知识）：
- SKAB (Skoltech Anomaly Benchmark) 来自一个水循环测试台
- 包含电机、水泵、管道组成的闭式循环系统
- 传感器：振动加速度RMS×2、电流、压力、温度×2、电压、流量

SKAB 数据列及典型范围（基于领域知识和文献）：
- Accelerometer1RMS: 泵侧振动, 正常 0.02-0.08g, 异常可达 0.3g+
- Accelerometer2RMS: 电机侧振动, 正常 0.01-0.05g
- Current: 电机电流, 正常 2.0-4.5A (小型泵)
- Pressure: 泵出口压力, 正常 1.5-3.5 Bar
- Temperature: 电机体温度, 正常 30-55°C
- Thermocouple: 循环液温度, 正常 20-40°C
- Voltage: 电机电压, 正常 220-240V (稳定)
- RateRMS: 流量, 正常 8-15 L/min
- anomaly: 异常标签 (0/1)

水泵系统常见故障模式：
1. 气蚀(Cavitation): 泵入口压力过低产生气泡 → 振动剧增+压力波动+流量下降
2. 轴承磨损: 振动缓慢增大 → 温度渐升 → 电流可能略增
3. 密封泄漏: 压力缓慢下降 → 流量下降 → 泵可能空转（电流降低）
4. 电机过载: 电流持续偏高 → 温度上升 → 电压可能波动
5. 管道部分堵塞: 压力升高 → 流量下降 → 电流增加（泵更努力工作）
6. 叶轮损伤: 振动突增 → 流量下降 → 效率降低（电流增但流量不增）
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

PUMP_CHANNELS = [
    ChannelInfo("vibration_pump", "泵侧振动加速度RMS", "g",
                normal_range=(0.02, 0.08), alarm_threshold=0.20,
                physical_meaning="泵体振动水平，反映机械平衡和气蚀状态"),
    ChannelInfo("vibration_motor", "电机侧振动加速度RMS", "g",
                normal_range=(0.01, 0.05), alarm_threshold=0.15,
                physical_meaning="电机振动水平，反映轴对中和轴承状态"),
    ChannelInfo("motor_current", "电机电流", "A",
                normal_range=(2.5, 4.0), alarm_threshold=5.5,
                physical_meaning="电机工作电流，反映泵的液力负荷"),
    ChannelInfo("pressure", "泵出口压力", "Bar",
                normal_range=(2.0, 3.2), alarm_threshold=4.0,
                physical_meaning="泵出口管道压力，反映泵的扬程输出"),
    ChannelInfo("motor_temp", "电机体温度", "°C",
                normal_range=(32.0, 50.0), alarm_threshold=70.0,
                physical_meaning="电机壳体温度，反映电机散热状态"),
    ChannelInfo("fluid_temp", "循环液温度", "°C",
                normal_range=(22.0, 36.0), alarm_threshold=50.0,
                physical_meaning="循环管路中的工质温度"),
    ChannelInfo("motor_voltage", "电机电压", "V",
                normal_range=(218.0, 242.0), alarm_threshold=None,
                physical_meaning="电机供电电压，正常情况下相对稳定"),
    ChannelInfo("flow_rate", "循环流量", "L/min",
                normal_range=(9.0, 14.0), alarm_threshold=None,
                physical_meaning="系统循环流量，反映泵的水力输出效率"),
]

DOMAIN = "pump"
DOMAIN_DISPLAY = "水循环泵系统"
SAMPLING_RATE = "1s"
SAMPLING_SECONDS = 1


def _generate_normal_pump(length: int, rng: np.random.Generator,
                           operation_mode: str = "steady",
                           load_level: float = 0.7) -> Dict[str, np.ndarray]:
    """
    生成正常运行的水泵系统数据

    泵系统特点：
    - 振动信号有较高的随机性（与磨煤机的光滑热信号不同）
    - 压力和流量有直接物理关联（泵特性曲线）
    - 电流反映液力负荷
    - 温度变化缓慢（热惯性大）
    """
    signals = {}

    # --- 负荷基准 ---
    if operation_mode == "steady":
        load = generate_base_signal(
            length, mean=load_level, noise_std=0.015,
            ar_coef=0.99, rng=rng
        )
    elif operation_mode == "varying":
        # 流量随工艺需求波动（如冷却负荷变化）
        load = generate_base_signal(
            length, mean=load_level, noise_std=0.04,
            seasonal_period=max(600, length // 8),
            seasonal_amplitude=0.1,
            ar_coef=0.995, rng=rng
        )
    elif operation_mode == "startup":
        ramp = int(length * 0.1)
        load = np.zeros(length)
        load[:ramp] = np.linspace(0, load_level, ramp)
        load[ramp:] = generate_base_signal(
            length - ramp, mean=load_level, noise_std=0.015, ar_coef=0.99, rng=rng
        )
    else:
        load = np.ones(length) * load_level

    load = np.clip(load, 0, 1)

    # --- 各通道 ---
    # 泵侧振动：基线噪声 + 负荷相关成分
    vib_base = 0.03 + 0.04 * load
    vib_noise = rng.exponential(0.005, length)  # 振动噪声偏正态右偏
    signals["vibration_pump"] = np.abs(vib_base + vib_noise)

    # 电机侧振动：通常比泵侧小
    signals["vibration_motor"] = np.abs(signals["vibration_pump"] * 0.6 +
                                        rng.normal(0, 0.003, length))

    # 电流：与液力负荷直接相关
    signals["motor_current"] = 1.5 + 3.0 * load + generate_base_signal(
        length, 0, 0.08, ar_coef=0.97, rng=rng
    )

    # 压力：泵特性曲线决定（流量越大，扬程越低 — 但在运行点附近近似正相关）
    signals["pressure"] = 1.5 + 2.0 * load + generate_base_signal(
        length, 0, 0.05, ar_coef=0.98, rng=rng
    )

    # 电机温度：缓慢响应，与 I²R 相关
    temp_base = 30 + 25 * (load ** 1.2)
    temp_noise = generate_base_signal(length, 0, 0.3, ar_coef=0.9995, rng=rng)
    signals["motor_temp"] = temp_base + temp_noise

    # 循环液温度：受电机散热和环境影响
    signals["fluid_temp"] = signals["motor_temp"] * 0.5 + 10 + generate_base_signal(
        length, 0, 0.2, ar_coef=0.9998, rng=rng
    )

    # 电压：相对稳定，小幅波动
    signals["motor_voltage"] = 230 + generate_base_signal(
        length, 0, 2.0, ar_coef=0.95, rng=rng
    )

    # 流量：与负荷正相关，与压力通过泵曲线关联
    signals["flow_rate"] = 6 + 9 * load + generate_base_signal(
        length, 0, 0.3, ar_coef=0.98, rng=rng
    )

    return signals


# ============================================================================
# 故障注入函数
# ============================================================================

def _inject_cavitation(signals, start, end, severity, rng=None):
    """
    气蚀故障：泵入口压力不足产生气泡

    物理表现：振动剧增(高频) + 压力波动 + 流量下降 + 噪声增大
    """
    amp_map = {"mild": (0.03, 0.06), "moderate": (0.08, 0.15), "severe": (0.18, 0.35)}
    vib_amp = rng.uniform(*amp_map.get(severity, (0.03, 0.06)))

    duration = end - start

    # 振动剧增（气泡破裂产生冲击）
    cav_vib = rng.exponential(vib_amp, duration)
    # 增加间歇性特征（气蚀有不稳定性）
    intermittent = rng.choice([0.3, 1.0, 1.5], duration, p=[0.2, 0.5, 0.3])
    signals["vibration_pump"][start:end] += cav_vib * intermittent

    signals["vibration_motor"][start:end] += cav_vib * 0.4 * intermittent

    # 压力波动增大
    pressure_osc = vib_amp * 8 * np.sin(2 * np.pi * np.arange(duration) / rng.integers(15, 40))
    signals["pressure"][start:end] += pressure_osc
    # 压力均值略降
    signals["pressure"] = inject_gradual_change(signals["pressure"], start, end, -vib_amp * 3, "sigmoid")

    # 流量下降
    signals["flow_rate"] = inject_gradual_change(signals["flow_rate"], start, end, -vib_amp * 12, "sigmoid")

    event = EventMetadata(
        event_type="fault", fault_name="cavitation",
        time_range=(start, end), severity=severity,
        parameters={"vibration_increase": round(vib_amp, 4)},
        affected_channels=["vibration_pump", "vibration_motor", "pressure", "flow_rate"],
        causal_chain=[
            "泵入口液位降低或管路阻力增大，入口压力低于饱和蒸汽压",
            "液体在叶轮入口产生气泡（气蚀现象）",
            f"气泡破裂产生冲击，泵侧振动RMS增大约{vib_amp:.3f}g",
            "振动通过轴传递至电机侧（幅度衰减约60%）",
            f"泵效率下降，流量减少，出口压力出现波动",
            "气蚀具有间歇性特征，振动幅值不均匀",
        ],
        description=(
            f"水泵出现{severity}级气蚀故障。"
            f"第{start}-{end}步振动显著增大（间歇性脉冲），"
            f"压力出现波动，流量下降。特征是振动的间歇性和压力的不稳定。"
        )
    )
    return signals, event


def _inject_bearing_wear(signals, start, end, severity, rng=None):
    """
    轴承磨损：渐进式故障

    物理表现：振动缓慢增大 + 温度渐升 + 电流可能略增
    """
    delta_map = {"mild": (0.01, 0.025), "moderate": (0.03, 0.06), "severe": (0.07, 0.15)}
    vib_delta = rng.uniform(*delta_map.get(severity, (0.01, 0.025)))

    # 振动缓慢增大
    signals["vibration_pump"] = inject_gradual_change(
        signals["vibration_pump"], start, end, vib_delta, "exponential"
    )
    signals["vibration_motor"] = inject_gradual_change(
        signals["vibration_motor"], start, end, vib_delta * 0.7, "exponential"
    )

    # 温度渐升
    temp_delta = vib_delta * 80
    signals["motor_temp"] = inject_gradual_change(
        signals["motor_temp"], start, end, temp_delta, "linear"
    )

    # 电流略增
    signals["motor_current"] = inject_gradual_change(
        signals["motor_current"], start, end, vib_delta * 5, "linear"
    )

    event = EventMetadata(
        event_type="fault", fault_name="bearing_wear",
        time_range=(start, end), severity=severity,
        parameters={"vib_delta": round(vib_delta, 4), "temp_delta": round(temp_delta, 2)},
        affected_channels=["vibration_pump", "vibration_motor", "motor_temp", "motor_current"],
        causal_chain=[
            "轴承滚道或滚动体表面出现微裂纹或点蚀",
            f"振动水平缓慢增大约{vib_delta:.3f}g（指数型增长曲线）",
            "摩擦增大导致轴承发热",
            f"电机温度渐升约{temp_delta:.1f}°C",
            "机械负载增加，电流小幅上升",
            "变化非常缓慢，需要从长期趋势判断",
        ],
        description=(
            f"水泵出现{severity}级轴承磨损。"
            f"从第{start}步开始，振动水平缓慢上升约{vib_delta:.3f}g，"
            f"温度渐升约{temp_delta:.1f}°C。变化缓慢，无突变。"
        )
    )
    return signals, event


def _inject_seal_leak(signals, start, end, severity, rng=None):
    """
    密封泄漏：压力缓慢下降，流量减少

    物理表现：系统压力下降 + 流量减少 + 电流可能下降（泵卸载）
    """
    delta_map = {"mild": (0.2, 0.4), "moderate": (0.5, 0.9), "severe": (1.0, 1.5)}
    p_delta = rng.uniform(*delta_map.get(severity, (0.2, 0.4)))

    signals["pressure"] = inject_gradual_change(
        signals["pressure"], start, end, -p_delta, "linear"
    )
    signals["flow_rate"] = inject_gradual_change(
        signals["flow_rate"], start, end, -p_delta * 3, "linear"
    )
    # 泵卸载，电流下降
    signals["motor_current"] = inject_gradual_change(
        signals["motor_current"], start, end, -p_delta * 0.3, "sigmoid"
    )

    event = EventMetadata(
        event_type="fault", fault_name="seal_leak",
        time_range=(start, end), severity=severity,
        parameters={"pressure_drop": round(p_delta, 3)},
        affected_channels=["pressure", "flow_rate", "motor_current"],
        causal_chain=[
            "泵轴机械密封老化或损伤，出现泄漏通道",
            f"系统压力缓慢下降约{p_delta:.2f}Bar",
            f"有效循环流量减少约{p_delta*3:.1f}L/min",
            "泵液力负荷减小，电流相应下降",
            "若泄漏严重，可能导致泵空转和汽蚀",
        ],
        description=(
            f"水泵密封出现{severity}级泄漏。"
            f"从第{start}步开始压力缓慢下降约{p_delta:.2f}Bar，"
            f"流量同步减少。电流因负荷减小而下降。"
        )
    )
    return signals, event


def _inject_pipe_blockage(signals, start, end, severity, rng=None):
    """
    管道部分堵塞

    物理表现：泵出口压力升高 + 流量下降 + 电流增加（泵在高阻力下工作）
    """
    delta_map = {"mild": (0.3, 0.5), "moderate": (0.6, 1.0), "severe": (1.2, 2.0)}
    p_delta = rng.uniform(*delta_map.get(severity, (0.3, 0.5)))

    signals["pressure"] = inject_gradual_change(
        signals["pressure"], start, end, p_delta, "exponential"
    )
    signals["flow_rate"] = inject_gradual_change(
        signals["flow_rate"], start, end, -p_delta * 2.5, "exponential"
    )
    signals["motor_current"] = inject_gradual_change(
        signals["motor_current"], start, end, p_delta * 0.5, "sigmoid"
    )
    # 温度因电流增加而上升
    signals["motor_temp"] = inject_gradual_change(
        signals["motor_temp"], start + (end-start)//3, end, p_delta * 2, "linear"
    )

    event = EventMetadata(
        event_type="fault", fault_name="pipe_blockage",
        time_range=(start, end), severity=severity,
        parameters={"pressure_rise": round(p_delta, 3)},
        affected_channels=["pressure", "flow_rate", "motor_current", "motor_temp"],
        causal_chain=[
            "管道内沉积物或异物导致过流面积减小",
            f"管路阻力增大，泵出口压力升高约{p_delta:.2f}Bar",
            f"流量因阻力增大而下降约{p_delta*2.5:.1f}L/min",
            "泵工作点偏移，电流增加以维持输出",
            "电流增大导致电机温度上升",
            "关键鉴别：压力升高+流量下降（与密封泄漏相反）",
        ],
        description=(
            f"水泵管道出现{severity}级堵塞。"
            f"压力升高约{p_delta:.2f}Bar，流量下降，电流增加。"
            f"注意：压力和流量的变化方向与密封泄漏完全相反。"
        )
    )
    return signals, event


def _inject_impeller_damage(signals, start, end, severity, rng=None):
    """
    叶轮损伤：效率突降

    物理表现：振动突增 + 流量急降 + 压力降低 + 电流先增后不变（效率低但负荷不变）
    """
    vib_map = {"mild": (0.04, 0.08), "moderate": (0.10, 0.18), "severe": (0.20, 0.35)}
    vib_amp = rng.uniform(*vib_map.get(severity, (0.04, 0.08)))

    # 振动阶跃增加（叶轮不平衡）
    signals["vibration_pump"] = inject_gradual_change(
        signals["vibration_pump"], start, start + max(5, (end-start)//10), vib_amp, "sigmoid"
    )
    # 之后维持高振动+随机波动
    signals["vibration_pump"][start:end] += rng.exponential(vib_amp * 0.3, end - start)

    signals["vibration_motor"][start:end] += vib_amp * 0.5

    # 流量下降（效率降低）
    signals["flow_rate"] = inject_gradual_change(
        signals["flow_rate"], start, end, -vib_amp * 25, "step"
    )

    # 压力也下降
    signals["pressure"] = inject_gradual_change(
        signals["pressure"], start, end, -vib_amp * 5, "step"
    )

    event = EventMetadata(
        event_type="fault", fault_name="impeller_damage",
        time_range=(start, end), severity=severity,
        parameters={"vib_increase": round(vib_amp, 4)},
        affected_channels=["vibration_pump", "vibration_motor", "flow_rate", "pressure"],
        causal_chain=[
            "叶轮叶片发生断裂、腐蚀或磨损",
            f"泵侧振动突然增大约{vib_amp:.3f}g（质量不平衡）",
            "叶轮水力性能下降，效率急剧降低",
            "流量和压力同时显著下降",
            "关键鉴别：振动为突变（非渐变），且流量/压力同时下降",
        ],
        description=(
            f"水泵叶轮出现{severity}级损伤。"
            f"第{start}步振动突增约{vib_amp:.3f}g，"
            f"流量和压力同时下降。区别于轴承磨损的渐变特征。"
        )
    )
    return signals, event


# ============================================================================
# 工厂函数和场景注册
# ============================================================================

def _make_pump_generator(operation_mode, load_level, fault_fn=None,
                          fault_kwargs=None, fault_position_ratio=0.5,
                          fault_duration_ratio=0.15, scenario_id="",
                          scenario_name="", base_desc=""):
    def generate(total_length=5000, seed=42):
        rng = np.random.default_rng(seed)
        signals = _generate_normal_pump(total_length, rng, operation_mode, load_level)
        events = []

        if fault_fn is not None:
            fs = int(total_length * fault_position_ratio)
            fe = min(int(fs + total_length * fault_duration_ratio), total_length - 1)
            kwargs = dict(fault_kwargs) if fault_kwargs else {}
            signals, event = fault_fn(signals, fs, fe, severity=kwargs.get("severity", "moderate"), rng=rng)
            events.append(event)

        df = pd.DataFrame(signals)
        el = [e.time_range[1] - e.time_range[0] for e in events]
        difficulty = compute_difficulty(total_length, el)
        nr = sum(el) / total_length if events else 0.0

        metadata = GenerationMetadata(
            scenario_id=scenario_id, scenario_name=scenario_name,
            domain="pump", domain_display=DOMAIN_DISPLAY,
            total_length=total_length, sampling_rate=SAMPLING_RATE,
            sampling_seconds=SAMPLING_SECONDS, channels=PUMP_CHANNELS,
            base_description=base_desc, injected_events=events,
            overall_description=events[0].description if events else base_desc,
            generation_seed=seed, difficulty=difficulty, needle_ratio=round(nr, 4),
        )
        return df, metadata
    return generate


def register_all_pump_scenarios():
    """注册全部水泵场景（~28个）"""

    # ---- 正常运行 (4个) ----
    for mode, load, sid, desc in [
        ("steady", 0.7, "normal_steady_70",
         "水泵系统70%负荷稳态运行，各传感器参数平稳。"),
        ("steady", 0.9, "normal_steady_90",
         "水泵系统高负荷运行，电流和温度偏高但在正常范围。"),
        ("varying", 0.6, "normal_varying",
         "水泵系统随工艺需求变负荷运行，流量和压力周期性波动。"),
        ("startup", 0.7, "normal_startup",
         "水泵系统启动过程，前10%为升速升载阶段。"),
    ]:
        gen = _make_pump_generator(
            mode, load, scenario_id=sid,
            scenario_name=f"水泵正常-{sid}", base_desc=desc
        )
        register_scenario("pump", sid, f"水泵正常-{sid}", gen, desc)

    # ---- 气蚀 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_cavitation_{sev}"
        gen = _make_pump_generator(
            "steady", 0.7, _inject_cavitation, {"severity": sev},
            0.5, 0.2, sid, f"水泵气蚀-{sev}",
            "水泵运行中出现气蚀现象。"
        )
        register_scenario("pump", sid, f"气蚀-{sev}", gen)

    # ---- 轴承磨损 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_bearing_{sev}"
        gen = _make_pump_generator(
            "steady", 0.7, _inject_bearing_wear, {"severity": sev},
            0.2, 0.55, sid, f"水泵轴承磨损-{sev}",
            "水泵轴承渐进磨损。"
        )
        register_scenario("pump", sid, f"轴承磨损-{sev}", gen)

    # ---- 密封泄漏 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_seal_{sev}"
        gen = _make_pump_generator(
            "steady", 0.7, _inject_seal_leak, {"severity": sev},
            0.5, 0.25, sid, f"水泵密封泄漏-{sev}",
            "水泵机械密封出现泄漏。"
        )
        register_scenario("pump", sid, f"密封泄漏-{sev}", gen)

    # ---- 管道堵塞 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_blockage_{sev}"
        gen = _make_pump_generator(
            "steady", 0.7, _inject_pipe_blockage, {"severity": sev},
            0.55, 0.2, sid, f"水泵管道堵塞-{sev}",
            "水泵管道出现部分堵塞。"
        )
        register_scenario("pump", sid, f"管道堵塞-{sev}", gen)

    # ---- 叶轮损伤 (3个) ----
    for sev in ["mild", "moderate", "severe"]:
        sid = f"fault_impeller_{sev}"
        gen = _make_pump_generator(
            "steady", 0.7, _inject_impeller_damage, {"severity": sev},
            0.55, 0.15, sid, f"水泵叶轮损伤-{sev}",
            "水泵叶轮出现损伤。"
        )
        register_scenario("pump", sid, f"叶轮损伤-{sev}", gen)

    # ---- 不同运行模式下的故障（增加多样性）(6个) ----
    for mode, load, fault_fn, fk, sid_suffix, name in [
        ("varying", 0.6, _inject_cavitation, {"severity": "moderate"},
         "cavitation_varying", "气蚀-变负荷"),
        ("varying", 0.8, _inject_bearing_wear, {"severity": "severe"},
         "bearing_varying", "轴承磨损-变负荷"),
        ("steady", 0.9, _inject_pipe_blockage, {"severity": "moderate"},
         "blockage_highload", "管道堵塞-高负荷"),
        ("steady", 0.5, _inject_seal_leak, {"severity": "severe"},
         "seal_lowload", "密封泄漏-低负荷"),
        ("startup", 0.7, _inject_impeller_damage, {"severity": "mild"},
         "impeller_startup", "叶轮损伤-启动"),
        ("varying", 0.65, _inject_seal_leak, {"severity": "mild"},
         "seal_varying_mild", "密封泄漏-变负荷-轻微"),
    ]:
        sid = f"fault_{sid_suffix}"
        gen = _make_pump_generator(
            mode, load, fault_fn, fk,
            0.5, 0.2, sid, f"水泵{name}",
            f"水泵{mode}运行模式下出现{name}。"
        )
        register_scenario("pump", sid, f"{name}", gen)


register_all_pump_scenarios()
