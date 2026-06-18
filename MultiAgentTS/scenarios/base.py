"""
ScenarioBase: 合成数据场景基础设施

核心设计：
- 每个场景脚本的 generate() 方法同时返回时序DataFrame和元数据
- 元数据中精确记录所有注入事件的位置、类型、参数，作为ground truth
- 提供通用的信号生成原语（趋势、周期、噪声、故障注入）
- 所有数值范围基于真实工业设备参数，由 Cursor Agent 根据领域知识设定

角色说明（防偏差）：
- 本文件及所有场景脚本由 Cursor Agent 直接编写，不经过 LLM Agent
- 元数据中的 description 字段由 Cursor Agent 在脚本中直接定义
- 只有 QAGeneratorAgent 调用 LLM 进行 QA 润色
"""

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Tuple, Optional, Any, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class ChannelInfo:
    """传感器通道信息"""
    name: str                    # 英文标识（用作DataFrame列名）
    display_name: str            # 显示名称（中文）
    unit: str                    # 单位
    normal_range: Tuple[float, float]  # 正常运行时的典型值范围
    alarm_threshold: Optional[float] = None  # 高报警阈值（信号过高时触发）
    alarm_low: Optional[float] = None        # 低报警阈值（信号过低时触发）
    physical_meaning: str = ""   # 物理含义描述


@dataclass
class EventMetadata:
    """单个注入事件的元数据 — 这就是 ground truth"""
    event_type: str              # "anomaly", "trend_change", "fault", "regime_switch", "transient"
    fault_name: str              # 具体故障名称，如 "bearing_overheat"
    time_range: Tuple[int, int]  # 精确的时间步范围 [start, end)
    severity: str                # "mild", "moderate", "severe"
    parameters: Dict[str, Any]   # 故障参数（幅度、斜率、持续时间等）
    affected_channels: List[str] # 受影响的传感器通道名
    causal_chain: List[str]      # 因果链描述（由 Cursor Agent 基于领域知识编写）
    description: str             # 事件的自然语言描述


@dataclass
class GenerationMetadata:
    """完整的生成元数据 — 贯穿整条管线的核心数据结构"""
    scenario_id: str             # 唯一场景ID，如 "coal_mill_bearing_overheat_mild"
    scenario_name: str           # 人类可读名称
    domain: str                  # "coal_mill", "transformer", "pump"
    domain_display: str          # 领域显示名
    total_length: int            # 总时间步数
    sampling_rate: str           # 采样间隔描述，如 "30s"
    sampling_seconds: int        # 采样间隔秒数
    channels: List[ChannelInfo]  # 通道信息列表
    base_description: str        # 基础运行状态描述（正常部分）
    injected_events: List[EventMetadata] = field(default_factory=list)
    overall_description: str = ""  # 整体数据描述（含故障）
    generation_seed: int = 42    # 随机种子（可复现）
    difficulty: str = "medium"   # easy/medium/hard/expert
    needle_ratio: float = 0.0   # 异常占总长度的比例

    def to_dict(self) -> Dict:
        """转为可JSON序列化的字典"""
        d = {}
        for k, v in self.__dict__.items():
            if k == 'channels':
                d[k] = [asdict(ch) for ch in v]
            elif k == 'injected_events':
                d[k] = [asdict(ev) for ev in v]
            else:
                d[k] = v
        return d

    def save(self, path: str):
        """保存元数据为JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> 'GenerationMetadata':
        """从JSON加载"""
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        channels = [ChannelInfo(**ch) for ch in d.pop('channels', [])]
        events = [EventMetadata(**ev) for ev in d.pop('injected_events', [])]
        # fix tuple conversion
        for ev in events:
            if isinstance(ev.time_range, list):
                ev.time_range = tuple(ev.time_range)
        for ch in channels:
            if isinstance(ch.normal_range, list):
                ch.normal_range = tuple(ch.normal_range)
        known = {f.name for f in fields(GenerationMetadata)} - {'channels', 'injected_events'}
        filtered = {k: v for k, v in d.items() if k in known}
        return GenerationMetadata(channels=channels, injected_events=events, **filtered)


# ============================================================================
# 故障幅度校准函数
# ============================================================================

def compute_fault_delta(
    channel: ChannelInfo,
    severity: str,
    direction: str = "increase",
) -> float:
    """基于报警阈值自动计算故障注入幅度

    将 severity 映射为 normal_mean → alarm_threshold 之间距离的比例：
    - mild:     40% of gap → 接近正常上限但未报警
    - moderate: 70% of gap → 接近报警阈值
    - severe:  100% of gap → 达到或超过报警阈值

    Parameters
    ----------
    channel : ChannelInfo
        目标通道信息（包含 normal_range, alarm_threshold, alarm_low）
    severity : str
        "mild" / "moderate" / "severe"
    direction : str
        "increase" → 使用 alarm_threshold (高报警)
        "decrease" → 使用 alarm_low (低报警)

    Returns
    -------
    float
        带符号的 delta 值。increase 为正，decrease 为负。
    """
    normal_mean = (channel.normal_range[0] + channel.normal_range[1]) / 2.0
    sev_map = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
    sev_factor = sev_map[severity]

    if direction == "increase":
        alarm = channel.alarm_threshold
        if alarm is not None:
            gap = alarm - normal_mean
        else:
            gap = (channel.normal_range[1] - normal_mean) * 2.0
        return sev_factor * gap
    else:
        alarm = channel.alarm_low
        if alarm is not None:
            gap = normal_mean - alarm
        else:
            gap = (normal_mean - channel.normal_range[0]) * 2.0
        return -sev_factor * gap


def compute_power_envelope(
    rated_power: float,
    severity: str,
    max_loss_fraction: float = 0.5,
) -> float:
    """计算功率包络限制的最大功率值

    Parameters
    ----------
    rated_power : float
        额定功率 (kW)
    severity : str
        "mild" / "moderate" / "severe"
    max_loss_fraction : float
        severe 时功率损失占额定功率的比例（默认 0.5 = 50%）

    Returns
    -------
    float
        故障状态下的最大功率 (kW)
    """
    sev_map = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
    return rated_power * (1.0 - sev_map[severity] * max_loss_fraction)


# ============================================================================
# 场景注册表
# ============================================================================

_SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_scenario(domain: str, scenario_id: str, name: str,
                      generate_fn: Callable, description: str = ""):
    """注册一个场景生成函数"""
    key = f"{domain}/{scenario_id}"
    _SCENARIO_REGISTRY[key] = {
        'domain': domain,
        'scenario_id': scenario_id,
        'name': name,
        'generate_fn': generate_fn,
        'description': description,
    }


def get_all_scenarios(domain: str = None) -> Dict[str, Dict]:
    """获取所有已注册的场景（可按领域过滤）"""
    if domain is None:
        return _SCENARIO_REGISTRY.copy()
    return {k: v for k, v in _SCENARIO_REGISTRY.items() if v['domain'] == domain}


def get_scenario(domain: str, scenario_id: str) -> Dict:
    """获取单个场景"""
    key = f"{domain}/{scenario_id}"
    if key not in _SCENARIO_REGISTRY:
        raise KeyError(f"Scenario '{key}' not found. Available: {list(_SCENARIO_REGISTRY.keys())}")
    return _SCENARIO_REGISTRY[key]


# ============================================================================
# 信号生成原语 — 构建真实工业时序的基础组件
# ============================================================================

def generate_base_signal(length: int, mean: float, noise_std: float,
                         trend_slope: float = 0.0,
                         seasonal_period: int = 0, seasonal_amplitude: float = 0.0,
                         ar_coef: float = 0.0,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    生成基础信号：均值 + 趋势 + 季节性 + AR(1)自相关 + 噪声

    重要：noise_std 是最终信号的目标标准差（不是 innovation 方差）。
    内部会根据 ar_coef 自动计算正确的 innovation 标准差，确保：
    Var(stationary) = innovation_var / (1 - ar_coef^2) = noise_std^2

    工业时序数据的典型特征：
    - 传感器读数围绕某个工作点波动
    - 存在缓慢趋势（设备老化、环境变化）
    - 可能存在周期性（日周期、班次周期）
    - 相邻时间步高度自相关（物理惯性，ar_coef > 0.99）
    """
    if rng is None:
        rng = np.random.default_rng()

    # 从目标 std 反推 innovation std
    if abs(ar_coef) < 1.0 and abs(ar_coef) > 0:
        innovation_std = noise_std * np.sqrt(1 - ar_coef ** 2)
    else:
        innovation_std = noise_std

    signal = np.zeros(length)
    noise = rng.normal(0, innovation_std, length)

    for i in range(length):
        # 均值 + 趋势
        base = mean + trend_slope * i

        # 季节性
        if seasonal_period > 0 and seasonal_amplitude > 0:
            base += seasonal_amplitude * np.sin(2 * np.pi * i / seasonal_period)

        # AR(1) 自相关 + 噪声
        if i == 0:
            signal[i] = base + noise[i]
        else:
            signal[i] = base + ar_coef * (signal[i-1] - (mean + trend_slope * (i-1))) + noise[i]

    return signal


def inject_gradual_change(signal: np.ndarray, start: int, end: int,
                          delta: float, shape: str = "linear") -> np.ndarray:
    """
    注入渐变事件（如渐进温升、缓慢退化）

    shape: "linear" | "exponential" | "sigmoid" | "step"
    delta: 变化总量（正=上升，负=下降）
    """
    result = signal.copy()
    duration = end - start
    if duration <= 0:
        return result

    t = np.linspace(0, 1, duration)

    if shape == "linear":
        profile = t * delta
    elif shape == "exponential":
        profile = (np.exp(3 * t) - 1) / (np.exp(3) - 1) * delta
    elif shape == "sigmoid":
        profile = 1 / (1 + np.exp(-10 * (t - 0.5))) * delta
    elif shape == "step":
        profile = np.ones(duration) * delta
    else:
        profile = t * delta

    result[start:end] += profile
    # 故障后保持最终值（如果 end < len）
    if end < len(result):
        result[end:] += delta

    return result


def inject_spike(signal: np.ndarray, position: int, amplitude: float,
                 width: int = 1, shape: str = "gaussian",
                 rng: np.random.Generator = None) -> np.ndarray:
    """
    注入脉冲/尖峰事件（如电流突变、压力冲击）

    shape: "gaussian" | "sharp" | "oscillating"
    """
    result = signal.copy()
    if rng is None:
        rng = np.random.default_rng()

    half_w = width // 2
    start = max(0, position - half_w)
    end = min(len(signal), position + half_w + 1)

    if shape == "gaussian":
        x = np.arange(start, end) - position
        profile = amplitude * np.exp(-0.5 * (x / max(half_w / 3, 1)) ** 2)
    elif shape == "sharp":
        profile = np.zeros(end - start)
        profile[position - start] = amplitude
    elif shape == "oscillating":
        x = np.arange(start, end) - position
        decay = np.exp(-np.abs(x) / max(half_w / 2, 1))
        profile = amplitude * np.sin(2 * np.pi * x / max(width / 3, 2)) * decay
    else:
        profile = np.ones(end - start) * amplitude

    result[start:end] += profile
    return result


def inject_intermittent_fault(signal: np.ndarray, start: int, end: int,
                              amplitude: float, on_duration: int = 10,
                              off_duration: int = 50,
                              rng: np.random.Generator = None) -> np.ndarray:
    """
    注入间歇性故障（如接触不良、间歇性堵塞）

    在 [start, end) 区间内，以 on_duration 和 off_duration 交替出现故障
    """
    result = signal.copy()
    if rng is None:
        rng = np.random.default_rng()

    pos = start
    is_on = False
    while pos < end:
        if is_on:
            burst_end = min(pos + on_duration, end)
            result[pos:burst_end] += amplitude * (1 + rng.normal(0, 0.1, burst_end - pos))
            pos = burst_end
        else:
            pos += off_duration + rng.integers(-off_duration // 4, off_duration // 4 + 1)
        is_on = not is_on

    return result


def apply_thermal_lag(source: np.ndarray, target: np.ndarray,
                      coupling_coef: float, lag_steps: int,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    模拟热传导滞后效应

    工业设备中，一个部件的温度变化会滞后地影响相邻部件。
    例如：轴承温度升高 → 经过若干步后 → 绕组温度也升高（但幅度衰减）

    source: 源信号（如轴承温度）
    target: 目标信号（如绕组温度）
    coupling_coef: 耦合系数（0-1，表示影响程度）
    lag_steps: 滞后步数
    """
    result = target.copy()
    if rng is None:
        rng = np.random.default_rng()

    # 计算源信号的变化量
    source_diff = np.diff(source, prepend=source[0])

    # 滞后传播：源信号的变化经过 lag_steps 后以衰减形式影响目标
    for i in range(lag_steps, len(result)):
        # 累积前 lag_steps 个时间步的源信号变化，加权衰减
        influence = 0.0
        for lag in range(lag_steps):
            weight = coupling_coef * np.exp(-lag / max(lag_steps / 3, 1))
            if i - lag >= 0:
                influence += weight * source_diff[i - lag]
        result[i] += influence

    return result


def apply_load_coupling(signals: Dict[str, np.ndarray],
                        load_signal: np.ndarray,
                        coupling_map: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    模拟负载耦合效应

    工业设备中，负载变化会同时影响多个传感器。
    例如：负载增加 → 电流增大 + 温度上升 + 振动增加

    coupling_map: {channel_name: coupling_coefficient}
    """
    result = {}
    # 负载的归一化变化量
    load_norm = (load_signal - np.mean(load_signal)) / (np.std(load_signal) + 1e-8)

    for name, sig in signals.items():
        if name in coupling_map:
            coef = coupling_map[name]
            # 负载变化线性耦合到各通道
            result[name] = sig + coef * load_norm * np.std(sig)
        else:
            result[name] = sig.copy()

    return result


def compute_difficulty(total_length: int, event_lengths: List[int]) -> str:
    """根据总长度和异常占比计算难度等级"""
    if total_length <= 0:
        return "easy"
    total_event = sum(event_lengths)
    ratio = total_event / total_length
    if total_length <= 1000:
        return "easy"
    elif total_length <= 3000:
        return "medium" if ratio > 0.05 else "hard"
    elif total_length <= 5000:
        return "hard" if ratio > 0.03 else "expert"
    else:
        return "expert"


# ============================================================================
# 可视化与保存工具
# ============================================================================

def save_scenario_output(df: pd.DataFrame, metadata: GenerationMetadata,
                         output_dir: str, generate_plot: bool = True):
    """
    保存场景输出：CSV + metadata.json + 可视化

    output_dir: 输出目录（如 Data/benchmark/coal_mill/）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名：scenario_id + length + seed
    base_name = f"{metadata.scenario_id}_L{metadata.total_length}_S{metadata.generation_seed}"

    # 保存 CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)

    # 保存元数据
    meta_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    metadata.save(meta_path)

    # 生成可视化
    if generate_plot:
        plot_path = os.path.join(output_dir, f"{base_name}_overview.png")
        plot_scenario_overview(df, metadata, plot_path)

    return csv_path, meta_path


def plot_scenario_overview(df: pd.DataFrame, metadata: GenerationMetadata,
                           save_path: str):
    """生成场景概览图：所有通道 + 故障区间高亮（用于人工审查）"""
    n_channels = len(df.columns)
    fig, axes = plt.subplots(n_channels, 1, figsize=(16, 3 * n_channels),
                              sharex=True)
    if n_channels == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_channels))

    for idx, col in enumerate(df.columns):
        ax = axes[idx]
        ax.plot(df[col].values, color=colors[idx], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)

        # 高亮故障区间
        for event in metadata.injected_events:
            if col in event.affected_channels or len(event.affected_channels) == 0:
                ax.axvspan(event.time_range[0], event.time_range[1],
                          alpha=0.2, color='red',
                          label=f"{event.fault_name} ({event.severity})")

        if idx == 0:
            ax.set_title(f"{metadata.scenario_name} | L={metadata.total_length} | "
                        f"Difficulty={metadata.difficulty}", fontsize=10)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 评测专用可视化（给 VLM 看的，不含任何答案泄漏）
# ============================================================================

# 通道名 → 英文显示名 + 单位的映射
CHANNEL_ENGLISH_LABELS = {
    # Coal Mill
    "motor_current": "Motor Current (A)",
    "winding_temp_1": "Winding Temp 1 (deg C)",
    "winding_temp_2": "Winding Temp 2 (deg C)",
    "bearing_temp_nondrive": "Bearing Temp Non-Drive (deg C)",
    "bearing_temp_drive": "Bearing Temp Drive (deg C)",
    "inlet_temp": "Inlet Temperature (deg C)",
    "inlet_valve_opening": "Inlet Valve Opening (%)",
    # Transformer
    "HUFL": "High Useful Load (MW)",
    "HULL": "High Useless Load (MVar)",
    "MUFL": "Middle Useful Load (MW)",
    "MULL": "Middle Useless Load (MVar)",
    "LUFL": "Low Useful Load (MW)",
    "LULL": "Low Useless Load (MVar)",
    "OT": "Oil Temperature (deg C)",
    # Pump
    "vibration_pump": "Pump Vibration RMS (g)",
    "vibration_motor": "Motor Vibration RMS (g)",
    "pressure": "Outlet Pressure (Bar)",
    "motor_temp": "Motor Body Temp (deg C)",
    "fluid_temp": "Fluid Temperature (deg C)",
    "motor_voltage": "Motor Voltage (V)",
    "flow_rate": "Flow Rate (L/min)",
    # Wind Turbine
    "wind_speed": "Wind Speed (m/s)",
    "rotor_speed": "Rotor Speed (RPM)",
    "generator_speed": "Generator Speed (RPM)",
    "power_output": "Active Power (kW)",
    "pitch_angle": "Pitch Angle (deg)",
    "nacelle_temp": "Nacelle Temperature (deg C)",
    "gearbox_oil_temp": "Gearbox Oil Temp (deg C)",
    "gen_bearing_temp": "Generator Bearing Temp (deg C)",
}

# 高对比色板 (tab10)
EVAL_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# 领域名英文映射
DOMAIN_ENGLISH = {
    "coal_mill": "Coal Mill",
    "transformer": "Transformer",
    "pump": "Water Pump",
    "wind_turbine": "Wind Turbine",
    "aero_engine": "Aero Engine",
}


def plot_eval_chart(df: pd.DataFrame, metadata: GenerationMetadata,
                    save_path: str):
    """
    生成评测专用可视化图（给 VLM 输入用）

    设计原则：
    - 高对比色板，每通道颜色清晰可辨
    - 英文标签（VLM 对英文识别更好）
    - 无故障高亮（防止答案泄漏）
    - 标题只显示领域和数据长度，不暴露故障信息
    - 较粗线宽 + 网格线辅助读数
    """
    n_channels = len(df.columns)
    fig, axes = plt.subplots(n_channels, 1,
                              figsize=(16, 2.5 * n_channels),
                              sharex=True)
    if n_channels == 1:
        axes = [axes]

    for idx, col in enumerate(df.columns):
        ax = axes[idx]
        color = EVAL_COLORS[idx % len(EVAL_COLORS)]

        # 主线：加粗、高对比
        ax.plot(df[col].values, color=color, linewidth=1.2, alpha=0.9)

        # Y 轴标签：英文 + 单位
        label = CHANNEL_ENGLISH_LABELS.get(col, col)
        ax.set_ylabel(label, fontsize=9, fontweight='bold')

        # 网格：浅灰虚线
        ax.grid(True, alpha=0.3, linestyle='--', color='#cccccc')

        # Y 轴刻度字体
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)

    # 标题：只显示领域 + 长度 + 采样率（不暴露故障信息）
    domain_en = DOMAIN_ENGLISH.get(metadata.domain, metadata.domain)
    title = f"{domain_en} | Length={metadata.total_length} | Sampling={metadata.sampling_rate}"
    axes[0].set_title(title, fontsize=11, fontweight='bold', pad=10)

    # X 轴标签
    axes[-1].set_xlabel("Time Step", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


def batch_generate_eval_charts(data_dir: str, output_suffix: str = "_eval.png"):
    """
    批量生成评测专用可视化

    遍历 data_dir 下所有 *_metadata.json，
    读取对应 CSV，生成 *_eval.png
    """
    import glob

    meta_files = glob.glob(os.path.join(data_dir, "**", "*_metadata.json"), recursive=True)
    print(f"Found {len(meta_files)} metadata files")

    generated = 0
    skipped = 0

    for meta_path in sorted(meta_files):
        csv_path = meta_path.replace("_metadata.json", ".csv")
        eval_png_path = meta_path.replace("_metadata.json", output_suffix)

        # 跳过已存在
        if os.path.exists(eval_png_path):
            skipped += 1
            continue

        if not os.path.exists(csv_path):
            continue

        try:
            metadata = GenerationMetadata.load(meta_path)
            df = pd.read_csv(csv_path)
            plot_eval_chart(df, metadata, eval_png_path)
            generated += 1

            if generated % 100 == 0:
                print(f"  Generated {generated} eval charts...")

        except Exception as e:
            print(f"  Error: {meta_path}: {e}")

    print(f"Done: generated={generated}, skipped={skipped}")
    return generated
