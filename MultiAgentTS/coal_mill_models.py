"""
磨煤机模型构建模块
Coal Mill Model Construction Module

构建正常运行和各故障工况的磨煤机模型
"""

import numpy as np
from time_blender.core import LambdaEvent, ConstantEvent
from time_blender.random_events import NormalEvent
from time_blender.deterministic_events import WaveEvent
from time_blender.coordination_events import SeasonalEvent, PastEvent
from coal_mill_physics import ThermalLagEvent, ARProcessEvent, PIDControlEvent, VibrationEvent

print("[初始化] 加载磨煤机模型构建模块...")


def create_normal_operation_model():
    """
    创建正常运行工况模型
    
    特点：
    - 日周期负荷变化明显
    - 参数间物理耦合强
    - 时间序列平滑连续
    - 无异常波动
    """
    
    print("  [构建] 正常运行工况模型...", end=" ")
    
    # ========== 日周期负荷 ==========
    def daily_load_cycle(t, i, memory, sub_events):
        hour = t.hour + t.minute / 60.0
        if 6 <= hour < 8:
            return 0.35 + (hour - 6) * 0.25  # 启动
        elif 8 <= hour < 18:
            return 0.85 + 0.05 * np.sin((hour - 8) * np.pi / 10)  # 高负荷
        elif 18 <= hour < 22:
            return 0.85 - (hour - 18) * 0.15  # 下降
        else:
            return 0.3 + 0.05 * np.sin((hour - 0) * np.pi / 6)  # 低负荷
    
    daily_load = LambdaEvent(daily_load_cycle, sub_events={})
    
    # 负荷波动
    load_noise = NormalEvent(mean=0.0, std=0.02)
    
    def combined_load(t, i, memory, sub_events):
        base = sub_events['daily'].execute(t)
        noise = sub_events['noise'].execute(t)
        return np.clip(base + noise, 0.1, 0.95)
    
    total_load = LambdaEvent(combined_load, sub_events={
        'daily': daily_load,
        'noise': load_noise
    })
    
    # ========== 电动机电流 ==========
    def motor_current_calc(t, i, memory, sub_events):
        load = sub_events['load'].execute(t)
        base_current = 48.0
        load_effect = (load - 0.5) * 20
        return np.clip(base_current + load_effect, 40.0, 60.0)
    
    motor_current_component = LambdaEvent(motor_current_calc, sub_events={
        'load': total_load
    })
    
    ar_current = ARProcessEvent(rho=0.93, noise_std=0.4)
    
    def motor_current_final(t, i, memory, sub_events):
        component = sub_events['component'].execute(t)
        ar = sub_events['ar'].execute(t)
        return np.clip(component + ar, 40.0, 60.0)
    
    motor_current = LambdaEvent(motor_current_final, sub_events={
        'component': motor_current_component,
        'ar': ar_current
    })
    
    # ========== 轴承温度 ==========
    def bearing_temp_source(t, i, memory, sub_events):
        current = sub_events['current'].execute(t)
        ambient_temp = 25.0
        return ambient_temp + (current - 48.0) * 0.3
    
    bearing_temp_source = LambdaEvent(bearing_temp_source, sub_events={
        'current': motor_current
    })
    
    bearing_temp_lagged = ThermalLagEvent(bearing_temp_source, time_constant=600.0)
    
    bearing_noise = NormalEvent(mean=0.0, std=0.3)
    
    def bearing_temp_final(t, i, memory, sub_events):
        lagged = sub_events['lagged'].execute(t)
        noise = sub_events['noise'].execute(t)
        return np.clip(lagged + noise, 40.0, 65.0)
    
    bearing_temp_ds = LambdaEvent(bearing_temp_final, sub_events={
        'lagged': bearing_temp_lagged,
        'noise': bearing_noise
    })
    
    bearing_temp_nds = LambdaEvent(bearing_temp_final, sub_events={
        'lagged': ThermalLagEvent(bearing_temp_source, time_constant=700.0),
        'noise': NormalEvent(mean=0.0, std=0.25)
    })
    
    # ========== 绕组温度 ==========
    def winding_temp_calc(t, i, memory, sub_events):
        current = sub_events['current'].execute(t)
        ambient = 25.0
        return ambient + (current - 48.0) * 0.15
    
    winding_temp_source = LambdaEvent(winding_temp_calc, sub_events={
        'current': motor_current
    })
    
    winding_temp_1 = LambdaEvent(
        lambda t, i, m, s: np.clip(
            ThermalLagEvent(winding_temp_source, 400.0)._execute(t, i) +
            NormalEvent(0, 0.2)._execute(t, i),
            35, 55
        ),
        sub_events={}
    )
    
    # ========== 进口温度 ==========
    inlet_base = ConstantEvent(135.0)
    inlet_noise = NormalEvent(mean=0, std=1.0)
    
    def inlet_temp_calc(t, i, memory, sub_events):
        base = sub_events['base'].execute(t)
        noise = sub_events['noise'].execute(t)
        return np.clip(base + noise, 120, 150)
    
    mill_inlet_temp = LambdaEvent(inlet_temp_calc, sub_events={
        'base': inlet_base,
        'noise': inlet_noise
    })
    
    # ========== 阀门开度 ==========
    valve_base = ConstantEvent(52.0)
    valve_noise = NormalEvent(mean=0, std=1.0)
    
    def valve_calc(t, i, memory, sub_events):
        base = sub_events['base'].execute(t)
        noise = sub_events['noise'].execute(t)
        return np.clip(base + noise, 35, 70)
    
    valve_opening = LambdaEvent(valve_calc, sub_events={
        'base': valve_base,
        'noise': valve_noise
    })
    
    print("✓")
    
    return {
        'load': total_load,
        'motor_current': motor_current,
        'bearing_temp_ds': bearing_temp_ds,
        'bearing_temp_nds': bearing_temp_nds,
        'winding_temp_1': winding_temp_1,
        'mill_inlet_temp': mill_inlet_temp,
        'valve_opening': valve_opening
    }


print("[✓] 模型构建模块加载完成")

