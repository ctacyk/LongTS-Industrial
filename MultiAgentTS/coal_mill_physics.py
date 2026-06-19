"""
磨煤机物理建模核心模块
Coal Mill Physics Modeling Core Module

提供基于真实物理特性的TimeBlender事件组件
"""

import numpy as np
from time_blender.core import Event, LambdaEvent, ConstantEvent
from time_blender.random_events import NormalEvent
from time_blender.util import set_random_seed

set_random_seed(42)


class ThermalLagEvent(Event):
    """
    热滞后事件：模拟热传递的延迟效应
    实现一阶动态系统：y[n] = α*x[n] + (1-α)*y[n-1]
    
    物理原理：
    - 热容量导致温度响应延迟
    - 时间常数通常为5-15分钟
    - 在加热时响应较慢，在散热时也响应较慢
    """
    
    def __init__(self, source_event, time_constant=600.0, name=None):
        super().__init__(name, parallel_events=None, push_down=True, allow_learning=False)
        self.source_event = source_event
        self.time_constant = time_constant
        self.current_value = 0.0
        self._causal_parameters.append(source_event)
    
    def _execute(self, t, i):
        dt = 30  # 采样间隔：30秒
        alpha = dt / (self.time_constant + dt)
        input_value = self.source_event.execute(t)
        self.current_value = alpha * input_value + (1 - alpha) * self.current_value
        return self.current_value
    
    def reset(self):
        self.current_value = 0.0
        super().reset()


class ARProcessEvent(Event):
    """
    自回归(AR)过程：实现强时间相关性
    x[n] = ρ*x[n-1] + ε[n]
    
    物理原理：
    - 系统具有惯性，当前状态依赖历史状态
    - AR系数ρ∈[0.85, 0.98]，越大越平滑
    - 实现真实的时间序列连续性
    """
    
    def __init__(self, rho=0.92, noise_std=0.3, name=None):
        super().__init__(name, parallel_events=None, push_down=True, allow_learning=False)
        self.rho = rho
        self.noise_std = noise_std
        self.current_value = 0.0
    
    def _execute(self, t, i):
        noise = np.random.normal(0, self.noise_std)
        self.current_value = self.rho * self.current_value + noise
        return self.current_value
    
    def reset(self):
        self.current_value = 0.0
        super().reset()


class PIDControlEvent(Event):
    """
    PID反馈控制事件：模拟阀门的自动调节
    u[n] = Kp*e + Ki*∑e + Kd*(e[n]-e[n-1])
    
    物理原理：
    - 温度反馈控制阀门开度
    - 存在过冲、振荡、滞后等控制特性
    - 控制品质影响系统稳定性
    """
    
    def __init__(self, target_event, feedback_event, kp=0.5, ki=0.05, kd=0.2, name=None):
        super().__init__(name, parallel_events=None, push_down=True, allow_learning=False)
        self.target_event = target_event
        self.feedback_event = feedback_event
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_error = 0.0
        self.prev_error = 0.0
        self._causal_parameters.extend([target_event, feedback_event])
    
    def _execute(self, t, i):
        target = self.target_event.execute(t)
        feedback = self.feedback_event.execute(t)
        error = target - feedback
        
        self.integral_error += error * 30 / 3600  # 累积误差(小时)
        derivative_error = error - self.prev_error
        self.prev_error = error
        
        output = (self.kp * error + 
                 self.ki * self.integral_error + 
                 self.kd * derivative_error)
        return output
    
    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        super().reset()


class VibrationEvent(Event):
    """
    振动事件：模拟轴承不平衡导致的周期性振动
    使用多谐波合成复杂振动信号
    
    物理原理：
    - 转速对应基频 (通常1-2 Hz)
    - 轴承缺陷产生谐波频率
    - 多个频率叠加产生复杂波形
    """
    
    def __init__(self, base_freq=1.0, harmonics=[1,2,3], amplitudes=None, name=None):
        super().__init__(name, parallel_events=None, push_down=True, allow_learning=False)
        self.base_freq = base_freq
        self.harmonics = harmonics
        self.amplitudes = amplitudes if amplitudes else [1.0/h for h in harmonics]
        self.phase = 0.0
    
    def _execute(self, t, i):
        dt = 30  # 采样间隔
        signal_val = 0.0
        
        for harm, amp in zip(self.harmonics, self.amplitudes):
            freq = self.base_freq * harm
            self.phase += 2 * np.pi * freq * dt
            signal_val += amp * np.sin(self.phase)
        
        return signal_val / sum(self.amplitudes)


class FaultIntensityEvent(Event):
    """
    故障强度事件：随时间增加的故障演进
    模拟故障从轻微到严重的演进过程
    """
    
    def __init__(self, start_time=0, duration=2880, initial_factor=0.1, name=None):
        super().__init__(name, parallel_events=None, push_down=True, allow_learning=False)
        self.start_time = start_time
        self.duration = duration
        self.initial_factor = initial_factor
    
    def _execute(self, t, i):
        if i < self.start_time:
            return 0.0
        elif i < self.start_time + self.duration:
            # 线性从initial_factor增加到1.0
            progress = (i - self.start_time) / self.duration
            return self.initial_factor + (1.0 - self.initial_factor) * progress
        else:
            return 1.0


print("[✓] 物理模块加载完成")

