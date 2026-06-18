"""
磨煤机 TimeBlender 场景库 — 15种故障 + 3种正常工况

遵循 data-synthesis-loop rule：
- 所有温度/慢响应通道使用 LambdaEvent + memory 热惯性递推
- 通道间通过 sub_events 依赖图自然耦合
- 参数基于真实数据统计校准

真实数据校准目标（Dataset/CoalMill/merged.csv）：
  电机电流:   mean=47.3, std=13.3, AC1=0.998
  绕组温度1:  mean=50.4, std=7.1,  AC1=0.9999
  绕组温度2:  mean=45.1, std=6.8,  AC1=0.9999
  非驱侧轴承: mean=23.9, std=5.9,  AC1=0.9999
  驱动侧轴承: mean=31.8, std=6.2,  AC1=0.9994
  入口温度:   mean=270.8,std=55.0, AC1=0.9997
  阀门开度:   mean=92.6, std=22.8, AC1=0.999
  停机: 27次/30天, 均1.2h, 间隔24.8h, 占4.5%

故障谱（15种）：
  机械类: 轴承过热(DS/NDS), 渐进磨损, 磨辊卡涩, 传动松动, 给煤机故障
  热力类: 冷却故障, 入口堵塞, 出口温度振荡, 热风管道泄漏
  电气类: 电流冲击, 绕组匝间短路, 电压波动
  工况异常: 超负荷, 频繁启停, 负荷突变
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from time_blender.core import Generator, LambdaEvent, ConstantEvent
from MultiAgentTS.scenarios.base import (
    GenerationMetadata, EventMetadata, ChannelInfo,
    compute_difficulty, save_scenario_output, plot_eval_chart,
    register_scenario, get_all_scenarios, compute_fault_delta,
)

# ============================================================
# 通道定义
# ============================================================
COAL_MILL_CHANNELS = [
    ChannelInfo("motor_current", "电机电流", "A",
                normal_range=(35.0, 60.0), alarm_threshold=70.0, alarm_low=25.0,
                physical_meaning="反映磨煤机研磨负荷，低于25A表示空载/给煤中断"),
    ChannelInfo("winding_temp_1", "电机绕组温度1", "°C",
                normal_range=(42.0, 62.0), alarm_threshold=85.0,
                physical_meaning="电机定子绕组温度，F级绝缘限值155°C"),
    ChannelInfo("winding_temp_2", "电机绕组温度2", "°C",
                normal_range=(38.0, 58.0), alarm_threshold=80.0,
                physical_meaning="冗余温度传感器"),
    ChannelInfo("bearing_temp_nondrive", "非驱动侧轴承温度", "°C",
                normal_range=(18.0, 35.0), alarm_threshold=55.0,
                physical_meaning="远离负载端轴承温度"),
    ChannelInfo("bearing_temp_drive", "驱动侧轴承温度", "°C",
                normal_range=(25.0, 42.0), alarm_threshold=60.0,
                physical_meaning="承载端轴承温度"),
    ChannelInfo("inlet_temp", "磨机入口温度", "°C",
                normal_range=(240.0, 310.0), alarm_threshold=330.0, alarm_low=180.0,
                physical_meaning="热风入口温度，低于180°C表示热风供给不足"),
    ChannelInfo("inlet_valve_opening", "入口调节阀开度", "%",
                normal_range=(75.0, 100.0), alarm_threshold=None,
                physical_meaning="控制热风进入量"),
]

DOMAIN = "coal_mill"
DOMAIN_DISPLAY = "磨煤机"
SAMPLING_RATE = "30s"
SAMPLING_SECONDS = 30

_CH = {ch.name: ch for ch in COAL_MILL_CHANNELS}


# ============================================================
# TimeBlender 基础构建块
# ============================================================
def thermal_inertia(t, i, memory, sub_events):
    """一阶低通递推"""
    target = sub_events['target'].execute(t)
    alpha = sub_events['alpha'].execute(t)
    noise_std = sub_events['noise'].execute(t)
    if 'prev' not in memory:
        memory['prev'] = target
    prev = memory['prev']
    memory['prev'] = prev + alpha * (target - prev) + np.random.normal(0, noise_std)
    return memory['prev']


# ============================================================
# 负荷模型（多种工况）
# ============================================================
def _load_steady(t, i, memory, sub_events):
    """稳态运行 + 随机停机"""
    if 'state' not in memory:
        memory.update({'state': 'running', 'load': 0.7, 'sd_timer': 0,
                       'su_timer': 0, 'next_sd': np.random.randint(2000, 5000), 'step': 0})
    memory['step'] += 1
    s = memory
    if s['state'] == 'running':
        s['load'] = np.clip(s['load'] + np.random.normal(0, 0.002), 0.55, 0.92)
        if s['step'] > s['next_sd']:
            s['state'] = 'shutdown'
            s['sd_timer'] = np.random.randint(80, 250)
            s['next_sd'] = s['step'] + s['sd_timer'] + np.random.randint(2000, 5000)
    elif s['state'] == 'shutdown':
        s['load'] = max(0.0, s['load'] - 0.08)
        s['sd_timer'] -= 1
        if s['sd_timer'] <= 0:
            s['state'] = 'starting'; s['su_timer'] = np.random.randint(60, 180)
    elif s['state'] == 'starting':
        s['load'] = min(0.75, s['load'] + 0.006)
        s['su_timer'] -= 1
        if s['su_timer'] <= 0: s['state'] = 'running'
    return s['load']


def _load_varying(t, i, memory, sub_events):
    """变负荷运行"""
    if 'load' not in memory:
        memory['load'] = 0.65; memory['step'] = 0
    memory['step'] += 1
    # 大幅波动 + 停机
    drift = np.random.normal(0, 0.004) + 0.03 * np.sin(2 * np.pi * memory['step'] / 3000)
    memory['load'] = np.clip(memory['load'] + drift, 0.3, 0.95)
    # 偶尔短暂减载
    if np.random.random() < 0.0003:
        memory['load'] = 0.2
    return memory['load']


def _load_frequent_startstop(t, i, memory, sub_events):
    """频繁启停工况"""
    if 'state' not in memory:
        memory.update({'state': 'running', 'load': 0.7, 'sd_timer': 0,
                       'su_timer': 0, 'next_sd': np.random.randint(300, 800), 'step': 0})
    memory['step'] += 1
    s = memory
    if s['state'] == 'running':
        s['load'] = np.clip(s['load'] + np.random.normal(0, 0.002), 0.55, 0.85)
        if s['step'] > s['next_sd']:
            s['state'] = 'shutdown'
            s['sd_timer'] = np.random.randint(40, 120)
            s['next_sd'] = s['step'] + s['sd_timer'] + np.random.randint(300, 800)
    elif s['state'] == 'shutdown':
        s['load'] = max(0.0, s['load'] - 0.1)
        s['sd_timer'] -= 1
        if s['sd_timer'] <= 0:
            s['state'] = 'starting'; s['su_timer'] = np.random.randint(40, 100)
    elif s['state'] == 'starting':
        s['load'] = min(0.7, s['load'] + 0.008)
        s['su_timer'] -= 1
        if s['su_timer'] <= 0: s['state'] = 'running'
    return s['load']


# ============================================================
# 通道目标值函数
# ============================================================
def _current_tgt(t, i, m, s):
    return 10 + 55 * s['load'].execute(t) + np.random.normal(0, 1.5)

def _winding_tgt(t, i, m, s):
    c = s['current'].execute(t)
    return 22 + 2 * np.sin(2 * np.pi * i / 2880) + (c ** 2) * 0.010

def _winding2(t, i, m, s):
    w1 = s['w1'].execute(t)
    if 'p' not in m: m['p'] = w1 - 4.5
    m['p'] += 0.01 * (w1 - 4.5 + np.random.normal(0, 0.1) - m['p'])
    return m['p']

def _bearing_nds_tgt(t, i, m, s):
    return 15 + 4 * np.sin(2 * np.pi * i / 2880) + s['current'].execute(t) * 0.15

def _bearing_ds_tgt(t, i, m, s):
    return 18 + 4 * np.sin(2 * np.pi * i / 2880) + s['current'].execute(t) * 0.22 + 4.0

def _inlet_tgt(t, i, m, s):
    ld = s['load'].execute(t)
    return (260 + 55 * ld + np.random.normal(0, 2)) if ld > 0.1 else (20 + np.random.normal(0, 1))

def _valve_tgt(t, i, m, s):
    ld = s['load'].execute(t)
    return (78 + 22 * ld + np.random.normal(0, 1)) if ld > 0.1 else max(0, np.random.normal(0, 0.5))


# ============================================================
# 故障目标值修改器（叠加在正常目标之上）
# ============================================================
def _fault_ramp(t, i, m, s):
    """通用渐变故障：在故障区间内 target 增加 delta*progress"""
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
    """振荡故障：在故障区间叠加周期性波动"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    amp = s['amp'].execute(t)
    period = s['period'].execute(t)
    if fs <= i < fe:
        return base + amp * np.sin(2 * np.pi * (i - fs) / period) * (1 + np.random.normal(0, 0.2))
    return base


def _fault_spike(t, i, m, s):
    """脉冲故障：随机出现尖峰"""
    base = s['base'].execute(t)
    fs, fe = int(s['fs'].execute(t)), int(s['fe'].execute(t))
    amp = s['amp'].execute(t)
    prob = s['prob'].execute(t)
    if fs <= i < fe and np.random.random() < prob:
        return base + np.random.uniform(0.5, 1.0) * amp
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


# ============================================================
# 模型构建工厂
# ============================================================
def _build_base_model(load_fn):
    """构建基础模型（正常运行），返回事件字典 + 关键中间事件引用"""
    load = LambdaEvent(load_fn, sub_events={})
    ct = LambdaEvent(_current_tgt, sub_events={'load': load})
    current = LambdaEvent(thermal_inertia, sub_events={
        'target': ct, 'alpha': ConstantEvent(0.3), 'noise': ConstantEvent(0.5)})
    wt = LambdaEvent(_winding_tgt, sub_events={'current': current})
    w1 = LambdaEvent(thermal_inertia, sub_events={
        'target': wt, 'alpha': ConstantEvent(0.006), 'noise': ConstantEvent(0.06)})
    w2 = LambdaEvent(_winding2, sub_events={'w1': w1})
    bnt = LambdaEvent(_bearing_nds_tgt, sub_events={'current': current})
    bn = LambdaEvent(thermal_inertia, sub_events={
        'target': bnt, 'alpha': ConstantEvent(0.004), 'noise': ConstantEvent(0.05)})
    bdt = LambdaEvent(_bearing_ds_tgt, sub_events={'current': current})
    bd = LambdaEvent(thermal_inertia, sub_events={
        'target': bdt, 'alpha': ConstantEvent(0.004), 'noise': ConstantEvent(0.06)})
    it = LambdaEvent(_inlet_tgt, sub_events={'load': load})
    inlet = LambdaEvent(thermal_inertia, sub_events={
        'target': it, 'alpha': ConstantEvent(0.06), 'noise': ConstantEvent(1.0)})
    vt = LambdaEvent(_valve_tgt, sub_events={'load': load})
    valve = LambdaEvent(thermal_inertia, sub_events={
        'target': vt, 'alpha': ConstantEvent(0.12), 'noise': ConstantEvent(0.3)})

    events = {
        'motor_current': current, 'winding_temp_1': w1, 'winding_temp_2': w2,
        'bearing_temp_nondrive': bn, 'bearing_temp_drive': bd,
        'inlet_temp': inlet, 'inlet_valve_opening': valve,
    }
    refs = {'load': load, 'current': current, 'ct': ct, 'wt': wt,
            'bnt': bnt, 'bdt': bdt, 'it': it, 'vt': vt, 'w1': w1}
    return events, refs


def _inject_fault(load_fn, fault_type, fs, fe, severity, params):
    """
    构建带故障注入的模型。
    所有 delta 基于 compute_fault_delta() 从报警阈值自动校准。
    所有因果链声明的受影响通道都有实际注入（无幽灵通道）。
    """
    load = LambdaEvent(load_fn, sub_events={})
    ct = LambdaEvent(_current_tgt, sub_events={'load': load})
    bnt_base = LambdaEvent(_bearing_nds_tgt, sub_events={'current': None})
    bdt_base = LambdaEvent(_bearing_ds_tgt, sub_events={'current': None})
    wt_base = LambdaEvent(_winding_tgt, sub_events={'current': None})
    it_base = LambdaEvent(_inlet_tgt, sub_events={'load': load})
    vt_base = LambdaEvent(_valve_tgt, sub_events={'load': load})

    cur_alpha, cur_noise = 0.3, 0.5
    w_alpha, w_noise = 0.006, 0.06
    bn_alpha, bn_noise = 0.004, 0.05
    bd_alpha, bd_noise = 0.004, 0.06
    in_alpha, in_noise = 0.06, 1.0
    va_alpha, va_noise = 0.12, 0.3

    ct_final = ct
    wt_final = wt_base
    bnt_final = bnt_base
    bdt_final = bdt_base
    it_final = it_base
    vt_final = vt_base
    w_alpha_ev = ConstantEvent(w_alpha)
    _spike_params = None  # set by current_spike to bypass thermal_inertia

    fsc, fec = ConstantEvent(float(fs)), ConstantEvent(float(fe))

    # ==== A-class: 幽灵通道修复 + 根因驱动耦合 ====

    if fault_type == "bearing_overheat_drive":
        # 根因: 驱动侧轴承润滑劣化 → 摩擦增大
        d_bd = compute_fault_delta(_CH['bearing_temp_drive'], severity, "increase")
        bdt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': bdt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_bd)})
        # 耦合: 轴承热传导至绕组 (~30% of bearing delta)
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.3
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})
        # 耦合: 轴承摩擦 → 机械负载增加 → 电流小幅上升
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.2
        ct_final = LambdaEvent(_fault_ramp, sub_events={
            'base': ct, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ct)})

    elif fault_type == "bearing_overheat_nondrive":
        # 根因: 非驱动侧轴承磨损 → 摩擦增大
        d_bn = compute_fault_delta(_CH['bearing_temp_nondrive'], severity, "increase")
        bnt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': bnt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_bn)})
        # 耦合: 热传导至绕组 (~25%)
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.25
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})

    elif fault_type == "roller_jam":
        # 根因: 磨辊/磨盘卡涩 → 研磨阻力骤增
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase")
        ct_final = LambdaEvent(_fault_step, sub_events={
            'base': ct, 'fs': fsc, 'delta': ConstantEvent(d_ct)})
        # 耦合: I²R → 绕组温度快速升高 (~50%)
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.5
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})

    elif fault_type == "overload":
        # 根因: 超额定负荷运行
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase")
        ct_final = LambdaEvent(_fault_step, sub_events={
            'base': ct, 'fs': fsc, 'delta': ConstantEvent(d_ct)})
        # 耦合: I²R → 绕组温度升高 (~60%)
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.6
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})
        # 耦合: 整体温升 → 轴承温度升高 (~30%)
        d_bd = compute_fault_delta(_CH['bearing_temp_drive'], severity, "increase") * 0.3
        bdt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': bdt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_bd)})

    elif fault_type == "load_surge":
        # 根因: 电网调度负荷突变
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase")
        ct_final = LambdaEvent(_fault_step, sub_events={
            'base': ct, 'fs': fsc, 'delta': ConstantEvent(d_ct)})
        # 耦合: I²R → 绕组温度滞后响应 (~40%)
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.4
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})
        # 耦合: 负荷增加 → 需要更多热风 → 入口温度响应 (~20%)
        d_it = compute_fault_delta(_CH['inlet_temp'], severity, "increase") * 0.2
        it_final = LambdaEvent(_fault_ramp, sub_events={
            'base': it_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_it)})

    elif fault_type == "current_spike":
        # 根因: 大块煤/异物进入磨盘 → 瞬时冲击
        # 脉冲直接注入到 thermal_inertia 输出（绕过滤波），保持尖峰锐度
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase")
        _spike_params = {'amp': d_ct, 'prob': 0.03}
        # ct_final 保持正常（不修改 target），脉冲在输出端注入
        # 耦合: I²R瞬时累积 → 绕组温度小幅上升 (~15%)
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.15
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})

    # ==== B-class: 补充缺失耦合 ====

    elif fault_type == "gradual_degradation":
        # 磨辊磨损 → 研磨效率降低 → 电流爬升
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase")
        ct_final = LambdaEvent(_fault_ramp, sub_events={
            'base': ct, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ct)})
        # 机械磨损 → 轴承温升 (~40%)
        d_bd = compute_fault_delta(_CH['bearing_temp_drive'], severity, "increase") * 0.4
        bdt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': bdt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_bd)})
        # I²R → 绕组温升 (~30%), 补充原缺失的耦合
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase") * 0.3
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})

    elif fault_type == "inlet_blockage":
        # 入口堵塞 → 通风面积减小 → 入口温度升高
        d_it = compute_fault_delta(_CH['inlet_temp'], severity, "increase")
        it_final = LambdaEvent(_fault_ramp, sub_events={
            'base': it_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_it)})
        # 控制系统增大阀门补偿 (固定响应幅度 ~8%)
        vt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': vt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(8.0)})
        # 补充: 通风受阻 → 研磨阻力增加 → 电流上升 (~30%)
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.3
        ct_final = LambdaEvent(_fault_ramp, sub_events={
            'base': ct, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_ct)})

    elif fault_type == "cooling_failure":
        # 冷却故障: 散热系数降低 → 绕组温度自然升高
        sev_map = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}
        reduction = sev_map[severity] * 0.8
        w_alpha_ev = LambdaEvent(_fault_alpha_reduce, sub_events={
            'normal_alpha': ConstantEvent(w_alpha), 'fs': fsc, 'fe': fec,
            'reduction': ConstantEvent(reduction)})

    elif fault_type == "duct_leak":
        # 热风泄漏 → 入口温度下降 + 阀门大开补偿
        d_it = compute_fault_delta(_CH['inlet_temp'], severity, "decrease")
        it_final = LambdaEvent(_fault_ramp, sub_events={
            'base': it_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_it)})
        # 阀门大开补偿 (最大开至 100%)
        vt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': vt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(10.0)})

    # ==== C-class: 仅 delta 校准 (结构不变) ====

    elif fault_type == "drive_looseness":
        # 传动松动: 电流周期振荡 (~60% of alarm gap)
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.6
        ct_final = LambdaEvent(_fault_oscillation, sub_events={
            'base': ct, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_ct), 'period': ConstantEvent(40.0)})

    elif fault_type == "feeder_fault":
        # 给煤不均: 电流大幅不规则波动 (~80% of alarm gap)
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.8
        ct_final = LambdaEvent(_fault_oscillation, sub_events={
            'base': ct, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_ct), 'period': ConstantEvent(80.0)})

    elif fault_type == "outlet_oscillation":
        # 出口温度振荡 → PID 控制异常 → 入口温度振荡 (~50%)
        d_it = compute_fault_delta(_CH['inlet_temp'], severity, "increase") * 0.5
        it_final = LambdaEvent(_fault_oscillation, sub_events={
            'base': it_base, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_it), 'period': ConstantEvent(60.0)})

    elif fault_type == "winding_short":
        # 绕组匝间短路: 局部热点 → 温度升高
        d_wt = compute_fault_delta(_CH['winding_temp_1'], severity, "increase")
        wt_final = LambdaEvent(_fault_ramp, sub_events={
            'base': wt_base, 'fs': fsc, 'fe': fec, 'delta': ConstantEvent(d_wt)})

    elif fault_type == "voltage_fluctuation":
        # 电压波动: 电流高频波动 (~50% of alarm gap)
        d_ct = compute_fault_delta(_CH['motor_current'], severity, "increase") * 0.5
        ct_final = LambdaEvent(_fault_oscillation, sub_events={
            'base': ct, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(d_ct), 'period': ConstantEvent(25.0)})

    # ==== 构建完整模型 ====
    current = LambdaEvent(thermal_inertia, sub_events={
        'target': ct_final, 'alpha': ConstantEvent(cur_alpha), 'noise': ConstantEvent(cur_noise)})

    # 脉冲绕过热惯性：注入到输出端而非 target 端
    if _spike_params is not None:
        current = LambdaEvent(_fault_spike, sub_events={
            'base': current, 'fs': fsc, 'fe': fec,
            'amp': ConstantEvent(_spike_params['amp']),
            'prob': ConstantEvent(_spike_params['prob'])})

    bnt_base._causal_parameters = []
    bnt_base.sub_events = {'current': current}
    bdt_base._causal_parameters = []
    bdt_base.sub_events = {'current': current}
    wt_base._causal_parameters = []
    wt_base.sub_events = {'current': current}

    w1 = LambdaEvent(thermal_inertia, sub_events={
        'target': wt_final, 'alpha': w_alpha_ev, 'noise': ConstantEvent(w_noise)})
    w2 = LambdaEvent(_winding2, sub_events={'w1': w1})
    bn = LambdaEvent(thermal_inertia, sub_events={
        'target': bnt_final, 'alpha': ConstantEvent(bn_alpha), 'noise': ConstantEvent(bn_noise)})
    bd = LambdaEvent(thermal_inertia, sub_events={
        'target': bdt_final, 'alpha': ConstantEvent(bd_alpha), 'noise': ConstantEvent(bd_noise)})
    inlet = LambdaEvent(thermal_inertia, sub_events={
        'target': it_final, 'alpha': ConstantEvent(in_alpha), 'noise': ConstantEvent(in_noise)})
    valve = LambdaEvent(thermal_inertia, sub_events={
        'target': vt_final, 'alpha': ConstantEvent(va_alpha), 'noise': ConstantEvent(va_noise)})

    return {
        'motor_current': current, 'winding_temp_1': w1, 'winding_temp_2': w2,
        'bearing_temp_nondrive': bn, 'bearing_temp_drive': bd,
        'inlet_temp': inlet, 'inlet_valve_opening': valve,
    }


# ============================================================
# 故障定义表
# ============================================================
FAULT_DEFS = {
    # fault_type: (affected_channels, causal_chain_template, description_template)
    "bearing_overheat_drive": (
        ["bearing_temp_drive", "winding_temp_1", "winding_temp_2", "motor_current"],
        ["驱动侧轴承润滑油膜破坏", "摩擦系数增大", "轴承温度渐进式上升", "热传导至绕组", "电机负载增加"],
        "磨煤机驱动侧轴承过热故障"),
    "bearing_overheat_nondrive": (
        ["bearing_temp_nondrive", "winding_temp_1", "winding_temp_2"],
        ["非驱动侧轴承磨损", "摩擦增大", "轴承温度上升", "热传导至绕组"],
        "磨煤机非驱动侧轴承过热故障"),
    "gradual_degradation": (
        ["motor_current", "bearing_temp_drive", "winding_temp_1"],
        ["磨辊表面磨损", "研磨效率降低", "电流缓慢爬升", "机械磨损导致轴承温升"],
        "磨煤机渐进性能退化"),
    "roller_jam": (
        ["motor_current", "winding_temp_1", "winding_temp_2"],
        ["磨辊/磨盘卡涩", "研磨阻力骤增", "电流急剧上升", "I²R损耗导致温度快速升高"],
        "磨煤机磨辊卡涩故障"),
    "drive_looseness": (
        ["motor_current"],
        ["传动皮带/联轴器松动", "负荷传递不均", "电流出现周期性振荡"],
        "磨煤机传动系统松动"),
    "feeder_fault": (
        ["motor_current"],
        ["给煤机给煤量不均", "磨煤机负荷大幅波动", "电流出现大幅不规则振荡"],
        "给煤机故障导致负荷波动"),
    "cooling_failure": (
        ["winding_temp_1", "winding_temp_2"],
        ["冷却风扇转速下降或积灰", "散热能力降低", "绕组温度异常升高", "轴承和电流不变"],
        "磨煤机冷却系统故障"),
    "inlet_blockage": (
        ["inlet_temp", "inlet_valve_opening", "motor_current"],
        ["入口管道煤粉沉积结渣", "通风面积减小", "入口温度异常升高", "控制系统增大阀门补偿"],
        "磨煤机入口堵塞"),
    "outlet_oscillation": (
        ["inlet_temp"],
        ["出口温度控制异常", "控制系统PID参数不当", "入口温度出现周期性振荡"],
        "磨煤机出口温度控制振荡"),
    "duct_leak": (
        ["inlet_temp", "inlet_valve_opening"],
        ["热风管道出现泄漏", "风量损失", "入口温度下降", "阀门大开补偿"],
        "热风管道泄漏"),
    "current_spike": (
        ["motor_current", "winding_temp_1", "winding_temp_2"],
        ["大块煤或异物进入磨盘", "瞬时研磨阻力冲击", "电流出现随机尖峰", "温度因I²R短暂升高"],
        "磨煤机电流冲击"),
    "winding_short": (
        ["winding_temp_1", "winding_temp_2"],
        ["绕组局部匝间短路", "短路回路产生额外热量", "局部温度异常升高", "电流和负荷不变"],
        "电机绕组匝间短路"),
    "voltage_fluctuation": (
        ["motor_current"],
        ["供电电压波动", "电机运行不稳定", "电流出现高频波动"],
        "供电电压波动"),
    "overload": (
        ["motor_current", "winding_temp_1", "winding_temp_2", "bearing_temp_drive"],
        ["长时间超额定负荷运行", "电流持续偏高", "所有温度通道升高", "绝缘加速老化"],
        "磨煤机超负荷运行"),
    "load_surge": (
        ["motor_current", "winding_temp_1", "inlet_temp"],
        ["电网调度负荷突变", "电流阶跃变化", "温度滞后响应"],
        "磨煤机负荷突变"),
}


# ============================================================
# 统一生成接口
# ============================================================
def generate_coal_mill(total_length: int = 5000, seed: int = 42,
                        fault_type: str = "normal",
                        severity: str = "moderate",
                        load_mode: str = "steady",
                        randomize_fault_window: bool = False,
                        fault_start_range: Tuple[float, float] = (0.10, 0.80),
                        fault_duration_range: Tuple[float, float] = (0.08, 0.25),
                        severity_jitter: float = 0.0,
                        ) -> Tuple[pd.DataFrame, GenerationMetadata]:
    """
    生成磨煤机数据

    fault_type: "normal" 或 FAULT_DEFS 中的任意 key
    load_mode: "steady", "varying", "frequent_startstop"

    训练数据扩展参数（默认关闭，不影响 benchmark）：
      randomize_fault_window: 是否随机化故障起止位置
      fault_start_range: 故障起始位置占总长度的比例范围
      fault_duration_range: 故障持续时间占总长度的比例范围
      severity_jitter: 严重度因子抖动幅度 (0=不抖动, 0.2=±20%)
    """
    np.random.seed(seed)

    start_date = pd.Timestamp('2024-02-19')
    end_date = start_date + pd.Timedelta(seconds=SAMPLING_SECONDS * total_length)

    load_fn = {"steady": _load_steady, "varying": _load_varying,
               "frequent_startstop": _load_frequent_startstop}.get(load_mode, _load_steady)

    # --- 故障窗口计算 ---
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

    # --- 严重度因子 ---
    base_sev_factor = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}.get(severity, 0.6)
    if severity_jitter > 0:
        _rng_sev = np.random.RandomState(seed + 20013)
        sev_factor = base_sev_factor * (1 + _rng_sev.uniform(-severity_jitter, severity_jitter))
    else:
        sev_factor = base_sev_factor

    events_meta = []

    if fault_type == "normal":
        model, _ = _build_base_model(load_fn)
    elif fault_type == "frequent_startstop":
        model, _ = _build_base_model(_load_frequent_startstop)
        events_meta.append(EventMetadata(
            event_type="anomaly", fault_name="frequent_startstop",
            time_range=(0, total_length), severity=severity,
            parameters={"avg_shutdown_interval": "300-800 steps"},
            affected_channels=["motor_current", "inlet_temp", "inlet_valve_opening"],
            causal_chain=["控制系统异常", "启停频率远高于正常", "所有通道频繁波动"],
            description="磨煤机频繁启停异常"))
    elif fault_type in FAULT_DEFS:
        model = _inject_fault(load_fn, fault_type, fault_start, fault_end, severity, {})
        fdef = FAULT_DEFS[fault_type]
        sev_desc = {"mild": "轻微", "moderate": "中等", "severe": "严重"}[severity]
        events_meta.append(EventMetadata(
            event_type="fault", fault_name=fault_type,
            time_range=(fault_start, fault_end), severity=severity,
            parameters={
                "severity_factor": round(sev_factor, 3),
                "randomized": randomize_fault_window,
            },
            affected_channels=fdef[0],
            causal_chain=fdef[1],
            description=f"{fdef[2]}（{sev_desc}级），第{fault_start}步至第{fault_end}步"))
    else:
        model, _ = _build_base_model(load_fn)

    gen = Generator(start_date=start_date, end_date=end_date, freq='30S')
    df = gen.generate(model, n=1)

    el = [e.time_range[1] - e.time_range[0] for e in events_meta]
    difficulty = compute_difficulty(total_length, el)
    nr = sum(el) / total_length if events_meta else 0.0

    scenario_id = f"coal_mill_{fault_type}_{severity}" if fault_type != "normal" else f"coal_mill_normal_{load_mode}"
    metadata = GenerationMetadata(
        scenario_id=scenario_id,
        scenario_name=f"磨煤机-{fault_type}-{severity}",
        domain=DOMAIN, domain_display=DOMAIN_DISPLAY,
        total_length=total_length,
        sampling_rate=SAMPLING_RATE, sampling_seconds=SAMPLING_SECONDS,
        channels=COAL_MILL_CHANNELS,
        base_description=f"磨煤机{load_mode}运行模式。",
        injected_events=events_meta,
        overall_description=events_meta[0].description if events_meta else f"磨煤机正常{load_mode}运行。",
        generation_seed=seed, difficulty=difficulty, needle_ratio=round(nr, 4),
    )
    return df, metadata


# ============================================================
# 场景注册
# ============================================================
def register_all_coal_mill_tb():
    """注册全部磨煤机 TimeBlender 场景"""
    # 正常工况 (3)
    for mode in ["steady", "varying", "frequent_startstop"]:
        sid = f"normal_{mode}"
        def gen(total_length=5000, seed=42, _m=mode, **kwargs):
            return generate_coal_mill(total_length, seed,
                                       "normal" if _m != "frequent_startstop" else "frequent_startstop",
                                       "moderate", _m, **kwargs)
        register_scenario(DOMAIN, sid, f"正常-{mode}", gen)

    # 15种故障 × 3严重度 (45)
    for ft in FAULT_DEFS:
        for sev in ["mild", "moderate", "severe"]:
            sid = f"{ft}_{sev}"
            def gen(total_length=5000, seed=42, _ft=ft, _sev=sev, **kwargs):
                return generate_coal_mill(total_length, seed, _ft, _sev, "steady", **kwargs)
            register_scenario(DOMAIN, sid, f"{ft}-{sev}", gen)


register_all_coal_mill_tb()


# ============================================================
# 验证脚本
# ============================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== Coal Mill TB — Fault Verification ===")
    os.makedirs('Data/CoalMill_TB/fault_samples', exist_ok=True)

    # 验证每种故障的单样本
    test_faults = ["normal", "bearing_overheat_drive", "current_spike", "inlet_blockage",
                   "cooling_failure", "gradual_degradation", "roller_jam", "drive_looseness",
                   "feeder_fault", "outlet_oscillation", "duct_leak", "winding_short",
                   "voltage_fluctuation", "overload", "load_surge"]

    for ft in test_faults:
        sev = "moderate" if ft != "normal" else "moderate"
        df, meta = generate_coal_mill(5000, 42, ft, sev, "steady")
        print(f"  {ft}: shape={df.shape}, events={len(meta.injected_events)}")

        # 保存图
        fig, axes = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for idx, col in enumerate(df.columns):
            axes[idx].plot(df[col].values, color=colors[idx], linewidth=0.6)
            axes[idx].set_ylabel(col, fontsize=8)
            axes[idx].grid(True, alpha=0.2)
            for ev in meta.injected_events:
                if col in ev.affected_channels:
                    axes[idx].axvspan(ev.time_range[0], ev.time_range[1], alpha=0.15, color='red')
        axes[0].set_title(f"Coal Mill: {ft} ({sev})", fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'Data/CoalMill_TB/fault_samples/{ft}_{sev}.png', dpi=100)
        plt.close()

    print(f"\nAll {len(test_faults)} fault types verified. Check Data/CoalMill_TB/fault_samples/")
    print(f"Total registered scenarios: {len(get_all_scenarios(DOMAIN))}")
