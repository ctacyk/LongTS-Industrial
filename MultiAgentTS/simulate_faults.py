import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta

# Import time_blender components
from time_blender.core import Generator, LambdaEvent, ConstantEvent

# -------------------------------------------------------------------------
# Fault Definitions
# -------------------------------------------------------------------------
# Each fault will be simulated in a separate 4-hour window.
# Fault active window: 90 min to 150 min (1.5h to 2.5h)
FAULT_START_MIN = 90
FAULT_END_MIN = 150

FAULTS = [
    {'id': 1, 'name': 'Coal Blockage (堵煤)',     'desc': 'Current Rises'},
    {'id': 2, 'name': 'Coal Moisture High (煤质变湿)', 'desc': 'Current Up, Temp Down'},
    {'id': 3, 'name': 'Grinding Roller Wear (磨辊磨损)',  'desc': 'Current Down, Noise Up'},
    {'id': 4, 'name': 'Foreign Object (异物进入)',   'desc': 'Current Spikes'},
    {'id': 5, 'name': 'Winding Overheat (绕组过热)', 'desc': 'Winding 1 Rises'},
    {'id': 6, 'name': 'Bearing Fault (轴承故障)', 'desc': 'DS Bearing Rises'},
    {'id': 7, 'name': 'Feeder Instability (给煤机波动)', 'desc': 'Current Oscillates'},
]

# -------------------------------------------------------------------------
# Physical Logic
# -------------------------------------------------------------------------

def thermal_inertia_logic(t, i, memory, sub_events):
    target = sub_events['target'].execute(t)
    alpha = sub_events['alpha'].execute(t)
    
    if 'prev_temp' not in memory:
        memory['prev_temp'] = target
        
    prev = memory['prev_temp']
    noise = np.random.normal(0, 0.05)
    new_temp = prev + (alpha * (target - prev)) + noise
    
    memory['prev_temp'] = new_temp
    return new_temp

def fault_injector_logic(t, i, memory, sub_events):
    # Determine if fault is active based on time
    # t is Timestamp
    # We assume simulation starts at 00:00
    minute = (t.hour * 60) + t.minute + (t.second / 60.0)
    
    target_fault_id = sub_events['target_id'].execute(t)
    
    if FAULT_START_MIN <= minute < FAULT_END_MIN:
        return float(target_fault_id)
    else:
        return 0.0

# -------------------------------------------------------------------------
# Model Construction
# -------------------------------------------------------------------------

def build_model(target_fault_id):
    # Fault Signal
    fault_signal = LambdaEvent(
        fault_injector_logic, 
        sub_events={'target_id': ConstantEvent(float(target_fault_id))}
    )

    # 1. Ambient
    ambient_temp = ConstantEvent(20.0)

    # 2. Motor Current
    def current_func(t, i, mem, sub):
        fid = sub['fault'].execute(t)
        base = 50.0
        noise = np.random.normal(0, 1.0)
        
        offset = 0.0
        
        if fid == 1.0: # Blockage
            # Ramp up
            minute = (t.hour * 60) + t.minute
            elapsed = minute - FAULT_START_MIN
            offset = min(30.0, elapsed * 1.0) 
            noise *= 1.5
            
        elif fid == 2.0: # Moisture High
            # Harder to grind -> Current Up
            offset = 8.0
            # Steam pressure -> slightly more noise
            noise *= 1.2
            
        elif fid == 3.0: # Roller Wear
            # Less grinding efficiency/force -> Current Down slightly
            offset = -5.0
            # Vibration -> High Noise
            noise = np.random.normal(0, 4.0)
            
        elif fid == 4.0: # Foreign Object
            # Random massive spikes
            if np.random.random() > 0.95: # 5% chance per step (10s)
                spike = np.random.uniform(15, 30)
                offset = spike
            else:
                offset = 0
                
        elif fid == 7.0: # Feeder Instability
            # Surging / Oscillation
            # Period approx 2 mins = 12 steps
            osc = np.sin(i / 2.0) * 10.0
            offset = osc
            
        return base + offset + noise

    motor_current = LambdaEvent(current_func, sub_events={'fault': fault_signal})

    # 3. Valve Opening
    def valve_func(t, i, mem, sub):
        fid = sub['fault'].execute(t)
        base = 85.0
        mod = np.sin(i / 100.0) * 5.0 
        noise = np.random.normal(0, 0.5)
        
        offset = 0.0
        if fid == 2.0: # Moisture
            # Control system tries to dry coal -> Open valve more
            offset = 10.0
            
        return base + offset + mod + noise

    valve_opening = LambdaEvent(valve_func, sub_events={'fault': fault_signal})

    # 4. Inlet Temperature
    def inlet_target_func(t, i, mem, sub):
        fid = sub['fault'].execute(t)
        
        if fid == 2.0: # Moisture High
            # Evaporation absorbs heat -> Temp Drops
            # Even with valve open, the latent heat load is high
            return 270.0 
        else:
            return 300.0

    inlet_target = LambdaEvent(inlet_target_func, sub_events={'fault': fault_signal})
    
    inlet_temp = LambdaEvent(
        thermal_inertia_logic,
        sub_events={'target': inlet_target, 'alpha': ConstantEvent(0.1)}
    )

    # 5. Winding Temperatures
    def winding_target_func(t, i, mem, sub):
        fid = sub['fault'].execute(t)
        curr = sub['current'].execute(t)
        amb = 20.0
        heat_rise = (curr ** 2) * 0.014
        
        if fid == 5.0: # Overheat
            return amb + heat_rise + 40.0
        else:
            return amb + heat_rise

    winding_target = LambdaEvent(winding_target_func, sub_events={'fault': fault_signal, 'current': motor_current})

    winding_temp_1 = LambdaEvent(
        thermal_inertia_logic,
        sub_events={'target': winding_target, 'alpha': ConstantEvent(0.01)}
    )
    
    def winding_2_logic(t, i, mem, sub):
        w1 = sub['w1'].execute(t)
        fid = sub['fault'].execute(t)
        
        if fid == 5.0:
            # Decouple
            curr = sub['current'].execute(t)
            normal_rise = (curr ** 2) * 0.014 + 20.0
            if 'w2_lag' not in mem: mem['w2_lag'] = normal_rise
            mem['w2_lag'] += 0.01 * (normal_rise - mem['w2_lag'])
            return mem['w2_lag']
        else:
            return w1 - 2.0 + np.random.normal(0, 0.1)

    winding_temp_2 = LambdaEvent(winding_2_logic, sub_events={'w1': winding_temp_1, 'fault': fault_signal, 'current': motor_current})

    # 6. Bearing Temperatures
    def bearing_target_ds_func(t, i, mem, sub):
        fid = sub['fault'].execute(t)
        curr = sub['current'].execute(t)
        amb = 20.0
        base = amb + (curr * 0.5) + 5.0
        
        if fid == 6.0: # Bearing Fault
            return base + 50.0
        else:
            return base

    bearing_target_ds = LambdaEvent(bearing_target_ds_func, sub_events={'fault': fault_signal, 'current': motor_current})
    
    bearing_temp_ds = LambdaEvent(
        thermal_inertia_logic,
        sub_events={'target': bearing_target_ds, 'alpha': ConstantEvent(0.005)}
    )
    
    def bearing_target_nds_func(t, i, mem, sub):
        curr = sub['current'].execute(t)
        return 20.0 + (curr * 0.3)

    bearing_target_nds = LambdaEvent(bearing_target_nds_func, sub_events={'current': motor_current})
    bearing_temp_nds = LambdaEvent(
        thermal_inertia_logic,
        sub_events={'target': bearing_target_nds, 'alpha': ConstantEvent(0.005)}
    )

    return {
        'Fault_ID': fault_signal,
        '电机电流': motor_current,
        '电机绕组温度1': winding_temp_1,
        '电机绕组温度2': winding_temp_2,
        '电机非驱动侧轴承温度': bearing_temp_nds,
        '电机驱动侧轴承温度': bearing_temp_ds,
        '磨机入口温度': inlet_temp,
        '磨机入口调节阀开度': valve_opening
    }

# -------------------------------------------------------------------------
# Execution & Visualization
# -------------------------------------------------------------------------

def generate_comparison_plot(fault_info, df_normal, df_faulty, start_date):
    # Use a cleaner style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Setup figure with 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] # Ensure Chinese support
    plt.rcParams['axes.unicode_minus'] = False
    
    # Main Title
    fig.suptitle(f"故障模拟对比: {fault_info['name']}\nFault Simulation Comparison", fontsize=18, fontweight='bold', y=0.98)
    
    # Columns to plot (Removed Valve Opening)
    columns_map = {
        '电机电流': ('Motor Current', 'Amperes (A)'),
        '磨机入口温度': ('Inlet Temperature', 'Temp (°C)'),
        '电机绕组温度1': ('Winding Temp 1', 'Temp (°C)'),
        '电机绕组温度2': ('Winding Temp 2', 'Temp (°C)'),
        '电机驱动侧轴承温度': ('DS Bearing Temp', 'Temp (°C)'),
        '电机非驱动侧轴承温度': ('NDS Bearing Temp', 'Temp (°C)')
    }
    
    col_keys = list(columns_map.keys())
    
    # Fault Start/End Times
    f_start = start_date + timedelta(minutes=FAULT_START_MIN)
    f_end = start_date + timedelta(minutes=FAULT_END_MIN)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    for idx, col in enumerate(col_keys):
        ax = axes_flat[idx]
        title_en, unit = columns_map[col]
        
        # Plot Normal (Blue, dashed, lighter)
        ax.plot(df_normal.index, df_normal[col], color='#1f77b4', alpha=0.6, 
                label='正常状态 (Normal)', linestyle='--', linewidth=1.5)
        
        # Plot Faulty (Red, solid, prominent)
        ax.plot(df_faulty.index, df_faulty[col], color='#d62728', alpha=0.9, 
                label='故障状态 (Faulty)', linewidth=2.0)
        
        # Highlight Fault Zone
        ax.axvspan(f_start, f_end, color='#ff7f0e', alpha=0.15, label='故障区间 (Fault Zone)')
        
        # Aesthetics
        ax.set_title(f"{col}\n{title_en}", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Date formatting on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Legend only on the first plot (or shared)
        if idx == 0:
            ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = 'Data/CoalMill/Comparisons'
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Fault_{fault_info['id']}_Comparison.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close(fig)

def run_simulations():
    start_date = pd.Timestamp('2024-01-01 00:00:00')
    end_date = start_date + timedelta(hours=4) # 4 Hours per simulation
    
    # 1. Generate Normal Baseline (Fault ID = 0)
    print("Generating Normal Baseline...")
    gen_normal = Generator(start_date=start_date, end_date=end_date, freq='10S')
    events_normal = build_model(target_fault_id=0)
    df_normal = gen_normal.generate(events_normal, n=1)
    
    # 2. Loop through Faults
    for fault in FAULTS:
        print(f"Simulating Fault {fault['id']}: {fault['name']}...")
        
        # Generate Faulty Data
        gen_faulty = Generator(start_date=start_date, end_date=end_date, freq='10S')
        events_faulty = build_model(target_fault_id=fault['id'])
        df_faulty = gen_faulty.generate(events_faulty, n=1)
        
        # Generate Plot
        generate_comparison_plot(fault, df_normal, df_faulty, start_date)

if __name__ == "__main__":
    np.random.seed(42)
    run_simulations()
