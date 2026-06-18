import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta

# Import time_blender components
# Assuming the library is installed or available in the python path as per the provided context
from time_blender.core import Generator, Event, LambdaEvent, ConstantEvent
from time_blender.random_events import NormalEvent, UniformEvent
from time_blender.deterministic_events import WaveEvent, ClipEvent

# -------------------------------------------------------------------------
# Custom Event Logic for Physical Modeling
# -------------------------------------------------------------------------

def thermal_inertia_logic(t, i, memory, sub_events):
    """
    Simulates thermal lag (Newton's Law of Cooling / First Order System).
    New_Temp = Old_Temp + Alpha * (Target_Temp - Old_Temp)
    """
    # Get inputs
    target = sub_events['target'].execute(t)
    alpha = sub_events['alpha'].execute(t)
    
    # Initialize memory
    if 'prev_temp' not in memory:
        memory['prev_temp'] = target # Start at steady state
        
    # Calculate physics
    prev = memory['prev_temp']
    # Add a small randomness to the cooling/heating rate
    noise = np.random.normal(0, 0.05)
    new_temp = prev + (alpha * (target - prev)) + noise
    
    memory['prev_temp'] = new_temp
    return new_temp

def operational_state_logic(t, i, memory, sub_events):
    """
    Simulates the mill's operational state machine.
    0 = Stopped
    1 = Normal Run
    1.2 = High Load / Stress
    """
    if 'state' not in memory:
        memory['state'] = 1.0
        memory['timer'] = 0
        memory['next_change'] = 2880 * 2  # Run for ~2 days initially (30s steps)

    # Stress Event Injection (Day 24-26 approx)
    # 30 days * 24h * 120 steps/h = 86400 steps
    if 69000 < i < 75000: 
        return 1.25 + np.random.normal(0, 0.05) # High load stress

    # Normal State Machine
    memory['timer'] += 1
    
    if memory['timer'] >= memory['next_change']:
        memory['timer'] = 0
        current_state = memory['state']
        
        if current_state > 0.1: # If running, switch to stop
            memory['state'] = 0.0
            # Stop duration: 2 to 6 hours (240 to 720 steps)
            memory['next_change'] = int(np.random.uniform(240, 720))
        else: # If stopped, switch to run
            memory['state'] = 1.0
            # Run duration: 1 to 4 days (2880 to 11520 steps)
            memory['next_change'] = int(np.random.uniform(2880, 11520))
            
    return memory['state']

def valve_logic(t, i, memory, sub_events):
    """
    Valve is binary: 100% when running, 0% when stopped.
    Adds quantization noise.
    """
    state = sub_events['state'].execute(t)
    
    if state > 0.1:
        # Open (99-101% due to sensor noise)
        base = 100.0
        noise = np.random.normal(0, 0.2)
    else:
        # Closed (-0.5 to 0.5% noise)
        base = 0.0
        noise = np.random.normal(0, 0.1)
        
    return base + noise

# -------------------------------------------------------------------------
# Model Construction
# -------------------------------------------------------------------------

def build_coal_mill_model():
    # 1. Global Time & Ambient Conditions
    # -----------------------------------
    # We will use a Lambda for Ambient to ensure correct 24h cycle based on timestamp
    def ambient_func(t, i, mem, sub):
        hour = t.hour + t.minute/60.0
        # Diurnal cycle: Coldest at 4AM, Hottest at 4PM
        diurnal = -5.0 * np.cos((hour - 4) * 2 * np.pi / 24)
        # Seasonal trend: Warming up by 10 degrees over the month
        seasonal = (i / 86400.0) * 10.0 
        return 15.0 + diurnal + seasonal + np.random.normal(0, 0.5)

    ambient_temp = LambdaEvent(ambient_func)

    # 2. Operational State (The Driver)
    # ---------------------------------
    op_state = LambdaEvent(operational_state_logic)

    # 3. Motor Current
    # ----------------
    # Base 50A * State + Noise
    def current_func(t, i, mem, sub):
        state = sub['state'].execute(t)
        if state < 0.1:
            return np.random.normal(0, 0.2) # Zero noise
        else:
            # Base load 50A * state factor (allows for overload)
            base = 50.0 * state 
            # Grinding noise is high freq
            noise = np.random.normal(0, 1.5)
            # Occasional coal chunk spikes
            spike = 0 if np.random.random() > 0.001 else np.random.uniform(10, 20)
            return base + noise + spike

    motor_current = LambdaEvent(current_func, sub_events={'state': op_state})

    # 4. Valve Opening
    # ----------------
    valve_opening = LambdaEvent(valve_logic, sub_events={'state': op_state})

    # 5. Inlet Temperature
    # --------------------
    # Target: 300C if Valve Open, Ambient if Valve Closed.
    # Fast Lag (Alpha 0.1)
    def inlet_target_func(t, i, mem, sub):
        v = sub['valve'].execute(t)
        amb = sub['ambient'].execute(t)
        if v > 50:
            return 290.0 + np.random.normal(0, 2.0) # Hot air setpoint
        else:
            return amb # Cools to ambient

    inlet_target = LambdaEvent(inlet_target_func, sub_events={'valve': valve_opening, 'ambient': ambient_temp})
    
    inlet_temp = LambdaEvent(
        thermal_inertia_logic,
        sub_events={
            'target': inlet_target,
            'alpha': ConstantEvent(0.05) # Fast response (gas)
        }
    )

    # 6. Winding Temperatures (1 & 2)
    # -------------------------------
    # Heat Source = Ambient + k * Current^2
    def winding_target_func(t, i, mem, sub):
        amb = sub['ambient'].execute(t)
        curr = sub['current'].execute(t)
        # Heating coefficient logic
        # 50A -> +30C rise over ambient -> 50^2 * k = 30 -> k = 0.012
        heat_rise = (curr ** 2) * 0.014 
        return amb + heat_rise

    winding_target = LambdaEvent(winding_target_func, sub_events={'ambient': ambient_temp, 'current': motor_current})

    # Winding 1
    winding_temp_1 = LambdaEvent(
        thermal_inertia_logic,
        sub_events={
            'target': winding_target,
            'alpha': ConstantEvent(0.008) # Medium thermal mass
        }
    )

    # Winding 2 (Correlated but slightly different noise/bias)
    def winding_2_func(t, i, mem, sub):
        w1 = sub['w1'].execute(t)
        # Slight drift/offset
        return w1 - 0.5 + np.random.normal(0, 0.1)

    winding_temp_2 = LambdaEvent(winding_2_func, sub_events={'w1': winding_temp_1})

    # 7. Bearing Temperatures
    # -----------------------
    # Heat Source = Ambient + k * Current (Friction is roughly linear with load)
    # Drive Side runs hotter.
    
    def bearing_target_ds_func(t, i, mem, sub):
        amb = sub['ambient'].execute(t)
        curr = sub['current'].execute(t)
        # 50A -> +20C rise -> k = 0.4
        return amb + (curr * 0.5) + 5.0 # Bias for drive side

    def bearing_target_nds_func(t, i, mem, sub):
        amb = sub['ambient'].execute(t)
        curr = sub['current'].execute(t)
        return amb + (curr * 0.3) # Cooler

    bearing_target_ds = LambdaEvent(bearing_target_ds_func, sub_events={'ambient': ambient_temp, 'current': motor_current})
    bearing_target_nds = LambdaEvent(bearing_target_nds_func, sub_events={'ambient': ambient_temp, 'current': motor_current})

    # Heavy Lag for bearings (Metal housing)
    bearing_temp_ds = LambdaEvent(
        thermal_inertia_logic,
        sub_events={'target': bearing_target_ds, 'alpha': ConstantEvent(0.0015)} # Very slow
    )
    
    bearing_temp_nds = LambdaEvent(
        thermal_inertia_logic,
        sub_events={'target': bearing_target_nds, 'alpha': ConstantEvent(0.0015)}
    )

    return {
        '电机电流': motor_current,
        '电机绕组温度1': winding_temp_1,
        '电机绕组温度2': winding_temp_2,
        '电机非驱动侧轴承温度': bearing_temp_nds,
        '电机驱动侧轴承温度': bearing_temp_ds,
        '磨机入口温度': inlet_temp,
        '磨机入口调节阀开度': valve_opening
    }

# -------------------------------------------------------------------------
# Execution & Saving
# -------------------------------------------------------------------------

def generate_data():
    # Configuration
    start_date = pd.Timestamp('2024-02-19 00:00:00')
    end_date = start_date + timedelta(days=30)
    
    # Build Model
    events_dict = build_coal_mill_model()
    
    # Initialize Generator
    gen = Generator(
        start_date=start_date,
        end_date=end_date,
        freq='30S'
    )
    
    print("Generating 30 days of synthetic coal mill data...")
    
    # Generate
    # When n=1, time_blender returns the DataFrame directly, not a list
    df = gen.generate(events_dict, n=1)
    
    # Post-processing to match exact column order and types
    column_order = [
        '电机电流', 
        '电机绕组温度1', 
        '电机绕组温度2', 
        '电机非驱动侧轴承温度', 
        '电机驱动侧轴承温度', 
        '磨机入口温度', 
        '磨机入口调节阀开度'
    ]
    
    # Ensure all columns exist and are ordered
    df = df[column_order]
    
    # Set index name
    df.index.name = 'Timestamp'
    
    # Output Path
    output_dir = 'Data/CoalMill'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/synthetic_coal_mill_30days.csv'
    
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)
    generate_data()