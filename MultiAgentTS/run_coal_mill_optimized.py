"""
MultiAgentTS-based Coal Mill Operation Data Generation - OPTIMIZED VERSION
Generate realistic coal mill data that matches actual operational patterns
Based on analysis of real coal mill sensor data
"""
from Agents.code_generator import CodeGenerator
from Agents.Reflector import Reflector
from Agents.expert_agent import ExpertAgent
from Analysis_pipeline.analysis_agent import AnalysisAgent
import time
from config import config
from shared_utils import initialize_components, write_log, run_generation_loop

# Initialize components
expert_agent, code_generator, reflector, analysis_expert = initialize_components()

# OPTIMIZED Coal Mill Dataset Description - Based on Real Data Analysis
CoalMill_Des = """
This dataset contains operational data from a coal mill system used in power generation.
The data includes 7 key operational parameters measured at 30-second intervals.

**Data Fields** (MUST match exactly):
1. timestamp - Recording timestamp (format: YYYY-MM-DD HH:MM:SS)
2. 电机电流 (Motor Current) - Range: 0-100 A, Normal: 40-80 A, Fault: <10 A
3. 电机绕组温度1 (Motor Winding Temperature 1) - Range: 20-70°C, Normal: 40-60°C
4. 电机绕组温度2 (Motor Winding Temperature 2) - Range: 20-70°C, Normal: 40-60°C (highly correlated with 温度1)
5. 电机非驱动侧轴承温度 (Motor Non-Drive Side Bearing Temperature) - Range: 15-45 kPa, Normal: 23-28 kPa
6. 电机驱动侧轴承温度 (Motor Drive Side Bearing Temperature) - Range: 10-30 t/h, Normal: 14-20 t/h
7. 磨机入口温度 (Mill Inlet Temperature) - Range: -2 to 3 kPa, Normal: 0.8-1.0 kPa, Fault: <-1 kPa
8. 磨机入口调节阀开度 (Mill Inlet Valve Opening) - Range: 0-100%, Normal: 100%, Fault: 0-50%

**CRITICAL OPERATIONAL PATTERNS** (from real data analysis):

1. **Normal Operation Mode** (80-90% of time):
   - 电机电流: Stable at 40-80 A with small fluctuations (±5-10 A)
   - 电机绕组温度1/2: 40-60°C, slowly varying, HIGHLY CORRELATED (difference <2°C)
   - 电机非驱动侧轴承温度: 23-28 kPa with gradual trends
   - 电机驱动侧轴承温度: 14-20 t/h with moderate fluctuations
   - 磨机入口温度: 0.8-1.0 kPa, relatively stable
   - 磨机入口调节阀开度: 100% (fully open)

2. **Fault/Shutdown Events** (10-20% of time):
   - 电机电流: SUDDEN DROP to <10 A (near zero)
   - 电机绕组温度1/2: RAPID DECREASE following current drop
   - 磨机入口温度: SUDDEN DROP to negative values (<-1 kPa)
   - 磨机入口调节阀开度: SUDDEN DROP to 0-50%
   - Duration: 30 minutes to several hours
   - Recovery: Gradual return to normal values

3. **Sensor Correlations** (MUST preserve):
   - 电机绕组温度1 and 温度2: Nearly identical (correlation >0.95)
   - 电机电流 drop → immediate 阀开度 drop → delayed 温度 drop
   - All fault indicators occur SIMULTANEOUSLY

4. **Temporal Characteristics**:
   - Temperature changes: SLOW (thermal inertia, time constant ~10-30 minutes)
   - Current/Valve changes: FAST (near-instantaneous during faults)
   - Pressure changes: MODERATE (time constant ~5-10 minutes)
   - Normal fluctuations: Small amplitude, high frequency noise
   - Trend changes: Gradual over hours, not abrupt
"""

# OPTIMIZED Generation Objective
base_obj = """
**OBJECTIVE: Generate Realistic Coal Mill Data with Fault Events**

**CRITICAL REQUIREMENTS**:

1. **Fault Event Generation** (HIGHEST PRIORITY):
   - Include 5-10 fault events over 30 days
   - Each fault event MUST have:
     * SIMULTANEOUS drops in: 电机电流 (<10 A), 磨机入口温度 (<-1 kPa), 磨机入口调节阀开度 (<50%)
     * DELAYED drops in: 电机绕组温度1/2 (following thermal lag)
     * Duration: 30 min to 6 hours (random)
     * Recovery: Gradual (not instantaneous)

2. **Sensor Correlation** (MUST PRESERVE):
   - 电机绕组温度1 and 温度2: ALWAYS within 2°C of each other
   - Use: 温度2 = 温度1 + random_noise(-1, 1)
   - Fault indicators MUST occur together (within 1-2 time steps)

3. **Realistic Dynamics**:
   - Temperature changes: Use exponential smoothing or low-pass filter (time constant 10-30 min)
   - Current/Valve: Can change rapidly during faults, stable otherwise
   - Add realistic sensor noise: Gaussian noise with small amplitude
   - Avoid perfect sine waves or overly regular patterns

4. **Normal Operation Patterns**:
   - 电机电流: 40-80 A, add random walk + noise
   - 电机绕组温度1/2: 40-60°C, slow variations
   - 电机非驱动侧轴承温度: 23-28 kPa, gradual trends
   - 电机驱动侧轴承温度: 14-20 t/h, moderate fluctuations
   - 磨机入口温度: 0.8-1.0 kPa (normal), <-1 kPa (fault)
   - 磨机入口调节阀开度: 100% (normal), 0-50% (fault)

5. **Data Quality**:
   - Exactly 30 days at 30-second intervals (86,400 rows)
   - Timestamp format: 'YYYY-MM-DD HH:MM:SS'
   - No NaN or missing values
   - All values within physically realistic bounds

**IMPLEMENTATION STRATEGY**:
1. Generate base normal operation signals with slow variations
2. Add realistic sensor noise
3. Insert fault events at random times
4. Apply thermal lag to temperature responses
5. Ensure 温度1 and 温度2 are nearly identical
6. Verify fault indicators are synchronized

**OUTPUT**:
- Save to specified CSV path
- Column order MUST match: timestamp, 电机电流, 电机绕组温度1, 电机绕组温度2, 电机非驱动侧轴承温度, 电机驱动侧轴承温度, 磨机入口温度, 磨机入口调节阀开度
"""

# Run generation loop
print("="*80)
print("Starting OPTIMIZED Coal Mill Data Generation")
print("="*80)

run_generation_loop(
    code_generator=code_generator,
    reflector=reflector,
    analysis_expert=analysis_expert,
    data_description=CoalMill_Des,
    base_objective=base_obj,
    dataset_path="aligned_output/merged.csv",
    output_data_path="Data/CoalMill/synthetic_coal_mill_optimized.csv",
    data_caption="Optimized Coal Mill Data with Realistic Fault Events",
    max_iterations=15,
    validate_similarity=False,
    expert_agent=expert_agent,
    use_expert_expansion=True
)

print("="*80)
print("Optimized Coal Mill Data Generation Complete!")
print("Output: Data/CoalMill/synthetic_coal_mill_optimized.csv")
print("="*80)

