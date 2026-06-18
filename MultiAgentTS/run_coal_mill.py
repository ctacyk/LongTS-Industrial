"""
MultiAgentTS-based Coal Mill Operation Data Generation
Generate realistic 30-day simulated data for EM ball-ring coal mill
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

# Coal Mill Dataset Description
CoalMill_Des = """
This dataset contains operational data from an EM ball-ring coal mill (EM球环型磨煤机).
The data spans 30 days with measurements taken every 30 seconds.

**Equipment Type**: EM Ball-Ring Coal Mill (EM球环型磨煤机)
**Application**: Fault Diagnosis and Condition Monitoring
**Time Period**: 30 days of continuous operation
**Sampling Interval**: 30 seconds

**Data Fields**:
1. timestamp - Recording timestamp (format: YYYY-MM-DD HH:MM:SS)
2. 电机电流 (Motor Current) - Electric motor current in Amperes
3. 电机绕组温度1 (Motor Winding Temperature 1) - First motor winding temperature in °C
4. 电机绕组温度2 (Motor Winding Temperature 2) - Second motor winding temperature in °C
5. 电机非驱动侧轴承温度 (Motor Non-Drive Side Bearing Temperature) - Temperature of non-drive side bearing in °C
6. 电机驱动侧轴承温度 (Motor Drive Side Bearing Temperature) - Temperature of drive side bearing in °C
7. 磨机入口温度 (Mill Inlet Temperature) - Temperature at mill inlet in °C
8. 磨机入口调节阀开度 (Mill Inlet Valve Opening) - Valve opening percentage (0-100%)

**Operational Characteristics**:
- The coal mill operates continuously with varying load conditions
- Motor current varies with grinding load (typically 40-60A during normal operation)
- Motor winding temperatures are correlated and typically range from 30-50°C
- Bearing temperatures are influenced by motor load and ambient conditions
- Mill inlet temperature is controlled by the inlet valve opening
- The system exhibits daily operational patterns and load variations
- Temperature measurements show thermal inertia and lag effects
- Valve opening adjustments affect mill inlet temperature with time delays

**Physical Relationships**:
- Higher motor current indicates higher grinding load
- Motor winding temperatures increase with sustained high current
- Bearing temperatures correlate with motor load and operating duration
- Mill inlet temperature is inversely related to valve opening (more open = cooler)
- All temperature sensors show gradual changes due to thermal mass
"""

# Generation Objective
base_obj = """
**Objective: Generate Realistic 30-Day Coal Mill Operation Data**

**Key Requirements**:
1. **Preserve Physical Relationships**: 
   - Motor current and winding temperatures must be correlated
   - Bearing temperatures should follow motor load patterns with thermal lag
   - Mill inlet temperature should respond to valve opening changes with appropriate delays
   
2. **Temporal Patterns**:
   - Daily operational cycles (production schedules, shift changes)
   - Weekly patterns (weekday vs weekend operations)
   - Gradual temperature changes reflecting thermal inertia
   - Realistic transient responses to load changes

3. **Operational Realism**:
   - Normal operating ranges for all parameters
   - Realistic startup and shutdown sequences
   - Load variations throughout the day
   - Control system responses (valve adjustments)
   - Random fluctuations within normal bounds

4. **Data Quality**:
   - 30-second sampling interval (consistent with original data)
   - No unrealistic jumps or discontinuities
   - Sensor noise appropriate for industrial measurements
   - Maintain correlations between related parameters

5. **Diversity and Complexity**:
   - Simulate various operating conditions (low, medium, high load)
   - Include realistic operational events (load changes, valve adjustments)
   - Add appropriate random variations to avoid overly periodic patterns
   - Ensure data is suitable for fault diagnosis applications

**Important Notes**:
- Use hours or 30-second intervals as the basic time unit
- Avoid simple sine wave patterns - use realistic industrial process dynamics
- Consider thermal time constants (temperatures change slowly)
- Valve opening changes should precede temperature changes
- Motor current changes should lead temperature changes
- All parameters should stay within physically realistic bounds

**Output Requirements**:
- Generate exactly 30 days of data (86,400 data points at 30-second intervals)
- Maintain the same column structure as the reference data
- Ensure timestamp continuity and correct interval
- Save output to the specified path in CSV format
"""

# Run generation loop
print("="*80)
print("Starting Coal Mill Data Generation with MultiAgentTS Framework")
print("="*80)

run_generation_loop(
    code_generator=code_generator,
    reflector=reflector,
    analysis_expert=analysis_expert,
    data_description=CoalMill_Des,
    base_objective=base_obj,
    dataset_path="Dataset/CoalMill/merged.csv",
    output_data_path="Data/CoalMill/synthetic_coal_mill_30days.csv",
    data_caption="EM Ball-Ring Coal Mill 30-Day Operation Data",
    max_iterations=1,
    validate_similarity=False,  # Focus on realism rather than exact similarity
    expert_agent=expert_agent,
    use_expert_expansion=True
)

print("="*80)
print("Coal Mill Data Generation Complete!")
print("Output saved to: Data/CoalMill/synthetic_coal_mill_30days.csv")
print("="*80)

