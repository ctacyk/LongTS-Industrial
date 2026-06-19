from Code_Generator import *
from Analysis_Agent import *
import time
from config import config
from shared_utils import initialize_components, write_log, run_generation_loop

# Initialize components
expert_agent, code_generator, reflector, analysis_expert = initialize_components()

# User input task description - only basic description needed, ExpertAgent will auto-expand
ETTh1_Des = """The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.
This dataset consists of 2 years data from two separated counties in China.
ETTh1 for 1-hour-level.
Each data point consists of the target value "oil temperature(OT)" and 6 power load features(HUFL,HULL,MUFL,MULL,LUFL,LULL).
Field	date	HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
Description	The recorded date	High UseFul Load	High UseLess Load	Middle UseFul Load	Middle UseLess Load	Low UseFul Load	Low UseLess Load	Oil Temperature (target)
"""

base_obj = """
**Improving the diversity of data**
**You need to read the example jupyter document carefully and learn how to use various events and models.**
**Imitate its control methods for cycles, trends, and random events.And flexibly apply it to current data requirements**
You should consider all Possible event about the data and Show it in code.
You need to simulate factors such as holidays, weather, industrial events, etc.
You should try your best to simulate the real events.
The data should have a certain periodicity in general, but the performance of the data should not have a very obvious period.
For example, the performance like a sine function is obviously not in line with the reality.
You need to reflect the relationship between variables in the code.
We also need to ensure the rationality of the numerical values.
Do not simply use simple events, as the generated data is too easily fitted by neural networks.
**While ensuring simulation, the occurrence of random events is also reflected in the data.**
For example you can add some random event to influence data trends, and these random event is real-life events you simulated.
Note the difference of the data between month and day.
**In the code, You need to achieve the finest granularity, with days or hours as the basic unit, to control the data generation results.**
Now, I need you generate 1 year ETT data for me.
"""

# Run generation loop
run_generation_loop(
    code_generator=code_generator,
    reflector=reflector,
    analysis_expert=analysis_expert,
    data_description=ETTh1_Des,
    base_objective=base_obj,
    dataset_path="Dataset/ETT/ETTh1.csv",
    output_data_path="Data/ETT/synthetic_data.csv",
    max_iterations=10,
    validate_similarity=False,
    expert_agent=expert_agent,  # Pass ExpertAgent
    use_expert_expansion=True  # Enable expert expansion
)
