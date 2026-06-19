from Code_Generator import *
from Analysis_Agent import *
import time
from config import config
from shared_utils import initialize_components, write_log, run_generation_loop

# Initialize components
expert_agent, code_generator, reflector, analysis_expert = initialize_components()

# User input task description - only basic description needed, ExpertAgent will auto-expand
POWER_des = """
Tetouan is a city located in the north of Morocco which occupies an area of around 10375 km² and its population is about 550.374 inhabitants, according to the last Census of 2014, and is increasing rapidly, approximately 1.96% annually. Since it is located along the Mediterranean Sea, its weather is mild and rainy in the winter, hot and dry during the summer months.

Morocco's per capita energy consumption is 0.56 toe (around 42% below the North Africa average), including around 900 kWh of electricity (38% below the regional average) (2020). The progression of total energy consumption slowed down between 2010 and 2019 (+3 %/year, compared to 4.5%/year over 2000-2010) and decreased by 7% in 2020 to around 21 Mtoe.

The national production of hydrocarbons is low. All the oil products are imported since the shutdown of the sole refinery of the country in 2015 (200 000 bbl/d). Oil product imports increased rapidly from 2015 to 2019 (+6%/year) and decreased by 12% in 2020 due to the COVID crisis. The power consumption data was collected from Supervisory Control and Data Acquisition System (SCADA) of Amendis which is a public service operator and in charge of the distribution of drinking water and electricity since 2002. The purpose of the electricity distribution network is to serve low and medium voltage consumers in Tetouan regions. For this purpose, the delivery and distribution of electrical energy from the point of delivery to the end user, the customer, is ensured by Amendis. The energy which is distributed comes from the National Office of Electricity and Drinking Water. After transforming the high voltage (63 kV) to medium voltage (20 kV), it is allowed to transport and distribute the energy.

With the electricity consumption being so crucial to the country, the idea is to study the impact on energy consumption. The dataset is exhaustive in its demonstration of energy consumption of the Tétouan city in Morocco. The distribution network is powered by Quads station.

The data consists of some observations of energy consumption on a 10-minute window.
The Dataset have 6 features.
Temperature: Weather Temperature.
Humidity: Weather Humidity.
Wind Speed: Wind Speed.
General Diffuse Flows: "Diffuse flow" is a catchall term to describe low-temperature (< 0.2° to ~ 100°C) fluids that slowly discharge through sulfide mounds, fractured lava flows, and assemblages of bacterial mats and macrofauna.
Diffuse Flows.
Quads Station Power Consumption.
"""

base_obj = """
**Improving the diversity of data**
***you need to consider how the weather-related features should change in combination with geographical factors. 
Quads Station Power Consumption should also have obvious peaks and valleys every day, 
and you need to use the timeblender library code to reflect it as much as possible. 
For example, you can use clipevent and top(bottom)_resistance to simulate the intraday pattern well, 
which requires you to carefully study the library code and the example jupyter file.***
You should consider all Possible event about the data and Show it in code.
You need to simulate factors such as weather, industrial events, etc.
You should try your best to simulate the real events.
**You need to reflect the relationship between variables in the code, which means the relationship between CO2 and other variables**.
We also need to ensure the rationality of the numerical values.
**While ensuring simulation, the occurrence of random events is also reflected in the data.**
For example you can add some random event to influence data trends, and these random event is real-life events you simulated.
**In the code, You need to achieve the finest granularity, with days or hours as the basic unit, to control the data generation results.**
**You need to pay attention to which variables have the same trend and which variables influence each other.**
**Now, I need you generate 3 month data for me.**
**
"""

# Run generation loop
run_generation_loop(
    code_generator=code_generator,
    reflector=reflector,
    analysis_expert=analysis_expert,
    data_description=POWER_des,
    base_objective=base_obj,
    dataset_path="Dataset/POWER/power.csv",
    output_data_path="Data/POWER/synthetic_data.csv",
    data_caption=POWER_des + " OT is Quads Station Power Consumption.",
    max_iterations=10,
    validate_similarity=False,  # Set to False, do not perform similarity comparison
    expert_agent=expert_agent,  # Pass ExpertAgent
    use_expert_expansion=True  # Enable expert expansion
)
