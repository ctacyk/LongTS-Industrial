import os
from analysis_agent import *
analysis_expert = AnalysisAgent(
    api_key=os.getenv("API_KEY", ""),
    base_url="https://xiaoai.plus/v1",
    model="gpt-4o",
)
response = analysis_expert.analysis_chat(image_path="../Data/synthetic_data_plot.webp")
print(response)