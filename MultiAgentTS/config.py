import os
from typing import Optional

class Config:
    """Project configuration management class"""

    def __init__(self):
        # API Configuration
        self.api_key = os.getenv('API_KEY', '')
        self.base_url = os.getenv('BASE_URL', 'https://xiaoai.plus/v1')

        # Model Configuration
        self.code_generator_model = os.getenv('CODE_GENERATOR_MODEL', 'gemini-3-pro-preview')
        self.analysis_model = os.getenv('ANALYSIS_MODEL', 'gemini-3-pro-preview')
        self.reflector_model = os.getenv('REFLECTOR_MODEL', 'gemini-3-pro-preview')

        # Path Configuration
        self.time_blender_path = os.getenv('TIME_BLENDER_PATH', './time_blender')

        # Other Configuration
        self.log_file_path = os.getenv('LOG_FILE_PATH', 'process_log.txt')
        self.data_report_path = os.getenv('DATA_REPORT_PATH', 'data_report.txt')

    def get_api_config(self) -> dict:
        """Get API configuration"""
        return {
            'api_key': self.api_key,
            'base_url': self.base_url
        }
        
    def get_model_config(self) -> dict:
        """Get model configuration"""
        return {
            'code_generator_model': self.code_generator_model,
            'analysis_model': self.analysis_model,
            'reflector_model': self.reflector_model
        }

    def get_path_config(self) -> dict:
        """Get path configuration"""
        return {
            'time_blender_path': self.time_blender_path,
            'log_file_path': self.log_file_path,
            'data_report_path': self.data_report_path
        }

# Global configuration instance
config = Config()