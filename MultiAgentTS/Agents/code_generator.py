from .code_agent import *
from .code_worker import *
from typing import Tuple, Optional
import os

class CodeGenerator:
    """
    CodeGenerator class for managing the process of generating, extracting, and executing Python code
    using LLM and provided generator code. Now supports multi-turn conversations.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat", ts_generator_path: str = "./time_blender"):
        """
        Initialize the CodeGenerator.
        :param api_key: API key for the LLM
        :param base_url: Base URL for the LLM API
        :param model: LLM model name
        :param ts_generator_path: Path to the directory containing generator code
        """
        # Input validation
        if not api_key:
            raise ValueError("API key cannot be empty")
        if not base_url:
            raise ValueError("Base URL cannot be empty")
        if not model:
            raise ValueError("Model name cannot be empty")
        if not ts_generator_path:
            raise ValueError("Time blender path cannot be empty")
            
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.generator_path = ts_generator_path

        # Validate generator path
        if not os.path.exists(ts_generator_path):
            raise FileNotFoundError(f"Time blender path not found: {ts_generator_path}")

        # Initialize helper objects
        self.code_reader = CodeReader()
        self.code_handler = PythonCodeHandler()
        self._init_code_agent()

    def _init_code_agent(self):
        """Initialize the CodeAgent with current settings."""
        try:
            self.code_agent = CodeAgent(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize CodeAgent: {e}")

    def set_generator_path(self, path: str):
        """
        Set the path for the generator code.
        :param path: Path to the generator code directory
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Generator path not found: {path}")
            
        self.generator_path = path

    def get_generator_code(self) -> str:
        """
        Read and return the generator code from the specified path.
        :return: Generator code as a string
        """
        try:
            return self.code_reader.read_all_files_in_directory(self.generator_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read generator code: {e}")

    def generate_code_response(self, objective: str, output_path: str = "./Data/synthetic_data.csv") -> str:
        """
        Generate Python code using LLM based on the given objective and generator code.
        This starts a new conversation each time.
        :param objective: Task description or objective
        :param output_path: Path where the generated data should be saved
        :return: Generated code as a string
        """
        if not objective:
            raise ValueError("Objective cannot be empty")

        # Add output path information to objective
        full_objective = f"{objective}\n\nThe generated code should save the synthetic data to the following CSV file path: {output_path}"
            
        try:
            generator_code = self.get_generator_code()
            return self.code_agent.create_chat(full_objective, generator_code)
        except Exception as e:
            raise RuntimeError(f"Failed to generate code: {e}")

    def continue_conversation(self, user_input: str) -> str:
        """
        Continue the existing conversation with new user input.
        :param user_input: New user input/message
        :return: LLM's response
        """
        if not user_input:
            raise ValueError("User input cannot be empty")
            
        try:
            return self.code_agent.add_message_and_chat(user_input)
        except Exception as e:
            raise RuntimeError(f"Failed to continue conversation: {e}")

    def reset_conversation(self):
        """
        Reset the conversation history while keeping the system prompt.
        """
        try:
            self.code_agent.reset_conversation()
        except Exception as e:
            raise RuntimeError(f"Failed to reset conversation: {e}")

    def execute_generated_code(self, response: str, output_file_path: str = "./extracted_script.py") -> Tuple[bool, str, str]:
        """
        Execute the generated Python code and return the results.
        :param response: LLM response.
        :param output_file_path: Path to save the generated code
        :return: Tuple (success, stdout, stderr)
        """
        if not response:
            raise ValueError("Response cannot be empty")
            
        try:
            return self.code_handler.process_response_and_execute(response, output_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to execute generated code: {e}")

    def generate_and_execute(self, objective: str, output_file_path: str = "./extracted_script.py", data_output_path: str = "./Data/synthetic_data.csv") -> Tuple[bool, str, str]:
        """
        Generate Python code and execute it, returning the execution results.
        This starts a new conversation each time.
        :param objective: Task description or objective
        :param output_file_path: Path to save the generated code
        :param data_output_path: Path where the generated data should be saved
        :return: Tuple (success, stdout, stderr)
        """
        # Input validation
        if not objective:
            raise ValueError("Objective cannot be empty")
            
        try:
            # Generate code
            response = self.generate_code_response(objective, data_output_path)
            print(response)
            # Execute code
            return self.execute_generated_code(response, output_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to generate and execute code: {e}")

    def show_conversation_history(self):
        """
        Display the current conversation history (for debugging purposes).
        """
        try:
            self.code_agent.show_conversation()
        except Exception as e:
            print(f"Failed to show conversation history: {e}")

    def get_message(self):
        try:
            return self.code_agent.get_message()
        except Exception as e:
            print(f"Failed to get messages: {e}")
            return []