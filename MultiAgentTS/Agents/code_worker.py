import os
import re
import subprocess
import json

class CodeReader:
    def __init__(self):
        pass

    @staticmethod
    def read_file_as_string(file_path):
        """
        Reads the content of a single file as a string.
        :param file_path: Path to the file
        :return: File content as string or an error message if reading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"

    def read_all_files_in_directory(self, directory_path):
        """
        Reads all files in the given directory and returns their names and contents as a list of dictionaries.
        :param directory_path: Path to the directory containing the files
        :return: List of dictionaries with 'file_name' and 'content' keys
        """
        if not os.path.isdir(directory_path):
            return f"Error: {directory_path} is not a valid directory."

        file_data = []
        try:
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):  # Ensure it's a file, not a directory
                    content = self.read_file_as_string(file_path)
                    file_data.append({
                        'file_name': file_name,
                        'content': content
                    })
        except Exception as e:
            return f"Error reading files in directory {directory_path}: {str(e)}"
        file_data = json.dumps(file_data, indent=4, ensure_ascii=False)
        return file_data

class PythonCodeHandler:
    def __init__(self):
        pass

    @staticmethod
    def extract_code_from_response(response_code):
        """
        Extracts Python code block from the provided response string.
        :param response_code: String containing the response with code blocks
        :return: Extracted Python code as a string, or None if no code block is found
        """
        match = re.search(r"```python(.*?)```", response_code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def save_code_to_file(code, file_path):
        """
        Saves the extracted code to a file.
        :param code: Python code as a string
        :param file_path: Path to save the Python file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(code)
            return True
        except Exception as e:
            print(f"Error saving code to file: {e}")
            return False

    @staticmethod
    def execute_code(file_path):
        """
        Executes the Python script at the given file path.
        :param file_path: Path to the Python file
        :return: Tuple of (stdout, stderr) from the execution
        """
        try:
            result = subprocess.run(["python", file_path], capture_output=True, text=True)
            return result.stdout, result.stderr
        except Exception as e:
            return "", f"Error executing script: {e}"

    def process_response_and_execute(self, response_code, output_file_path):
        """
        Main function to extract Python code, save it to a file, and execute it.
        :param response_code: String containing the response with code blocks
        :param output_file_path: Path to save the extracted Python script
        :return: Tuple of (success, stdout, stderr)
        """
        extracted_code = self.extract_code_from_response(response_code)
        if not extracted_code:
            return False, "", "No Python code block found."

        if not self.save_code_to_file(extracted_code, output_file_path):
            return False, "", "Failed to save code to file."

        stdout, stderr = self.execute_code(output_file_path)
        return True, stdout, stderr