import base64
import os
from openai import OpenAI
from typing import Optional, List
import time

class AnalysisAgent:
    """
    AnalysisAgent class for interacting with an LLM API to generate responses.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o", temperature: float = 0.9):
        """
        Initialize conversation Agent
        :param api_key: API key
        :param base_url: API base URL
        :param model: Model name to use (default: gpt-4o)
        :param temperature: Generation temperature (default: 0.7)
        """
        # Input validation
        if not api_key:
            raise ValueError("API key cannot be empty")
        if not base_url:
            raise ValueError("Base URL cannot be empty")
        if not model:
            raise ValueError("Model name cannot be empty")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

        # Initialize conversation history with system role definition
        self.messages = [
            {
                "role": "system",
                "content": "You are an expert in time series data analysis."
            }
        ]
        
        try:
            self.client = OpenAI(api_key=self.api_key,
                                 base_url=self.base_url)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

    def reset_conversation(self):
        """
        Reset conversation history (keep system role definition)
        """
        self.messages = self.messages[:1]  # Keep system message

    def show_conversation(self):
        """
        Print current conversation history (for debugging)
        """
        for idx, msg in enumerate(self.messages[1:]):  # Skip system message
            print(f"[{idx + 1}] {msg['role'].upper()}:")
            content = msg['content']
            if content:
                print(content[:200] + "..." if len(content) > 200 else content)
            else:
                print("(empty message)")
            print("\n" + "-" * 50 + "\n")

    def insert_txt_into_input(self, txt_file_path: str) -> str:
        """
        Read the content of specified txt file and return as string
        :param txt_file_path: Path to txt file
        :return: File content (string)
        """
        # Input validation
        if not txt_file_path:
            raise ValueError("File path cannot be empty")

        if not os.path.exists(txt_file_path):
            raise FileNotFoundError(f"Text file not found: {txt_file_path}")

        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            print(f"Failed to read txt file: {str(e)}")
            return ""

    def encode_image(self, image_path: str) -> str:
        """
        Read image file, convert image to base64 string, and return base64 encoded string
        :param image_path: Path to image file
        :return: Base64 encoded image string
        """
        # Input validation
        if not image_path:
            raise ValueError("Image path cannot be empty")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            # Convert image to base64 string
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            return image_base64
        except Exception as e:
            print(f"Failed to read image: {str(e)}")
            return ""

    def encode_images_in_folder(self, folder_path: str) -> List[str]:
        """
        Encode all .webp files in folder and return list of base64 encoded strings
        :param folder_path: Path to folder
        :return: List of image base64 encoded strings
        """
        # Input validation
        if not folder_path:
            raise ValueError("Folder path cannot be empty")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_base64_list = []
        try:
            for filename in os.listdir(folder_path):
                if filename.endswith(".webp"):
                    image_path = os.path.join(folder_path, filename)
                    image_base64 = self.encode_image(image_path)
                    if image_base64:  # If conversion successful, add to list
                        image_base64_list.append(image_base64)
        except Exception as e:
            print(f"Failed to read folder images: {str(e)}")

        return image_base64_list

    def analysis_chat(self, txt_file_path: Optional[str] = None, folder_path: Optional[str] = None, caption: str = "", dataset_description: str = ""):
        """
        Merge basic input, txt file content and image content into a single message, and call LLM for conversation
        :param txt_file_path: Optional txt file path (inserted into input)
        :param folder_path: Optional folder path containing all .webp files
        :param caption: Image description (default: "Image content below:")
        :param dataset_description: Dataset description information
        :return: LLM's response
        """
        base_input = """Please carefully examine this time-series data image, where the horizontal axis represents time and the vertical axis represents values. 
                        You are required to analyze the data changes solely based on the image, using natural language to provide a detailed explanation of the data trends, including but not limited to the following:                       
                        Describe the detailed trend and the periodic of every day. And you need to provide a detailed description of the occurrence of peaks and valleys (and the detail number of peaks and valleys). 
                        If the image displays periodic fluctuations, please describe the periodic characteristics and the performance of each day within each cycle.
                        Make sure your description is detailed, well-organized, and provides an intuitive analysis of the data changes along with possible explanations (the events in the real world).
                        **Be as detailed as possible**"""

        # If dataset description provided, add to input
        if dataset_description:
            base_input = f"Dataset Description: {dataset_description}\n\n{base_input}"

        try:
            # Get base64 encoding of all webp images in folder
            image_base64_list = []
            if folder_path:
                image_base64_list = self.encode_images_in_folder(folder_path)

            # Add images one by one to message content
            image_messages = []
            for image_base64 in image_base64_list:
                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{image_base64}"
                    }
                })

            # Add merged input to conversation history
            content_parts = [{"type": "text", "text": base_input + caption}]
            content_parts.extend(image_messages)
            
            self.messages.append({
                "role": "user",
                "content": content_parts
            })

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    stream=False
                )
                assistant_reply = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_reply})
                return assistant_reply
            except Exception as e:
                print(f"API request failed: {str(e)}")
                self.messages.pop()  # Remove failed user message
                return None
        except Exception as e:
            print(f"Error in analysis chat process: {str(e)}")
            return None

    def add_message_and_chat(self, input_text: str) -> Optional[str]:
        """
        Add message and get reply
        """
        # Input validation
        if not input_text:
            raise ValueError("Input text cannot be empty")

        self.messages.append({"role": "user", "content": input_text})
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Create API request
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    stream=False
                )

                # Get assistant reply
                assistant_reply = response.choices[0].message.content

                # Add assistant reply to conversation history
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_reply
                })

                return assistant_reply

            except Exception as e:
                retry_count += 1
                print(f"API request failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    self.messages.pop()  # Remove failed user message
                    return None