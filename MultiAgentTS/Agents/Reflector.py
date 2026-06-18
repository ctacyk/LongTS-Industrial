import requests
import openai
from openai import OpenAI
from typing import Optional, List
import time


class Reflector:
    """
    Reflector class for interacting with an LLM API to generate responses.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize the conversation Agent
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
                "content": "You are an expert in time series data generation and simulation using Python. "
                           "Your job is to review generated result from code-agent."
                           "You will always get the report for the synthetic from the code."
                           "You need to determine whether the generated data fits the real-world situation."
                           "And then determine whether the code is acceptable"
                           "Please respond with 'ACCEPTABLE' if the code is good, or respond with 'NOT ACCEPTABLE' and provide detailed suggestions otherwise."
            }
        ]
        
        try:
            self.client = OpenAI(api_key=self.api_key,
                                 base_url=self.base_url)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

    def create_chat(self, user_content: str) -> Optional[str]:
        """
        Chat with LLM, send user message and return reply
        """
        # Input validation
        if not user_content:
            raise ValueError("User content cannot be empty")

        self.messages.append({"role": "user", "content": user_content})
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
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
                retry_count += 1
                print(f"API request failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    self.messages.pop()  # Remove failed user message
                    return None

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

    def get_message(self) -> List[dict]:
        return self.messages

    def add_message_and_chat(self, input_text: str) -> Optional[str]:
        """
        Add message and get reply
        """
        return self.create_chat(input_text)

    def set_context(self, input_messages: List[dict]):
        """
        Set context messages
        """
        if not isinstance(input_messages, list):
            raise TypeError("input_messages must be a list")

        # Skip system message, replace other messages
        self.messages[1:] = input_messages[1:]