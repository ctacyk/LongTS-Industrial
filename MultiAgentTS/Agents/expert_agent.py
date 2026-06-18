"""
Expert Agent for expanding and enriching dataset descriptions with domain knowledge.
This agent takes basic dataset descriptions and expands them with:
- Detailed dimension analysis
- Relationships between variables
- Domain-specific knowledge
- Influencing factors
- Temporal patterns
"""

from openai import OpenAI
from typing import Optional


class ExpertAgent:
    """
    Expert Agent for expanding dataset descriptions with domain knowledge.
    """

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat"):
        """
        Initialize the Expert Agent.
        :param api_key: API key for the LLM
        :param base_url: Base URL for the LLM API
        :param model: LLM model name
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if not base_url:
            raise ValueError("Base URL cannot be empty")
        if not model:
            raise ValueError("Model name cannot be empty")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation_history = []

    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for the Expert Agent.
        :return: System prompt string
        """
        return """You are an expert domain knowledge specialist for time series data generation.
Your role is to expand and enrich dataset descriptions with comprehensive domain knowledge.

When given a basic dataset description, you should provide:

1. **Detailed Dimension Analysis**:
   - For each dimension/variable, explain:
     * Expected trends and patterns
     * Influencing factors
     * Typical value ranges
     * Temporal characteristics (daily, weekly, seasonal patterns)
     * Anomalies or special behaviors

2. **Relationships Between Dimensions**:
   - Correlations between variables
   - Causal relationships
   - Dependencies and interactions
   - How changes in one dimension affect others

3. **Domain-Specific Knowledge**:
   - Industry-specific insights
   - Physical or business constraints
   - Real-world events that impact the data
   - Seasonal and cyclical patterns
   - External factors (weather, holidays, events, etc.)

4. **Data Generation Guidance**:
   - Realistic value ranges for each dimension
   - Typical patterns to simulate
   - Edge cases and anomalies to consider
   - Temporal granularity recommendations
   - Validation criteria

5. **Event Simulation Recommendations**:
   - Types of events that should be simulated
   - Typical event durations and frequencies
   - Impact patterns on different dimensions
   - Recovery patterns after events

Format your response in a clear, structured manner with sections and bullet points.
Be specific and practical, focusing on actionable insights for data generation."""

    def expand_description(self, basic_description: str, dataset_name: str = "") -> str:
        """
        Expand a basic dataset description with domain knowledge.
        :param basic_description: Basic description of the dataset
        :param dataset_name: Name of the dataset (optional)
        :return: Expanded description with domain knowledge
        """
        if not basic_description:
            raise ValueError("Basic description cannot be empty")

        # Create the user prompt
        user_prompt = f"""Please expand and enrich the following dataset description with comprehensive domain knowledge:

Dataset Name: {dataset_name if dataset_name else "Unknown"}

Basic Description:
{basic_description}

Provide a detailed expansion that includes:
1. Detailed analysis of each dimension/variable
2. Relationships between dimensions
3. Domain-specific knowledge and insights
4. Temporal patterns and characteristics
5. Event simulation recommendations
6. Data generation guidance

Make the expansion practical and actionable for synthetic data generation."""

        # Reset conversation history for a fresh expansion
        self.conversation_history = []

        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": self._create_system_prompt()
        })

        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })

        # Get response from LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=4000
            )

            expanded_description = response.choices[0].message.content

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": expanded_description
            })

            return expanded_description

        except Exception as e:
            raise RuntimeError(f"Failed to expand description: {e}")

    def refine_expansion(self, feedback: str) -> str:
        """
        Refine the expansion based on feedback.
        :param feedback: Feedback on the expansion
        :return: Refined expansion
        """
        if not feedback:
            raise ValueError("Feedback cannot be empty")

        if not self.conversation_history:
            raise RuntimeError("No previous expansion to refine. Call expand_description first.")

        # Add user feedback
        self.conversation_history.append({
            "role": "user",
            "content": f"Please refine the expansion based on this feedback:\n{feedback}"
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=4000
            )

            refined_description = response.choices[0].message.content

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": refined_description
            })

            return refined_description

        except Exception as e:
            raise RuntimeError(f"Failed to refine expansion: {e}")

    def get_event_recommendations(self, expanded_description: str) -> str:
        """
        Get specific event simulation recommendations based on the expanded description.
        :param expanded_description: The expanded dataset description
        :return: Event recommendations
        """
        if not expanded_description:
            raise ValueError("Expanded description cannot be empty")

        user_prompt = f"""Based on the following dataset description, provide specific recommendations for event simulation:

{expanded_description}

Please provide:
1. List of realistic events that should be simulated
2. For each event:
   - Typical duration
   - Frequency of occurrence
   - Impact on each dimension
   - Recovery pattern
   - Realistic parameters for simulation

Format as a structured list that can be used for code generation."""

        # Reset conversation for event recommendations
        self.conversation_history = []

        self.conversation_history.append({
            "role": "system",
            "content": self._create_system_prompt()
        })

        self.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=3000
            )

            recommendations = response.choices[0].message.content

            self.conversation_history.append({
                "role": "assistant",
                "content": recommendations
            })

            return recommendations

        except Exception as e:
            raise RuntimeError(f"Failed to get event recommendations: {e}")

    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []

    def show_conversation(self):
        """
        Display the current conversation history (for debugging).
        """
        for i, msg in enumerate(self.conversation_history):
            print(f"\n--- Message {i + 1} ---")
            print(f"Role: {msg['role']}")
            print(f"Content: {msg['content'][:200]}...")  # Show first 200 chars

