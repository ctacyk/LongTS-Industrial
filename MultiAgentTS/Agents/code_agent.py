import requests
import openai
from openai import OpenAI


class CodeAgent:
    """
    CodeAgent class for interacting with an LLM API to generate responses.
    """

    def __init__(self, api_key, base_url, model="gpt-4o", temperature=0.7):
        """
        Initialize the conversation Agent
        :param api_key: API key
        :param base_url: API base URL
        :param model: Model name to use (default: gpt-4o)
        :param temperature: Generation temperature (default: 0.7)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

        # Initialize conversation history with system role definition
        self.messages = [
            {
                "role": "system",
                "content": "You are an expert in time series data generation and simulation using Python."
                           "Always respond with both Python code and analysis text."
                           "Strictly follow the output requirements in your responses."
            }
        ]
        self.client = OpenAI(api_key=self.api_key,
                        base_url=self.base_url)
    def create_chat(self, objective, generator_code):
        """
        Generate a response based on the objective and provided code.
        :param objective: Description of the task (Objective)
        :param generator_code: Provided Python code or modules
        :return: Content of the LLM's response
        """
        user_content = f"""
                    ## Objective
                    {objective}
                    ### Provided Code
                        Read carefully and try your best to understand the code.
                        Use the following time_blender modules for data generation:
                        ## Introduction

                        TimeBlender is a programmatic compositional time series generator. By *programmatic* it is meant that series are
                        specified through programming; and, by *compositional*, that the programming structures used to this end
                        can be combined (or, well, *blended*) to achieve complex results.
                        
                        This software has a dual purpose:
                        
                          - to allow the author to study time series from a generative point of view, either by implementing existing
                            concepts or researching new ones.
                          - and to produce artificial time series of practical interest.
                          
                        On the one hand, these two objectives are antagonistic in the sense that early research might result in inadequate or not 
                        optimal ways of achieving results (i.e., because new ways are being sought, and some of these might prove pointless), 
                        which hinders practical uses.  On the other hand, they are complementary in the sense that good results are not easy to 
                        achieve without good foundations, and these require research. Obviously, the present software is being developed 
                        because the latter aspect seems stronger, owing to a perception that in industrial practice we lack good synthetic time 
                        series generation. For example, while general ARMA-based generators are easy to find, the author could not locate 
                        generators for user behavior in financial applications such as banking (see model examples below).
                        
                        Target applications include:
                        
                          - Permit data scientists to work with artificial, but realistic, data while access to real data is not available
                            (either because it does not exist yet, or because bureaucratic procedures create unreasonable delays).
                          
                          - Data augmentation, particularly for RNN models.
                          
                          - Artificial stress scenarios simulation (e.g., a market crash).
                          
                        ## Features
                        
                        Main features:
                        
                          - Event-based: each time point is generated based on an *event* class. Several standard such events are provided,
                                         and it is easy to add more.
                          - Programmatic: events can be specified by arbitrary (Python) programs, hence they are not limited to traditional
                                          statistical techniques. For example, agent-based models could be defined to simulate
                                          market data. 
                          - Compositional: events can be composed to obtain complex events.
                          - Pandas-based: Pandas is used in various parts to allow convenient post-generation processing options and integration 
                                          with other tools.
                        
                        Standard models, which work both as examples and as a basic model library, include:
                        
                          - AR, MA and ARMA.
                          - Seasonal effects.
                          - Banking behavior of salary earners.
                          - Kondratiev business cycle.
                          
                        Please note that some of the above are provided as rather naive implementations. It is hoped that more sophisticated
                        models take their place as the library improves.

                    ```python 
                    {
                               {generator_code}
                    }
                    ```
                                          **Constraints**:
                                             - The code must adhere to the given constraints and be executable without requiring unavailable dependencies.
                                             - Analyze the real condition and use time-blender to simulate the data of it.You should define custom events to simulate real conditions.
                                             - Make sure the code is runnable.
           
                                          ### Output Requirements
                                          - Enclose the entire Python script within a **single code block** using  python .
                                          - The script must:
                                            Save the generated data to a CSV file(**path is ./Data/synthetic_data.csv - this is a placeholder, the actual path will be specified by the system**).
                                          Generate Python code that adheres to the above requirements, and give me the analysis of your generation.
                                          """
        self.messages.append({"role": "user", "content": user_content})
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
            # Error handling
            print(f"API request failed: {str(e)}")
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
            print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
            print("\n" + "-" * 50 + "\n")

    def get_message(self):
        return self.messages

    def add_message_and_chat(self, input):
        self.messages.append({"role": "system",
                              "content": "You are an expert in time series data generation and simulation using Python."
                              "Always respond with both Python code and analysis text."
                              "**Strictly follow the output requirements in your responses**."
                              "The output CSV file path will be specified by the system, do not hardcode the path."
            })
        self.messages.append({"role": "user", "content": input})
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
            # Error handling
            print(f"API request failed: {str(e)}")
            self.messages.pop()  # Remove failed user message
            return None