import inspect
from typing import (
    Any,
    Dict,
    List,
    Optional
)

import instructor
from instructor.client import Instructor
from langchain.llms.base import BaseLLM
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import Generation, LLMResult, RunInfo
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from pydantic import Field

from src.constants import LLAMA3_2_API_BASE, LLAMA3_2_API_KEY, LLAMA3_2_API_MODEL_NAME
from src.logging import trace_logger


class RouterLLM(BaseLLM):
    """
    A custom implementation of a Language Learning Model (LLM) Router
    to handle dynamic prompt generation and response formatting.

    Attributes:
        system_message (Optional[str]): Default message to guide the LLM's behavior.
        attributes (List[str]): List of attributes to include in the LLM response.
        remove_attributes (List[str]): Attributes to exclude from the LLM response.
        language (str): Language code (e.g., "en") for the LLM interaction.
        openai_api_base (str): API base URL for OpenAI.
        openai_api_key (str): API key for OpenAI.
        model (str): Name of the OpenAI model to use.
        top_p (float): Top-p sampling for the response generation.
        temperature (float): Temperature for controlling randomness in responses.
        max_tokens (int): Maximum number of tokens in a response.
        max_tries (int): Number of attempts for generating a valid response.
        frequency_penalty (float): Penalty to reduce repetitive text.
        presence_penalty (float): Penalty to reduce redundancy in context.
        response_format (Optional[Dict]): Format specification for the output.
        qualifier_dict (Optional[Dict]): Qualification rules for response attributes.
        complexity_dict (Optional[Dict]): Complexity rules for response attributes.
        input_model_params (Optional[Dict]): User-specified model parameters.
        model_params (Optional[Dict]): Parameters used during generation.
        client (Optional[Any]): OpenAI client instance for handling requests.
        set_params (List): List of parameters allowed to be directly set.

    Methods:
        __init__: Initializes the RouterLLM with optional parameters.
        _generate: Generates responses based on input prompts.
        _llm_type: Returns the LLM type as a string.
    """

    system_message: Optional[str] = Field(default=None)
    attributes: List[str] = []
    remove_attributes: List[str] = []
    language: str = Field(default="en")
    openai_api_base: str = "your_api_base_url"
    openai_api_key: str = "your_api_key"
    model: str = "your_model_name"
    top_p: float = 0.2
    temperature: float = 0
    max_tokens: int = 300
    max_tries: int = 1
    frequency_penalty: float = 0.5
    presence_penalty: float = 0.5
    response_format: Optional[Dict] = None
    qualifier_dict: Optional[Dict] = None
    complexity_dict: Optional[Dict] = None
    diagram_dict: Optional[Dict] = None
    policy_dict: Optional[Dict] = None
    general_dict: Optional[Dict] = None
    sql_dict: Optional[Dict] = None
    input_model_params: Optional[Dict] = None
    model_params: Optional[Dict] = {}
    openai_client: Optional[Any] = None
    client: Optional[Any] = None
    set_params: List = ['remove_attributes', 'language', 'openai_api_base', 'openai_api_key', 'model', 'mode']
    mode: str = "json_schema_with_response_format"

    # , 'response_model'
    def __init__(self, system_message=None, **kwargs):
        """
        Initializes the RouterLLM instance with default or user-provided values.

        Args:
            system_message (str, optional): A message to guide the LLM's behavior.
            kwargs: Additional parameters for model configuration.
        """
        self.input_model_params = {}  # Initialize here
        self.__update__(system_message, **kwargs)

    def __model_update__(self, **kwargs):
        """
        Updates the model parameters with the provided keyword arguments.

        Args:
            kwargs: Additional parameters for model configuration.
        """

        self.openai_api_base = kwargs.get("openai_api_base", LLAMA3_2_API_BASE)
        self.openai_api_key = kwargs.get("openai_api_key", LLAMA3_2_API_KEY)
        self.model = kwargs.get("model", LLAMA3_2_API_MODEL_NAME)

        # Organize input parameters for further usage
        for key, value in kwargs.items():
            if '_dict' in key or key in self.set_params:
                setattr(self, key, value)
            elif (key in ['tags', 'callbacks']) | ('_prompt' in key) | ('_keys' in key):
                continue
            else:
                self.input_model_params[key] = value

                # Initialize OpenAI client with required settings
        if self.mode == 'json_schema_with_response_format':
            self.openai_client = OpenAI(
                base_url=self.openai_api_base,
                api_key=self.openai_api_key
            )

            self.client: Instructor = instructor.from_openai(
                self.openai_client,
                mode=instructor.Mode.TOOLS
            )
        else:
            self.openai_client = OpenAI(
                base_url=self.openai_api_base,
                api_key=self.openai_api_key
            )

    def __update__(self, system_message=None, **kwargs):
        """
        Updates the class parameters with the provided keyword arguments.

        Args:
            system_message (str): A message to guide the LLM's behavior.
            kwargs: Additional parameters for model configuration.
        """
        # Update the model parameters with the provided keyword arguments
        # Provide a default system message if none is supplied
        if system_message is None:
            system_message = "You are an expert at extracting information from user input. Always format your responses as JSON."

        # Pass system_message as part of kwargs for Pydantic validation
        kwargs['system_message'] = system_message

        # Initialize superclass attributes
        super().__init__(**kwargs)

        # Save the system message and other parameters
        self.system_message = kwargs['system_message']
        del kwargs['system_message']

        # Organize input parameters for further usage
        self.input_model_params = {}
        self.__model_update__(**kwargs)

    def fetch_response_model(self, **kwargs):
        """
        Fetches the response model from the keyword arguments.
        """
        # Check if a response model is provided in the keyword arguments
        if kwargs.get("response_model", None) is not None:
            response_model = kwargs.get("response_model")

            if hasattr(response_model, 'set_definition'):
                response_model = response_model.set_definition(self.language)
        else:
            # Raise an exception if the response model is not provided
            raise Exception(
                f"`response_model` input not received at `{self.__class__.__name__}` within {inspect.currentframe().f_code.co_name}")
        return response_model

    def __generate_function_call__(self, prompt, **kwargs):
        pass

    def fetch_tool_functions(self, **kwargs):

        tool_functions = kwargs.get("tool_functions", None)

        if tool_functions is None:
            raise Exception("Tool functions are not provided.")

        functions = []

        for tool_function in tool_functions:
            # if isinstance(tool_function, instructor.OpenAISchema):
            functions.append({"type": "function", "function": tool_function.openai_schema})
        if functions:
            return functions
        else:
            raise Exception(
                "No functions found in the tool functions. Please stick to functions inheriting from `OpenAISchema`")

    @staticmethod
    def is_valid_base64_url(s):
        try:
            return "data:image/" in s
        except Exception:
            return False

    def function_executor(self, execute_func, dataframe, tool_functions, **kwargs):
        tool_results = []
        tool_resultants = []

        chat_history: List = kwargs.get("chat_history")
        completion_message: ChatCompletionMessage = kwargs.get("completion_message")
        query = chat_history[1]['content'].split('\n')[0]

        if not completion_message.tool_calls:
            trace_logger.info(f"{query}\033[90m🛠️ No Tool call will be invoked.\033[0m")

        for tool_call in completion_message.tool_calls:
            trace_logger.info(f"{query}\033[90m🛠️ {tool_call.function.name} {tool_call.function.arguments}\033[0m")
            tool_result = execute_func(tool_call, tool_functions, dataframe)

            if isinstance(tool_result, tuple):
                for tool_res in tool_result:
                    if isinstance(tool_res, str):
                        if not self.is_valid_base64_url(tool_res):
                            tool_resultants.append(tool_res)
            else:
                if isinstance(tool_result, str):
                    if not self.is_valid_base64_url(tool_result):
                        tool_resultants.append(tool_result)

            trace_logger.info(f"\n\nTool Resultants:\n\n{tool_resultants}\n\n")
            chat_history.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": tool_resultants,
            })

            tool_results.append(tool_result)
        return {"tool_results": tool_results, "chat_history": chat_history}

    # @log_time_to_sentry(step_name="`RouterLLM` generate Structured Response")
    def __generate_response__(self, prompt, chat_history=[], **kwargs):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        if chat_history:
            messages.extend(chat_history)

        if self.mode == 'json_schema_with_response_format':
            # Use local variables instead of self.model_params
            model_params = {
                "messages": messages,
                "response_model": kwargs.get("response_model"),
                "model": self.model,
            }
            model_params.update(self.input_model_params)

            try:
                response = self.client.chat.completions.create(**model_params)
            except Exception as e:
                print(f"Error generating response: {e}")
                return {}
        elif self.mode == 'function_calling':
            # Similar refactoring for other modes
            model_params = dict(
                messages=messages,
                tools=kwargs.get("tool_functions"),
                tool_choice="auto",
                model=self.model,  # The specific model instance
            )

            # Update model parameters with any additional input parameters
            model_params.update(self.input_model_params)

            try:
                # Generate response from the OpenAI client using the defined parameters
                completion = self.openai_client.chat.completions.create(**model_params)
                completion_message = completion.choices[0].message

                model_params['messages'].append(completion_message)

                if completion_message.tool_calls is None:
                    return {'completion_message': completion_message}
                else:
                    return {'completion_message': completion_message, 'chat_history': model_params['messages']}
            except Exception as e:
                # Print error message if response generation fails
                print(f"Error while generating response: {e}")
                return {}
        else:
            # to be implemented for text responses only later
            return {}

        return response

    def __generate_text_response__(self, prompt, chat_history=[], **kwargs):

        messages = chat_history

        messages.append({"role": "user", "content": prompt})

        self.__model_update__(**kwargs)

        self.model_params = dict(
            messages=messages,
            model=self.model  # The specific model instance
        )

        response = self.openai_client.chat.completions.create(
            **self.model_params
        )

        resp = response.choices[0].message

        if len(resp.tool_calls):
            response = resp.tool_calls[0]
        else:
            response = resp.content.replace("\\n", "\n")

        return response

    def structured_output_to_pydantic_model(self, prompt: str, **kwargs):
        """
        Generates a structured output from a Pydantic model based on the provided prompt.

        This method constructs a request to the language model using the specified prompt and
        additional parameters. It handles the response generation and any potential errors
        that may occur during the process.

        Args:
            prompt (str): The input prompt for which the model should generate a response.
            **kwargs: Additional keyword arguments that may include:
                - response_model: The model to be used for generating the response.

        Returns:
            dict: The response generated by the language model, or an empty dictionary
                in case of an error.

        Raises:
            Exception: Raises an exception if the `response_model` is not provided.
        """
        if self.mode == 'json_schema_with_response_format':
            # Check if a response model is provided in the keyword arguments
            response_model = self.fetch_response_model(**kwargs)

            # Define parameters for generating a response
            return self.__generate_response__(
                prompt=prompt,
                response_model=response_model,
                chat_history=kwargs.get("chat_history", []),
            )  # Return the generated response
        else:
            tool_functions = self.fetch_tool_functions(**kwargs)

            # Define parameters for generating a response
            return self.__generate_response__(
                prompt=prompt,
                tool_functions=tool_functions,
                chat_history=kwargs.get("chat_history", []),
            )  # Return the generated response

    def _generate(self, prompts: List[str], stop=None, **kwargs):
        """
        Generates a response for a list of prompts using the LLM client.

        Args:
            prompts (List[str]): A list of user input prompts for the LLM.
            stop (Optional[str]): Optional stopping criteria for the LLM generation.
            kwargs: Additional parameters for model behavior.

        Returns:
            LLMResult: The generated responses from the LLM.
        """
        # define the response model's language setup if the model exists
        response_model = self.fetch_response_model(**kwargs)

        generations = []
        for prompt in prompts:
            # Define parameters for generating a response
            response = self.__generate_response__(
                prompt=prompt,
                response_model=response_model,
                chat_history=kwargs.get("chat_history", []),
            )

            # Set attributes to include in the response if not already defined
            if not self.attributes:
                self.attributes = list(response.__dict__.keys())
                for attribute in self.remove_attributes:
                    if attribute in self.attributes:
                        self.attributes.remove(attribute)

            response.user_message = prompt

            # Process response choices based on attributes and associated dictionaries
            for attribute in self.attributes:
                if hasattr(response, attribute) and hasattr(self, f"{attribute}_dict"):
                    try:
                        getattr(response, attribute).choice = getattr(self, f"{attribute}_dict")[
                            getattr(response, attribute).category]
                    except Exception as e:
                        print(f"Response: {response} with erorr {e}")
                        raise e
            try:
                # Convert the response to JSON format and store it
                generation = Generation(text=response.model_dump_json(indent=2))
                generations.append(generation)
            except AttributeError as e:
                trace_logger.error(f"Error generating response: {e}")
                raise e

        return LLMResult(generations=[generations])

    def _generate_helper(
            self,
            prompts: list[str],
            stop: Optional[list[str]],
            run_managers: list[CallbackManagerForLLMRun],
            new_arg_supported: bool,
            **kwargs: Any,
    ) -> LLMResult:
        try:
            output = (
                self._generate(
                    prompts,
                    stop=stop,
                    # TODO: support multiple run managers
                    run_manager=run_managers[0] if run_managers else None,
                    **kwargs,
                )
                if new_arg_supported
                else self._generate(prompts, stop=stop, **kwargs)
            )
        except BaseException as e:
            for run_manager in run_managers:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise e
        flattened_outputs = output.flatten()
        for manager, flattened_output in zip(run_managers, flattened_outputs):
            manager.on_llm_end(flattened_output)
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    @property
    def _llm_type(self):
        """
        Returns the type of the LLM being used.

        Returns:
            str: The type of the LLM, here 'router_llm'.
        """
        return "router_llm"
