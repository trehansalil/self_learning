from autogen import UserProxyAgent, AssistantAgent

from src.constants import *

seed = 43
config_list_gpt = [
    {
        "model": CODER_API_MODEL_NAME,
        "api_key": CODER_API_KEY,
        "base_url": CODER_API_BASE
    },
    {
        "model": QWEN_API_MODEL_NAME,
        "api_key": QWEN_API_KEY,
        "base_url": QWEN_API_BASE
    }
]

agent_config = {
    "seed": seed,  # for reproducibility
    "config_list": config_list_gpt,
    "temperature": 0.7,
}

config_list_coder = [
    {
        "model": LLAMA3_2_API_MODEL_NAME,
        "api_key": LLAMA3_2_API_KEY,
        "base_url": LLAMA3_2_API_BASE
    },
    {
        "model": CODER_API_MODEL_NAME,
        "api_key": CODER_API_KEY,
        "base_url": CODER_API_BASE
    },
    {
        "model": QWEN_API_MODEL_NAME,
        "api_key": QWEN_API_KEY,
        "base_url": QWEN_API_BASE
    },
    {
        "model": THINKER_API_MODEL_NAME,
        "api_key": THINKER_API_KEY,
        "base_url": THINKER_API_BASE
    }
]

coder_agent_config = {
    "seed": seed,  # for reproducibility
    "config_list": config_list_coder,
    "temperature": 0.2,
    "top_p": 0.1
}

assistant = AssistantAgent(
    name='assistant',
    system_message="You are responsible for code generation. Please communicate with user agent and fix any issues.",
    llm_config=coder_agent_config,
)

user = UserProxyAgent(
    name='user',
    system_message="You are responsible for code executor. Please communicate with assistant agent and fix any issues. Ask for user feedback at the end of code execution.",
    human_input_mode="TERMINATE",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    llm_config=agent_config,
)

user.initiate_chat(assistant,
                   message="Please prepare a technical report for Facebook stock in the last 12 months. Save any relevant plots in the relevant folders. Please look at MACD, 50 DMA, 200 DMA, RSI")
