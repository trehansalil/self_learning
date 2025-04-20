from autogen import AssistantAgent, UserProxyAgent

from constants import *

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

config_list_specialist = [
    {
        "model": THINKER_API_MODEL_NAME,
        "api_key": THINKER_API_KEY,
        "base_url": THINKER_API_BASE
    }
]

specialist_agent_config = {
    "seed": seed,  # for reproducibility
    "config_list": config_list_specialist,
    "temperature": 0.4,
    "top_p": 0.1
}

# user agent
user = UserProxyAgent(name="user",
                      human_input_mode="ALWAYS",
                      code_execution_config=False,
                      llm_config=agent_config)

# 1. specialist

specialist = AssistantAgent(name="specialist",
                            system_message="""You are an expert customer care representative of a bank. 
                    Your objective is to provide the user with the best possible banking solutions and service. 
                    Always greet the customer at the start.
                    If the user responds with 'exit' or 'TERMINATE', reply once with "HANDOVER_TO_SURVEYOR" and stop responding.""",
                            llm_config=specialist_agent_config)
# 2. surveyor
surveyor = AssistantAgent(name="surveyor",
                          system_message="""You are responsible for collecting customer satisfaction ratings. 
                    Ask the user to rate their service experience on a scale of 1 to 10.
                    If the user responds with 'exit' or 'TERMINATE', say "Thank you for your time!" and stop responding.""",
                          llm_config=coder_agent_config)

user.initiate_chats(
    [
        {"recipient": specialist,
         "message": "Hello",
         "summary_method": "reflection_with_llm"
         },
        {"recipient": surveyor,
         "message": "Please ask the user to rate their service experience on a scale of 1 to 10.",
         "carryover": ["context", "summary"],
         "trigger_condition": "HANDOVER_TO_SURVEYOR"
         }
    ]
)
