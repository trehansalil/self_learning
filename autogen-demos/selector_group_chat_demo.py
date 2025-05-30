import asyncio
from typing import Sequence, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination, HandoffTermination, \
    StopMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ModelInfo, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.constants import QWEN_API_MODEL_NAME, QWEN_API_KEY, QWEN_API_BASE, generate_system_message
from src.tools import get_data_context_tool, get_policy_context_tool, get_general_context_tool, check_grammar_tool

other_client = OpenAIChatCompletionClient(
    model=QWEN_API_MODEL_NAME,
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_BASE,
    temperature=0.7,
    max_tokens=1024,
    seed=42,
    model_info=ModelInfo(function_calling=True, family=ModelFamily.ANY, vision=False, json_output=True,
                         structured_output=True)
)

greeting_client = OpenAIChatCompletionClient(
    model=QWEN_API_MODEL_NAME,
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_BASE,
    temperature=0.3,
    top_p=0.1,
    max_tokens=1024,
    seed=42,
    model_info=ModelInfo(function_calling=True, family=ModelFamily.ANY, vision=False, json_output=True,
                         structured_output=True)
)
system_message = "When you have completed the task, respond with 'TERMINATE' to end the conversation."

greeting_agent = AssistantAgent(
    name="greeting_agent",
    system_message=generate_system_message("English") + system_message,
    description="Handles simple greetings and hellos, general conversation about the agentic system. Provides friendly introductions and helps users understand how to interact with the system.",
    model_client=greeting_client,
    tools=[]  # No tools needed
)

policy_data_agent = AssistantAgent(
    name="policy_data_agent",
    system_message="You are a Labour and Economic Policy Expert Agent with 12+ years of Experience in the International Market. Your task is to provide intuitive insights based on the context provided to you. " + system_message,
    description="Handles labour/economic policy and data related queries in a country.",
    model_client=other_client,
    tools=[get_policy_context_tool, get_data_context_tool]
)

# data_agent = AssistantAgent(
#     name="data_agent",
#     system_message="Answers questions about economic and labor data.",
#     model_client=other_client,
#     tools=[]
# )

general_agent = AssistantAgent(
    name="general_agent",
    system_message="You are an Expert All-Rounder having all-round knowledge to almost all topics related to General Knowledge and planning. " + system_message,
    description="Handles general knowledge-based queries."
                "You have details about this chatbot and any queries regrading this chatbot should be handled by you",
    model_client=other_client,
    tools=[get_general_context_tool]
)

grammar_agent = AssistantAgent(
    name="grammar_agent",
    system_message="""  
        You are a friendly grammar helper for Arabic Government Professionals.  Analyze the following text, 
        correct any grammar mistakes, and explain the errors in a way that a 
        child can easily understand.  Don't just list the errors; explain them 
        in a paragraph using simple but concise language.

        First, provide the corrected text.

        Then, leave two new lines.

        Finally, provide the explanation. If there are no errors, reply with an empty string.
        
        Note: Kindly refrain from responding to user messages directly. Instead, focus on correcting any grammatical issues where applicable.
    """,
    description="This agent corrects grammar mistakes in text provided by arabic government professionals, explains the errors in simple terms, and returns both the corrected text and the explanations.",
    model_client=other_client,
    tools=[check_grammar_tool]
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=5)
handoff_termination = HandoffTermination(target="user")
stop_message_termination = StopMessageTermination()
termination = text_mention_termination | stop_message_termination | handoff_termination | max_messages_termination

selector_prompt = """Select the most appropriate agent to handle the current message based on their specialties:

{roles}

Current conversation context:
{history}

Available agents: {participants}

Selection criteria:
- greeting_agent: For initial greetings and basic conversation
- policy_agent: For questions about labor and economic policies
- data_agent: For questions about economic data and statistics
- grammar_agent: For grammar corrections and explanations
- general_agent: For general knowledge questions not covered by other agents

Select exactly one agent based on the last message in the conversation.
""" + system_message


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> Optional[str]:
    if messages[-1].source != greeting_agent.name:
        return greeting_agent.name
    return None


team = SelectorGroupChat(
    participants=[policy_data_agent, greeting_agent, general_agent, grammar_agent],
    model_client=other_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    # selector_func=selector_func,
    allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)


async def main():
    result = await team.run(task=task)
    print("Conversation History:")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    print(f"Stop Reason: {result.stop_reason}")
    if (result.stop_reason.__contains__("TERMINATE")) & (result.messages[-1].content.__contains__("TERMINATE")):
        for message in result.messages[:-1][::-1]:
            if isinstance(message, TextMessage):
                final_response = message.content
                return final_response
    else:
        return ""


task = "What is your name?"

if __name__ == "__main__":
    # asyncio.run(Console(team.run_stream(task=task)))
    # Execute the asynchronous main function
    response = asyncio.run(main())
    print(response)
