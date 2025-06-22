import code
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_core import code_executor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.teams import RoundRobinGroupChat

from autogen_agentchat.conditions import TextMentionTermination

import asyncio

from src.constants import EURI_API_BASE_URL, EURI_API_KEY, EURI_MODEL_NAME_GPT_4_1_NANO

async def main():
    model = OpenAIChatCompletionClient(
        model=EURI_MODEL_NAME_GPT_4_1_NANO,
        api_key=EURI_API_KEY,
        base_url=EURI_API_BASE_URL
    )

    planner = AssistantAgent(
        name='Plnner',
        model_client=model,
        system_message=("You're a planner agent. You will be given a excel data file"
        "and a question about it. You can develop python code to answer the question"
        "Your job is to plan the steps to answer the question and then you should along with any linked dependencies"
        "provide the plan to the developer agent"
        )
    )

    developer = AssistantAgent(
        name='Developer',
        model_client=model,
        system_message=("You're a developer agent. You will be given a excel data file"
        "and a question about it. You can develop python code to answer the question"
        "You should always begin with you plan to answer your question then you"
        "should write the code to answer the question"
        "You should always write the code in code blocks with language with Python language only"
        "if you need several code blocks, write them one at a time"
        "You will be working with a code executor agent to execute the code. If the code"
        "is executed successfully, you can continue."
        "If a library is required, you should install it using pip install <library_name>"
        "Once you have the code execution results, you should provide the final answer and"
        "After that you should exactly say TERMINATE to terminate the conversation at the end"
        "If the executed code output is in markdown then display it as it is, no need to format it within ```markdown```"
        )
    )

    docker = DockerCommandLineCodeExecutor(
        image='custom-python-3.10',
        work_dir="temp",
    )

    await docker.start()

    executor = CodeExecutorAgent(
        name="CodeExecutor",
        code_executor=docker,
    )

    # Create a group chat with the agents
    team = RoundRobinGroupChat(
        participants=[planner, developer, executor],
        termination_condition=TextMentionTermination(text="TERMINATE"),
        max_turns=35,
    )

    task = "My dataset is Active Worker Per Occupation Level.xlsx. What are the columns in the dataset?"
    task = "My dataset is Active Worker Per Occupation Level.xlsx. Is this a case of multi-level columns? If yes, then what would be the levels of columns? What do they represent? cAn you please rename this to just 1 level for simplicity? Show me the data in markdown format in the dataset after the renaming"

    async for msg in team.run_stream(task = task):
        if isinstance(msg, TextMessage):
            print(f"{msg.source}: {msg.content}")
        elif isinstance(msg, TaskResult):
            print(f"Stop Reason: {msg.stop_reason}")

    await docker.stop()

if __name__ == "__main__":
    asyncio.run(main())