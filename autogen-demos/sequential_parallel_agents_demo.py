# Create an OpenAI model client
from src.constants import LLAMA3_2_API_BASE, LLAMA3_2_API_KEY, LLAMA3_2_API_MODEL_NAME, QWEN_API_MODEL_NAME, QWEN_API_KEY, QWEN_API_BASE
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.base import Response, TaskResult
from typing_extensions import AsyncGenerator

from autogen_agentchat.ui._console import aprint

from autogen_core.models import ModelFamily, ModelInfo


model_client = OpenAIChatCompletionClient(
    model=QWEN_API_MODEL_NAME,
    api_key=QWEN_API_KEY,
    base_url=QWEN_API_BASE,
    model_info=ModelInfo(vision=False, function_calling=True, family=ModelFamily.ANY, json_output=True, structured_output=True),
)

# Create the writer agent
writer = AssistantAgent("writer", model_client=model_client, system_message="Draft a short paragraph on climate change.")

# Create the reviewer agent
reviewer = AssistantAgent("reviewer", model_client=model_client, system_message="Review the draft and suggest improvements.")

manager = AssistantAgent("manager", model_client=model_client, system_message="Finalizes the draft with reviewers suggested improvements. Remove all BS and handover the final response to the User")

# Build the graph
builder = DiGraphBuilder()
builder.add_node(writer).add_node(reviewer).add_node(manager)
builder.add_edge(writer, reviewer)
builder.add_edge(reviewer, manager)
# builder.add_edge(manager, writer)

# Build and validate the graph
graph = builder.build()

# Create the flow

flow = GraphFlow([writer, reviewer, manager], graph=graph)

import asyncio


async def generator(event):
    yield event

async def main():
    stream = flow.run_stream(task="Write a short note on the impact of GDP on UAE economy.")
    async for event in stream:  # type: ignore
        print("\n\n")

        if isinstance(event, TaskResult):
            await Console(generator(event), output_stats=True)
        else:
            print(f"{event.source.capitalize()}:\n", end="", flush=True)
            print(event.content, end="", flush=True)
    event
if __name__ == "__main__":
    asyncio.run(main())
