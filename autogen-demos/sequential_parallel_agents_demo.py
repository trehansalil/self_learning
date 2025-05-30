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

model_client1 = OpenAIChatCompletionClient(
    model=LLAMA3_2_API_MODEL_NAME,
    api_key=LLAMA3_2_API_KEY,
    base_url=LLAMA3_2_API_BASE,
    model_info=ModelInfo(vision=False, function_calling=True, family=ModelFamily.ANY, json_output=True, structured_output=True),
)

# Create the writer agent
writer = AssistantAgent("writer", model_client=model_client1, system_message="Draft a short paragraph on climate change.")

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

# Create the writer agent
writer = AssistantAgent("writer", model_client=model_client1, system_message="Draft a short paragraph on climate change.")

# Create two editor agents
editor1 = AssistantAgent("labour_market_expert", model_client=model_client, system_message="Edit the paragraph for speaking from the labour market purview. Also, share your Knowledge cut-off")

editor2 = AssistantAgent("political_economic_Expert", model_client=model_client, system_message="Edit the paragraph for speaking from the economic and political purview. Also, share your Knowledge cut-off")

# Create the final reviewer agent
final_reviewer = AssistantAgent(
    "UAE_King",
    model_client=model_client,
    system_message="Consolidate the inputs and write a royal outlook on the topic based on the expert opinion and self interest.",
)

# Build the workflow graph
builder = DiGraphBuilder()
builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(final_reviewer)

# Fan-out from writer to editor1 and editor2
builder.add_edge(writer, editor1)
builder.add_edge(writer, editor2)

# Fan-in both editors into final reviewer
builder.add_edge(editor1, final_reviewer)
builder.add_edge(editor2, final_reviewer)

# Build and validate the graph
graph = builder.build()

# Create the flow
flow2 = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
)

import asyncio


async def generator(event):
    yield event

async def main(query):
    stream = flow.run_stream(task=query)
    async for event in stream:  # type: ignore
        print("\n\n")

        if isinstance(event, TaskResult):
            await Console(generator(event), output_stats=True)
        else:
            print(f"{event.source.capitalize()}:\n", end="", flush=True)
            print(event.content, end="", flush=True)

async def main2(query):
    await Console(flow2.run_stream(task=query))
if __name__ == "__main__":
    query = "How does Inflation impact the Growth of expat individuals in the UAE Economy."
    # asyncio.run(main(query))
    asyncio.run(main2(query))
