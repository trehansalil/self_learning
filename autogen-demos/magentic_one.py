import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console

from autogen_core.models import ModelFamily, ModelInfo

from src.constants import LLAMA3_2_API_BASE, LLAMA3_2_API_KEY, LLAMA3_2_API_MODEL_NAME


async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model=LLAMA3_2_API_MODEL_NAME,
        api_key=LLAMA3_2_API_KEY,
        base_url=LLAMA3_2_API_BASE, 
        model_info=ModelInfo(vision=False, function_calling=True, family=ModelFamily.ANY, json_output=True, structured_output=True))

    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(team.run_stream(task="Provide a different proof for Fermat's Last Theorem"))
    await model_client.close()


asyncio.run(main())