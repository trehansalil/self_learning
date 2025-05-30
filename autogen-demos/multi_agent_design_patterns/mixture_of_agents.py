import asyncio

from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient

from multi_agent_design_patterns.agent_class.base import WorkerAgent, OrchestratorAgent
from multi_agent_design_patterns.data_class.base import UserTask
from src.constants import LLAMA3_2_API_BASE, LLAMA3_2_API_KEY, LLAMA3_2_API_MODEL_NAME, QWEN_API_MODEL_NAME, \
    QWEN_API_KEY, QWEN_API_BASE, ModelInfo, ModelFamily


async def main(task):
    runtime = SingleThreadedAgentRuntime()
    model_client = OpenAIChatCompletionClient(
        model=QWEN_API_MODEL_NAME,
        api_key=QWEN_API_KEY,
        base_url=QWEN_API_BASE,
        model_info=ModelInfo(vision=False, function_calling=True, family=ModelFamily.ANY, json_output=True, structured_output=True),
    )
    await WorkerAgent.register(runtime, "worker", lambda: WorkerAgent(model_client=model_client))
    await OrchestratorAgent.register(
        runtime,
        "orchestrator",
        lambda: OrchestratorAgent(model_client=model_client, worker_agent_types=["worker"] * 3, num_layers=3),
    )

    runtime.start()
    result = await runtime.send_message(UserTask(task=task), AgentId("orchestrator", "default"))

    await runtime.stop_when_idle()
    await model_client.close()

    print(f"{'-'*80}\nFinal result:\n{result.result}")

if __name__ == '__main__':
    query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    task = (
        "I have 432 cookies, and divide them 3:4:2 between Alice, Bob, and Charlie. How many cookies does each person get?",
        query
    )

    asyncio.run(main(task))