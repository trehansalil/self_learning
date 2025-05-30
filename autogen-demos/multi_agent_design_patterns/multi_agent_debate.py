import asyncio

from autogen_core import (
    DefaultTopicId,
    TypeSubscription,
    SingleThreadedAgentRuntime,
)

from autogen_ext.models.openai import OpenAIChatCompletionClient

from multi_agent_design_patterns.agent_class.base import MathSolver, MathAggregator
from multi_agent_design_patterns.data_class.base import Question
from src.constants import LLAMA3_2_API_BASE, LLAMA3_2_API_KEY, LLAMA3_2_API_MODEL_NAME, QWEN_API_MODEL_NAME, \
    QWEN_API_KEY, QWEN_API_BASE


async def main(query):
    runtime = SingleThreadedAgentRuntime()


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

    await MathSolver.register(
        runtime,
        "MathSolverA",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverA",
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathSolver.register(
        runtime,
        "MathSolverB",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverB",
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathSolver.register(
        runtime,
        "MathSolverC",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverC",
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathSolver.register(
        runtime,
        "MathSolverD",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverD",
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathAggregator.register(runtime, "MathAggregator", lambda: MathAggregator(num_solvers=4))

    # Subscriptions for topic published to by MathSolverA.
    await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverD"))
    await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverB"))

    # Subscriptions for topic published to by MathSolverB.
    await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverA"))
    await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverC"))

    # Subscriptions for topic published to by MathSolverC.
    await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverB"))
    await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverD"))

    # Subscriptions for topic published to by MathSolverD.
    await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverC"))
    await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverA"))

    # All solvers and the aggregator subscribe to the default topic.

    runtime.start()
    await runtime.publish_message(Question(content=query), DefaultTopicId())
    # Wait for the runtime to stop when idle.
    await runtime.stop_when_idle()
    # Close the connection to the model client.
    await model_client.close()

if __name__ == '__main__':
    query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    asyncio.run(main(query))