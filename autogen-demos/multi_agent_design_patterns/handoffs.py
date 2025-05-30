import asyncio
import uuid

from autogen_core import (
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from autogen_core.models import (
    SystemMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from multi_agent_design_patterns.agent_class.base import AIAgent, HumanAgent, UserAgent
from multi_agent_design_patterns.data_class.base import UserLogin
from src.constants import QWEN_API_MODEL_NAME, QWEN_API_BASE, QWEN_API_KEY, ModelFamily, ModelInfo


def execute_order(product: str, price: int) -> str:
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."


def look_up_item(search_query: str) -> str:
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id


def execute_refund(item_id: str, reason: str = "not provided") -> str:
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"


execute_order_tool = FunctionTool(execute_order, description="Price should be in USD.")
look_up_item_tool = FunctionTool(
    look_up_item, description="Use to find item ID.\nSearch query can be a description or keywords."
)
execute_refund_tool = FunctionTool(execute_refund, description="")

sales_agent_topic_type = "SalesAgent"
issues_and_repairs_agent_topic_type = "IssuesAndRepairsAgent"
triage_agent_topic_type = "TriageAgent"
human_agent_topic_type = "HumanAgent"
user_topic_type = "User"

def transfer_to_sales_agent() -> str:
    return sales_agent_topic_type


def transfer_to_issues_and_repairs() -> str:
    return issues_and_repairs_agent_topic_type


def transfer_back_to_triage() -> str:
    return triage_agent_topic_type


def escalate_to_human() -> str:
    return human_agent_topic_type


transfer_to_sales_agent_tool = FunctionTool(
    transfer_to_sales_agent, description="Use for anything sales or buying related."
)
transfer_to_issues_and_repairs_tool = FunctionTool(
    transfer_to_issues_and_repairs, description="Use for issues, repairs, or refunds."
)
transfer_back_to_triage_tool = FunctionTool(
    transfer_back_to_triage,
    description="Call this if the user brings up a topic outside of your purview,\nincluding escalating to human.",
)
escalate_to_human_tool = FunctionTool(escalate_to_human, description="Only call this if explicitly asked to.")


async def main(task):
    runtime = SingleThreadedAgentRuntime()

    model_client = OpenAIChatCompletionClient(
        model=QWEN_API_MODEL_NAME,
        api_key=QWEN_API_KEY,
        base_url=QWEN_API_BASE,
        model_info=ModelInfo(vision=False, function_calling=True, family=ModelFamily.ANY, json_output=True, structured_output=True),
    )

    # Register the triage agent.
    triage_agent_type = await AIAgent.register(
        runtime,
        type=triage_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description="A triage agent.",
            system_message=SystemMessage(
                content="You are a customer service bot for ACME Inc. "
                        "Introduce yourself. Always be very brief. "
                        "Gather information to direct the customer to the right department. "
                        "But make your questions subtle and natural."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[
                transfer_to_issues_and_repairs_tool,
                transfer_to_sales_agent_tool,
                escalate_to_human_tool,
            ],
            agent_topic_type=triage_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    # Add subscriptions for the triage agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(
        TypeSubscription(topic_type=triage_agent_topic_type, agent_type=triage_agent_type.type))

    # Register the sales agent.
    sales_agent_type = await AIAgent.register(
        runtime,
        type=sales_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description="A sales agent.",
            system_message=SystemMessage(
                content="You are a sales agent for ACME Inc."
                        "Always answer in a sentence or less."
                        "Follow the following routine with the user:"
                        "1. Ask them about any problems in their life related to catching roadrunners.\n"
                        "2. Casually mention one of ACME's crazy made-up products can help.\n"
                        " - Don't mention price.\n"
                        "3. Once the user is bought in, drop a ridiculous price.\n"
                        "4. Only after everything, and if the user says yes, "
                        "tell them a crazy caveat and execute their order.\n"
                        ""
            ),
            model_client=model_client,
            tools=[execute_order_tool],
            delegate_tools=[transfer_back_to_triage_tool],
            agent_topic_type=sales_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    # Add subscriptions for the sales agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(
        TypeSubscription(topic_type=sales_agent_topic_type, agent_type=sales_agent_type.type))

    # Register the issues and repairs agent.
    issues_and_repairs_agent_type = await AIAgent.register(
        runtime,
        type=issues_and_repairs_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description="An issues and repairs agent.",
            system_message=SystemMessage(
                content="You are a customer support agent for ACME Inc."
                        "Always answer in a sentence or less."
                        "Follow the following routine with the user:"
                        "1. First, ask probing questions and understand the user's problem deeper.\n"
                        " - unless the user has already provided a reason.\n"
                        "2. Propose a fix (make one up).\n"
                        "3. ONLY if not satisfied, offer a refund.\n"
                        "4. If accepted, search for the ID and then execute refund."
            ),
            model_client=model_client,
            tools=[
                execute_refund_tool,
                look_up_item_tool,
            ],
            delegate_tools=[transfer_back_to_triage_tool],
            agent_topic_type=issues_and_repairs_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    # Add subscriptions for the issues and repairs agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(
        TypeSubscription(topic_type=issues_and_repairs_agent_topic_type, agent_type=issues_and_repairs_agent_type.type)
    )

    # Register the human agent.
    human_agent_type = await HumanAgent.register(
        runtime,
        type=human_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: HumanAgent(
            description="A human agent.",
            agent_topic_type=human_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    # Add subscriptions for the human agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(
        TypeSubscription(topic_type=human_agent_topic_type, agent_type=human_agent_type.type))

    # Register the user agent.
    user_agent_type = await UserAgent.register(
        runtime,
        type=user_topic_type,
        factory=lambda: UserAgent(
            description="A user agent.",
            user_topic_type=user_topic_type,
            agent_topic_type=triage_agent_topic_type,  # Start with the triage agent.
        ),
    )
    # Add subscriptions for the user agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))

    # Start the runtime.
    runtime.start()

    # Create a new session for the user.
    session_id = str(uuid.uuid4())
    await runtime.publish_message(UserLogin(), topic_id=TopicId(user_topic_type, source=session_id))

    # Run until completion.
    await runtime.stop_when_idle()
    await model_client.close()

if __name__ == '__main__':
    query = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    task = (
        "I have 432 cookies, and divide them 3:4:2 between Alice, Bob, and Charlie. How many cookies does each person get?",
        query
    )

    asyncio.run(main(task))