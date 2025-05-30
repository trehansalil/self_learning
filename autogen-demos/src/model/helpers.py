from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.ui import Console


async def run_team_stream(team, task, model_client) -> None:
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]

    await model_client.close()
