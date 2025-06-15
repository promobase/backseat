import os
from typing import Any

from agents import (
    Agent,
    AgentUpdatedStreamEvent,
    ItemHelpers,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    Runner,
    TResponseInputItem,
)
from agents.result import RunResultBase
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent


def load_dotenv(dotenv_path=".env"):
    """
    Loads environment variables from a .env file into os.environ.
    Each line in the .env file should be in KEY=VALUE format.
    Lines starting with # are treated as comments.
    """
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


async def run_loop(agent: Agent[Any], *, stream: bool = True) -> None:  # noqa: C901
    """Run a simple REPL loop with the given agent.

    This utility allows quick manual testing and debugging of an agent from the
    command line. Conversation state is preserved across turns. Enter ``exit``
    or ``quit`` to stop the loop.

    Args:
        agent: The starting agent to run.
        stream: Whether to stream the agent output.
    """

    current_agent = agent
    input_items: list[TResponseInputItem] = []
    while True:
        try:
            user_input = input(" > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        input_items.append({"role": "user", "content": user_input})

        result: RunResultBase

        try:
            if stream:
                result = Runner.run_streamed(current_agent, input=input_items)
                async for event in result.stream_events():
                    if isinstance(event, RawResponsesStreamEvent):
                        if isinstance(event.data, ResponseTextDeltaEvent):
                            print(event.data.delta, end="", flush=True)
                    elif isinstance(event, RunItemStreamEvent):
                        if event.item.type == "tool_call_item":
                            print("\n[tool called]", flush=True)
                        elif event.item.type == "tool_call_output_item":
                            print(f"\n[tool output: {event.item.output}]", flush=True)
                        elif event.item.type == "message_output_item":
                            message = ItemHelpers.text_message_output(event.item)
                            print(f"\n{message}", end="", flush=True)
                    elif isinstance(event, AgentUpdatedStreamEvent):
                        print(f"\n[Agent updated: {event.new_agent.name}]", flush=True)
                print()
            else:
                result = await Runner.run(current_agent, input_items)
                if result.final_output is not None:
                    print(result.final_output)
        except Exception as e:
            error_msg = f"[Error]: {e}"
            input_items.append({"role": "system", "content": error_msg})
            print(error_msg)
            continue

        current_agent = result.last_agent
        input_items = result.to_input_list()
