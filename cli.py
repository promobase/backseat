import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, TContext, function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.mcp import MCPServer, MCPServerStdio

from backseat.utils import load_dotenv, run_loop

load_dotenv()
"""
design: automate browser tasks on social platforms & ads.
High level goals:
1. login flow (e.g. login to Facebook, Google, etc.), user input needed
2. x-plat content creation API (e.g. posting to Xiaohongshu, FB, IG, Tiktok, etc)
3. x-plat ads creation API & mgmt

architecture:
using playwright-mcp for baseline tools for llms to access browser automation. We will also develop more fine-tuned mcp tools, e.g. with bs4, for example, to enhance browsing capabilities. next, using openai agents sdk for planning & orchestrating complex user flows.

progress:
1. now, we focus on baseline implementations
"""

playwrightmcp_config_path = Path(__file__).parent / "backseat" / "config" / "playwrightmcp.json"

config = {
    "mcpServers": {
        "playwright": {
            "command": "npx",
            "args": [
                "@playwright/mcp@latest",
                "--config",
                str(playwrightmcp_config_path),
            ],
        },
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
    }
}

PLANNER_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}
You're expert in understanding user requests specifically on web browser automations. You specialize in social media platforms & 2B ads platforms. You will be given a user input, decompose it into smaller steps, use your tools and triaging agents(each with browser automation capabilities) to complete the user requests.

RULES:
1. for each step, think out loud on what's the sequence of actions needed.
"""
BROWSER_AGENT_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}
You're expert in understanding user requests specifically on web browser automations. You specialize in social media platforms & 2B ads platforms. You will be given a user input, decompose it into smaller steps, use your tools and triaging agents(each with browser automation capabilities) to complete the user requests.
Use tools to automate browser tasks. You will be decompose user irequests into smaller steps and ONLY use tools to complete them.
RULES:
1. for each step, think out loud on what's the sequence of actions needed.
2. use wait_for tools in case certain sites/actions are slow to respond. Especially for browser automation tools, you might see timeout errors, it's common. just retry it
3. for each step, you should check for current browser state, e.g. url, snapshot, etc, and then decide what to do next.
"""


# ---- tools ----
@function_tool
async def wait_for(seconds: int) -> str:
    """
    Wait for a specified number of seconds.
    """
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds."


@dataclass
class TBrowserContext:
    pass


@dataclass
class BaseBrowserAgent(Agent, Generic[TContext]):
    pass


#  ---- agents ----
async def run(mcp_server: MCPServer):
    openai_client = AsyncOpenAI()
    browser_agent = Agent(
        name="Browser Automation Agent",
        instructions=BROWSER_AGENT_INSTRUCTIONS,
        mcp_servers=[mcp_server],
        tools=[wait_for],
        model=OpenAIChatCompletionsModel(
            "gpt-4.1-mini",
            openai_client=openai_client,
        ),
    )
    _planner_agent = Agent(
        name="Browser Planning Agent",
        instructions=PLANNER_INSTRUCTIONS,
        handoffs=[browser_agent],
        model=OpenAIChatCompletionsModel(
            "o4-mini",
            openai_client=openai_client,
        ),
    )
    await run_loop(browser_agent)
    # result = await Runner.run(agent, "What's Google's latest stock price?")
    # print(result.final_output)
    pass


async def main():
    async with (
        MCPServerStdio(name="playwright", params=config["mcpServers"]["playwright"]) as playwright_server,
        MCPServerStdio(name="fetch", params=config["mcpServers"]["fetch"]) as fetch_server,
    ):
        await asyncio.gather(playwright_server.connect(), fetch_server.connect())
        await run(playwright_server)


if __name__ == "__main__":
    asyncio.run(main())
