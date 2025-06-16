import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, TContext, function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.mcp import MCPServerStdio

from backseat.utils import get_logger, load_dotenv, run_loop

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
                "github:promobase/playwright-mcp@latest",
                "--cdp-endpoint",
                "http://localhost:9222",
                # "--config",
                # str(playwrightmcp_config_path),
            ],
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
async def main():
    logger = get_logger("playwrightmcp.cli.main")
    logger.info("CLI main started.")

    try:
        logger.info("Initializing Playwright MCP client (npx @playwright/mcp@latest)...")
        async with (
            MCPServerStdio(name="playwright", params=config["mcpServers"]["playwright"]) as playwright_server,
            # MCPServerStdio(
            #     name="local-playwright",
            #     params={
            #         "command": "uv",
            #         "args": ["run", "backseat/server.py"],
            #     },
            # ) as local_playwright_server,
        ):
            # await local_playwright_server.connect()
            # NOTE: let's try not to use local version -- instead we extend the MS playwright server capabilities.
            await playwright_server.connect()
            logger.info("Successfully connected to Playwright MCP client (npx).")

            openai_client = AsyncOpenAI()
            model_settings = ModelSettings(temperature=0)
            browser_agent = Agent(
                name="Browser Automation Agent",
                instructions=BROWSER_AGENT_INSTRUCTIONS,
                mcp_servers=[playwright_server],
                tools=[wait_for],
                model=OpenAIChatCompletionsModel(
                    "gpt-4.1-mini",
                    openai_client=openai_client,
                ),
                model_settings=model_settings,
            )
            logger.info("Starting agent run_loop...")
            await run_loop(browser_agent)
            logger.info("Agent run_loop finished.")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
