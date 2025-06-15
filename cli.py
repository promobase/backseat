import asyncio
from dataclasses import dataclass
from pathlib import Path

from agents import Agent, run_demo_loop
from agents.mcp import MCPServer, MCPServerStdio

from backseat.utils import load_dotenv

load_dotenv()

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
    }
}


@dataclass
class BrowserContext:
    pass


async def run(mcp_server: MCPServer):
    agent = Agent[BrowserContext](
        name="Browser Automation Agent",
        instructions="Use tools to automate browser tasks. If you need user input, use the `need_user_input` tool. You will be decompose user irequests into smaller steps and ONLY use tools to complete them.",
        mcp_servers=[mcp_server],
    )
    await run_demo_loop(agent)
    # result = await Runner.run(agent, "What's Google's latest stock price?")
    # print(result.final_output)
    pass


async def main():
    async with MCPServerStdio(name="playwright", params=config["mcpServers"]["playwright"]) as mcp_server:
        await mcp_server.connect()
        await run(mcp_server)


if __name__ == "__main__":
    asyncio.run(main())
