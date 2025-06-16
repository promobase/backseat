import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from playwright.async_api import Browser, BrowserContext, Page


def get_logger(name: str) -> logging.Logger:
    """Creates and configures a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# --- Strongly-Typed Application Context ---
# We define a dataclass to hold all the state that the lifespan manager will create.
# This makes accessing state in your tools type-safe and explicit.
@dataclass
class PlaywrightContext:
    """A container for all managed Playwright resources."""

    browser: Browser
    context: BrowserContext
    page: Page
    config: dict[str, Any]
    logger: logging.Logger
    cdp_url: str


def load_config() -> dict[str, Any]:
    """Load configuration from playwrightmcp.json."""
    # Note: This code expects a 'config/playwrightmcp.json' file relative to this script.
    # Example playwrightmcp.json:
    # {
    #   "browser": {
    #     "launchOptions": {
    #       "headless": true,
    #       "args": []
    #     },
    #     "contextOptions": {},
    #     "userDataDir": "./user_data"
    #   },
    #   "outputDir": "./output"
    # }
    config_path = Path(__file__).parent / "config" / "playwrightmcp.json"
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}. Using empty config.")
        return {}
    with open(config_path) as f:
        return json.load(f)


# --- Lifespan Context Manager ---
@asynccontextmanager
async def playwright_lifespan(server: FastMCP) -> AsyncIterator[PlaywrightContext]:
    """
    Manages the Playwright browser lifecycle.
    On startup: launches the browser and creates a page.
    On shutdown: gracefully closes the browser.
    """
    # Import Playwright types here to avoid circular dependency issues if they were at top-level
    import sys

    from playwright.async_api import async_playwright

    logger = get_logger(__name__)
    p = None
    browser = None

    # --- Startup Logic ---
    try:
        logger.info("Application starting up: Initializing Playwright...")

        config = load_config()
        browser_config = config.get("browser", {})
        launch_options = browser_config.get("launchOptions", {})
        context_options = browser_config.get("contextOptions", {})

        cdp_port = browser_config.get("cdp_port", 9222)
        args = launch_options.get("args", [])
        if not any(f"--remote-debugging-port={cdp_port}" in s for s in args):
            args.append(f"--remote-debugging-port={cdp_port}")
        launch_options["args"] = args

        user_data_dir_path = browser_config.get("userDataDir")
        p = await async_playwright().start()
        if user_data_dir_path:
            user_data_dir = Path(user_data_dir_path)
            user_data_dir.mkdir(parents=True, exist_ok=True)
            # launch_persistent_context returns a context, not a browser
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir), **launch_options, **context_options
            )
            browser = context.browser
        else:
            browser = await p.chromium.launch(**launch_options)
            context = await browser.new_context(**context_options)

        page = await context.new_page()
        cdp_url = f"http://localhost:{cdp_port}"
        logger.info(f"Browser launched with headless: {launch_options.get('headless', False)}")
        logger.info(f"CDP session available at: {cdp_url}")

        app_context = PlaywrightContext(
            browser=browser,
            context=context,
            page=page,
            config=config,
            logger=logger,
            cdp_url=cdp_url,
        )

        logger.info("About to yield app_context to application.")
        yield app_context
        logger.info("Returned from yield, application is shutting down.")

    except Exception as e:
        logger.error(f"Exception during startup: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # --- Shutdown Logic ---
        logger.info("Application shutting down: Closing browser...")
        if browser:
            await browser.close()
        if p:
            await p.stop()
        logger.info("Playwright shutdown complete.")


# --- Application Setup ---
mcp = FastMCP("Playwright-MCP-App", lifespan=playwright_lifespan, dependencies=["fastmcp", "playwright"])


def get_ctx() -> PlaywrightContext:
    """Helper function to retrieve the type-safe lifespan context from the app."""
    ctx = mcp.get_context()
    return ctx.request_context.lifespan_context


#  ---- tools ----
@mcp.tool()
async def navigate_to(url: str) -> str:
    """Navigate the browser to a specific URL."""
    ctx = get_ctx()
    try:
        await ctx.page.goto(url)
        ctx.logger.info(f"Successfully navigated to {url}")
        return f"Navigated to {url}"  # noqa: TRY300
    except Exception as e:
        ctx.logger.exception(f"Failed to navigate to {url}: {e}")
        return f"Error: Could not navigate to {url}. Reason: {e}"


@mcp.tool()
async def get_page_title() -> str:
    """Get the current page title."""
    ctx = get_ctx()
    title = await ctx.page.title()
    return f"Current page title is: '{title}'"


@mcp.tool()
async def screenshot(filename: str = "screenshot.png") -> str:
    """Take a screenshot of the current page."""
    ctx = get_ctx()
    output_dir = Path(ctx.config.get("outputDir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = output_dir / filename
    await ctx.page.screenshot(path=str(screenshot_path))
    ctx.logger.info(f"Screenshot saved to {screenshot_path}")
    return f"Screenshot saved to {screenshot_path.resolve()}"


@mcp.tool()
def get_browser_status() -> str:
    """Returns the connection status and CDP URL of the managed browser."""
    ctx = get_ctx()
    status = "connected" if ctx.browser.is_connected() else "disconnected"
    return f"Browser status: {status}\nCDP URL: {ctx.cdp_url}"


if __name__ == "__main__":
    mcp.run()
