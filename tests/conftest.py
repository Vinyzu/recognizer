from typing import AsyncGenerator, Generator

import botright
import pytest
import pytest_asyncio
from botright.extended_typing import BrowserContext as BotrightBrowser
from botright.extended_typing import Page as BotrightPage
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import sync_playwright

from recognizer import Detector


@pytest.fixture
def detector() -> Detector:
    detector = Detector()
    return detector


@pytest.fixture
def sync_page() -> Generator[SyncPage, None, None]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(locale="en-US")
        page = context.new_page()

        yield page


@pytest_asyncio.fixture
async def async_page() -> AsyncGenerator[AsyncPage, None]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        page = await context.new_page()

        yield page


@pytest_asyncio.fixture
async def botright_client():
    botright_client = await botright.Botright(headless=True)
    yield botright_client
    await botright_client.close()


@pytest_asyncio.fixture
async def browser(botright_client, **launch_arguments) -> BotrightBrowser:
    browser = await botright_client.new_browser(**launch_arguments)
    yield browser
    await browser.close()


@pytest_asyncio.fixture
async def botright_page(browser) -> BotrightPage:
    page = await browser.new_page()
    yield page
    await page.close()
