from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

from playwright.async_api import async_playwright, Page as AsyncPage
from playwright.sync_api import sync_playwright, Page as SyncPage
from recognizer import Detector


@pytest.fixture
def detector() -> Detector:
    detector = Detector()
    return detector


@pytest.fixture
def sync_page() -> Generator[SyncPage, None, None]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(locale="en-US")
        page = context.new_page()

        yield page


@pytest_asyncio.fixture
async def async_page() -> AsyncGenerator[AsyncPage, None]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale="en-US")
        page = await context.new_page()

        yield page
