import pytest
import pytest_asyncio
from patchright.async_api import async_playwright
from patchright.sync_api import sync_playwright

from recognizer import Detector


@pytest.fixture
def detector() -> Detector:
    detector = Detector()
    return detector


@pytest.fixture
def sync_playwright_object():
    with sync_playwright() as playwright_object:
        yield playwright_object


@pytest.fixture
def sync_browser(sync_playwright_object):
    browser = sync_playwright_object.chromium.launch_persistent_context(user_data_dir="./user_data", channel="chrome", headless=True, no_viewport=True, locale="en-US")

    yield browser
    browser.close()


@pytest.fixture
def sync_page(sync_browser):
    page = sync_browser.new_page()

    yield page
    page.close()


@pytest_asyncio.fixture
async def async_playwright_object():
    async with async_playwright() as playwright_object:
        yield playwright_object


@pytest_asyncio.fixture
async def async_browser(async_playwright_object):
    browser = await async_playwright_object.chromium.launch_persistent_context(user_data_dir="./user_data", channel="chrome", headless=True, no_viewport=True, locale="en-US")

    yield browser
    await browser.close()


@pytest_asyncio.fixture
async def async_page(async_browser):
    page = await async_browser.new_page()

    yield page
    await page.close()
