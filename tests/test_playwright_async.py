from __future__ import annotations

from contextlib import suppress

import pytest
from playwright.async_api import Page

from recognizer.agents.playwright import AsyncChallenger


@pytest.mark.asyncio
async def test_async_challenger(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res

@pytest.mark.asyncio
async def test_async_challenger1(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res