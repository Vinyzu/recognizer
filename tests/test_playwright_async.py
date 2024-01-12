from __future__ import annotations

import pytest

from playwright.async_api import Page
from recognizer.agents.playwright import AsyncChallenger


@pytest.mark.asyncio
@pytest.mark.xfail
async def test_async_challenger(async_page: Page):
    challenger = AsyncChallenger(async_page)

    await async_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")
    res = await challenger.solve_recaptcha()
    assert res
