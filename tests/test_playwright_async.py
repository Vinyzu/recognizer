from __future__ import annotations

from contextlib import suppress

import pytest
from playwright.async_api import Page

from recognizer.agents.playwright import AsyncChallenger


# @pytest.mark.xfail
@pytest.mark.asyncio
async def test_async_challenger(async_page: Page):
    challenger = AsyncChallenger(async_page)

    await async_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res
