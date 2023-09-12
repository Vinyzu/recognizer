# -*- coding: utf-8 -*-
# Author     : Vinyzu
# GitHub     : https://github.com/Vinyzu
# Description:
from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
import botright
from recaptcha_challenger.agents.playwright import AsyncChallenger


async def bytedance():
    # playwright install chromium
    # playwright install-deps chromium
    botright_client = await botright.Botright(headless=False)
    browser = await botright_client.new_browser()
    page = await browser.new_page()
    challenger = AsyncChallenger(page)

    await page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")
    await challenger.solve_recaptcha()

if __name__ == "__main__":
    asyncio.run(bytedance())