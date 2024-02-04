from __future__ import annotations

import asyncio

import botright

from recognizer.agents.playwright import AsyncChallenger


async def bytedance():
    # playwright install chromium
    # playwright install-deps chromium
    botright_client = await botright.Botright(headless=False)
    browser = await botright_client.new_browser()
    page = await browser.new_page()
    challenger = AsyncChallenger(page)

    await page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")
    await challenger.solve_recaptcha()
    await botright_client.close()


if __name__ == "__main__":
    asyncio.run(bytedance())
