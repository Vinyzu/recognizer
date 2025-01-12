from __future__ import annotations

import asyncio

from patchright.async_api import async_playwright

from recognizer.agents.playwright import AsyncChallenger


async def bytedance():
    # patchright install chromium
    # patchright install-deps chromium
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./user_data",
            channel="chrome",
            headless=False,
            no_viewport=True,
            locale="en-US",
        )
        page = await context.new_page()
        challenger = AsyncChallenger(page)

        await page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")
        await challenger.solve_recaptcha()


if __name__ == "__main__":
    asyncio.run(bytedance())
