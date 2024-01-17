# reCognizer v1.1
![Tests & Linting](https://github.com/Vinyzu/recognizer/actions/workflows/tests.yml/badge.svg)

#### reCognizer is a free-to-use AI based [reCaptcha](https://developers.google.com/recaptcha) Solver. <br> Usable with an easy-to-use API, also available for Async and Sync Playwright. <br> You can pass almost any format into the Challenger, from full-page screenshots, only-captcha images and no-border images to single images in a list.

#### Note: You Should use an undetected browser engine like [Botright](https://github.com/Vinyzu/Botright) to solve the Captchas consistently. <br>  reCaptcha detects normal Playwright easily and you probably wont get any successful solves despite correct recognitions.

## Install it from PyPI

```bash
pip install recognizer
```

---

## Basic Usage

```py
# Only for Type-Hints
from typing import TypeVar, Sequence, Union
from pathlib import Path
from os import PathLike

accepted_image_types = TypeVar("accepted_image_types", Path, Union[PathLike[str], str], bytes, Sequence[Path], Sequence[Union[PathLike[str], str]], Sequence[bytes])

# Real Code
from recognizer import Detector

detector = Detector()

task_type: str = "bicycle"
images: accepted_image_types = "recaptcha_image.png"
area_captcha: bool = False

response, coordinates = detector.detect(task_type, images, area_captcha=area_captcha)
```

---

## Playwright Usage
### Sync Playwright

```py
from playwright.sync_api import sync_playwright, Playwright
from recognizer.agents.playwright import SyncChallenger


def run(playwright: Playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()

    challenger = SyncChallenger(page)
    page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    challenger.solve_recaptcha()

    browser.close()


with sync_playwright() as playwright:
    run(playwright)
```


### Async Playwright

```py
import asyncio

from playwright.async_api import async_playwright, Playwright
from recognizer.agents.playwright import AsyncChallenger


async def run(playwright: Playwright):
    browser = await playwright.chromium.launch()
    page = await browser.new_page()

    challenger = AsyncChallenger(page)
    await page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    await challenger.solve_recaptcha()

    await browser.close()


async def main():
    async with async_playwright() as playwright:
        await run(playwright)


asyncio.run(main())
```
---

## Copyright and License
© [Vinyzu](https://github.com/Vinyzu/)

[GNU GPL](https://choosealicense.com/licenses/gpl-3.0/)

(Commercial Usage is allowed, but source, license and copyright has to made available. reCaptcha Challenger does not provide and Liability or Warranty)

---

## Thanks to

[QIN2DIM](https://github.com/QIN2DIM) (For basic project structure)

---

![Version](https://img.shields.io/badge/reCognizer-v1.1-blue)
![License](https://img.shields.io/badge/License-GNU%20GPL-green)
![Python](https://img.shields.io/badge/Python-v3.x-lightgrey)

[![my-discord](https://img.shields.io/badge/My_Discord-000?style=for-the-badge&logo=google-chat&logoColor=blue)](https://discordapp.com/users/935224495126487150)
[![buy-me-a-coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-000?style=for-the-badge&logo=ko-fi&logoColor=brown)](https://ko-fi.com/vinyzu)