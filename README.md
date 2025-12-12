<h1 align="center">
    🎭 reCognizer
</h1>


<p align="center">
    <a href="https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python/releases/latest">
        <img alt="Patchright Version" src="https://img.shields.io/github/v/release/microsoft/playwright-python?display_name=release&label=Version">
    </a>
    <a href="https://github.com/Vinyzu/recognizer/actions">
        <img src="https://github.com/Vinyzu/recognizer/actions/workflows/ci.yml/badge.svg">
    </a>
    <a href="https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python/releases">
        <img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/pepy/dt/patchright?color=seagreen">
    </a>
    <a href="https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-GNU%20GPL-green">
    </a>
</p>

#### reCognizer is a free-to-use AI based [reCaptcha](https://developers.google.com/recaptcha) Solver. <br> Usable with an easy-to-use API, also available for Async and Sync Playwright. <br> You can pass almost any format into the Challenger, from full-page screenshots, only-captcha images and no-border images to single images in a list.

#### Note: You Should use an undetected browser engine like [Patchright](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python) or [Botright](https://github.com/Vinyzu/Botright) to solve the Captchas consistently. <br>  reCaptcha detects normal Playwright easily and you probably wont get any successful solves despite correct recognitions.

---
<details open>
    <summary><h3>Sponsors</h1></summary>

<a href="https://www.thordata.com/?ls=github&lk=Vinyzu" target="_blank">
  <img alt="Thordata Banner" src="https://github.com/user-attachments/assets/58460bff-b057-4191-88da-80e0bd29b745" width="90%" />
</a>

[Thordata](https://www.thordata.com/?ls=github&lk=Vinyzu) - Your First Plan is on Us! 💰Get 100% of your first residential proxy purchase back as wallet balance, up to $900.
### **⚡ Why Thordata?**
🌍 190+ real residential & ISP IP locations\
🔐 Fully encrypted, ultra-secure connections\
🚀 Optimized for web scraping, ad verification & automation workflows

🔥Don’t wait — this is your **best time to start** with [Thordata](https://www.thordata.com/?ls=github&lk=Vinyzu) and experience the safest, fastest proxy network.

</details>

---

## Install it from PyPI

```bash
pip install recognizer
```

---

## Examples

### Possible Image Inputs
![Accepted Formats](https://i.ibb.co/nztTD9Z/formats.png)

### Example Solve Video (Good IP & Botright)
https://github.com/Vinyzu/recognizer/assets/50874994/95a713e3-bb46-474b-994f-cb3dacae9279

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

detector = Detector(optimize_click_order=True)

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

    challenger = SyncChallenger(page, click_timeout=1000)
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

    challenger = AsyncChallenger(page, click_timeout=1000)
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
<br>
[GNU GPL](https://choosealicense.com/licenses/gpl-3.0/)

(Commercial Usage is allowed, but source, license and copyright has to made available. reCaptcha Challenger does not provide and Liability or Warranty)

---

## Projects/AIs Used
[YOLO11m-seg](https://github.com/ultralytics/ultralytics)
<br>
[flavour/CLIP ViT-L/14](https://huggingface.co/flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K)
<br>
[CIDAS/clipseg](https://huggingface.co/CIDAS/clipseg-rd64-refined)
[]()

## Thanks to

[QIN2DIM](https://github.com/QIN2DIM) (For basic project structure)

---

## Disclaimer

This repository is provided for **educational purposes only**. \
No warranties are provided regarding accuracy, completeness, or suitability for any purpose. **Use at your own risk**—the authors and maintainers assume **no liability** for **any damages**, **legal issues**, or **warranty breaches** resulting from use, modification, or distribution of this code.\
**Any misuse or legal violations are the sole responsibility of the user**. 

---

![Version](https://img.shields.io/pypi/v/reCognizer?display_name=release&label=reCognizer)
![License](https://img.shields.io/badge/License-GNU%20GPL-green)
![Python](https://img.shields.io/badge/Python-v3.x-lightgrey)

[![my-discord](https://img.shields.io/badge/My_Discord-000?style=for-the-badge&logo=google-chat&logoColor=blue)](https://discordapp.com/users/935224495126487150)
[![buy-me-a-coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-000?style=for-the-badge&logo=ko-fi&logoColor=brown)](https://ko-fi.com/vinyzu)
