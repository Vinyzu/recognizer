from __future__ import annotations

from contextlib import suppress
from typing import Optional, Union

from recognizer import Detector

from playwright.async_api import Page, FrameLocator, Request, TimeoutError


class AsyncChallenger:
    def __init__(self, page: Page, click_timeout: Optional[int] = None) -> None:
        self.page = page
        self.detector = Detector()

        self.click_timeout = click_timeout
        self.dynamic = False

        self.page.on('request', self.request_handler)

    async def request_handler(self, request: Request) -> None:
        if request.url.startswith("https://www.google.com/recaptcha/"):
            if "reload" in request.url or "userverify" in request.url:
                with suppress(Exception):
                    response = await request.response()
                    assert response

                    self.dynamic = "dynamic" in await response.text()

    async def click_checkbox(self) -> bool:
        # Clicking Captcha Checkbox
        try:
            checkbox = self.page.frame_locator("iframe[title='reCAPTCHA']")
            await checkbox.locator(".recaptcha-checkbox-border").click()
            return True
        except TimeoutError:
            print("[ERROR] Could not click reCaptcha Checkbox.")
            return False

    async def check_result(self) -> Union[str, None]:
        with suppress(ReferenceError):
            captcha_token: str = await self.page.evaluate("grecaptcha.getResponse()")
            return captcha_token

        with suppress(ReferenceError):
            enterprise_captcha_token: str = await self.page.evaluate("grecaptcha.enterprise.getResponse()")
            return enterprise_captcha_token

        return None

    async def detect_tiles(self, prompt: str, area_captcha: bool, verify: bool, captcha_frame: FrameLocator) -> Union[str, bool]:
        image = await self.page.screenshot()
        response, coordinates = self.detector.detect(prompt, image, area_captcha=area_captcha)

        if not any(response):
            if not verify:
                print("[ERROR] Detector did not return any Results.")
                return await self.retry(captcha_frame, verify=verify)
            else:
                return False

        for (coord_x, coord_y) in coordinates:
            await self.page.mouse.click(coord_x, coord_y)
            if self.click_timeout:
                await self.page.wait_for_timeout(self.click_timeout)

        return True

    async def retry(self, captcha_frame, verify=False) -> Union[str, bool]:
        if not verify:
            print("[INFO] Reloading Captcha and Retrying")

        # Resetting Values
        self.dynamic = False
        # Clicking Reload Button
        if verify:
            reload_button = captcha_frame.locator("#recaptcha-verify-button")
        else:
            reload_button = captcha_frame.locator("#recaptcha-reload-button")
        await reload_button.click()
        return await self.handle_recaptcha()

    async def handle_recaptcha(self) -> Union[str, bool]:
        try:
            # Getting the Captcha Frame
            captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
            label_obj = captcha_frame.locator("//strong")
            prompt = await label_obj.text_content()

            if not prompt:
                raise ValueError("reCaptcha Task Text did not load.")

        except TimeoutError:
            # Checking if Captcha Token is available
            if captcha_token := await self.check_result():
                return captcha_token

            print("[ERROR] reCaptcha Frame did not load.")
            return False

        # Getting Recaptcha Tiles
        recaptcha_tiles = await captcha_frame.locator("[class='rc-imageselect-tile']").all()
        # Checking if Captcha Loaded Properly
        for _ in range(10):
            if len(recaptcha_tiles) in (9, 16):
                break

            await self.page.wait_for_timeout(1000)
            recaptcha_tiles = await captcha_frame.locator("[class='rc-imageselect-tile']").all()

        # Detecting Images and Clicking right Coordinates
        area_captcha = len(recaptcha_tiles) == 16
        not_yet_passed = await self.detect_tiles(prompt, area_captcha, area_captcha, captcha_frame)

        if self.dynamic and not area_captcha:
            while not_yet_passed:
                await self.page.wait_for_timeout(5000)
                not_yet_passed = await self.detect_tiles(prompt, area_captcha, True, captcha_frame)

        # Resetting value if challenge fails
        # Submit challenge
        submit_button = captcha_frame.locator("#recaptcha-verify-button")
        await submit_button.click()

        await self.page.wait_for_timeout(1000)
        if captcha_token := await self.check_result():
            return captcha_token
        else:
            # Retrying
            return await self.handle_recaptcha()

    async def solve_recaptcha(self) -> Union[str, bool]:
        # Resetting Values
        if not await self.click_checkbox():
            return False

        await self.page.wait_for_timeout(2000)
        return await self.handle_recaptcha()
