from __future__ import annotations

from contextlib import suppress
from typing import Optional, Union

from recognizer import Detector

from playwright.sync_api import Page, FrameLocator, Request, TimeoutError


class SyncChallenger:
    def __init__(self, page: Page, click_timeout: Optional[int] = None, retry_times: int = 15) -> None:
        """
        Initialize a reCognizer AsyncChallenger instance with specified configurations.

        Args:
            page (Page): The Playwright Page to initialize on.
            click_timeout (int, optional): Click Timeouts between captcha-clicks.
            retry_times (int, optional): Maximum amount of retries before raising an Exception. Defaults to 15.
        """
        self.page = page
        self.detector = Detector()

        self.click_timeout = click_timeout
        self.retry_times = retry_times
        self.retried = 0
        self.dynamic = False

        self.page.on('request', self.request_handler)

    def request_handler(self, request: Request) -> None:
        if request.url.startswith("https://www.google.com/recaptcha/"):
            if "reload" in request.url or "userverify" in request.url:
                with suppress(Exception):
                    response = request.response()
                    assert response

                    self.dynamic = "dynamic" in response.text()

    def click_checkbox(self) -> bool:
        # Clicking Captcha Checkbox
        try:
            checkbox = self.page.frame_locator("iframe[title='reCAPTCHA']")
            checkbox.locator(".recaptcha-checkbox-border").click()
            return True
        except TimeoutError:
            print("[ERROR] Could not click reCaptcha Checkbox.")
            return False

    def check_result(self) -> Union[str, None]:
        with suppress(ReferenceError):
            captcha_token: str = self.page.evaluate("grecaptcha.getResponse()")
            return captcha_token

        with suppress(ReferenceError):
            enterprise_captcha_token: str = self.page.evaluate("grecaptcha.enterprise.getResponse()")
            return enterprise_captcha_token

        return None

    def detect_tiles(self, prompt: str, area_captcha: bool, verify: bool, captcha_frame: FrameLocator) -> Union[str, bool]:
        image = self.page.screenshot()
        response, coordinates = self.detector.detect(prompt, image, area_captcha=area_captcha)

        if not any(response):
            if not verify:
                print("[ERROR] Detector did not return any Results.")
                return self.retry(captcha_frame, verify=verify)
            else:
                return False

        for (coord_x, coord_y) in coordinates:
            self.page.mouse.click(coord_x, coord_y)
            if self.click_timeout:
                self.page.wait_for_timeout(self.click_timeout)

        return True

    def retry(self, captcha_frame, verify=False) -> Union[str, bool]:
        self.retried += 1
        if self.retried >= self.retry_times:
            raise RecursionError(f"Exceeded maximum retry times of {self.retry_times}")

        # Resetting Values
        self.dynamic = False
        # Clicking Reload Button
        if verify:
            reload_button = captcha_frame.locator("#recaptcha-verify-button")
        else:
            print("[INFO] Reloading Captcha and Retrying")
            reload_button = captcha_frame.locator("#recaptcha-reload-button")
        reload_button.click()
        return self.handle_recaptcha()

    def handle_recaptcha(self) -> Union[str, bool]:
        try:
            # Getting the Captcha Frame
            captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
            label_obj = captcha_frame.locator("//strong")
            prompt = label_obj.text_content()

            if not prompt:
                raise ValueError("reCaptcha Task Text did not load.")

        except TimeoutError:
            # Checking if Captcha Token is available
            if captcha_token := self.check_result():
                return captcha_token

            print("[ERROR] reCaptcha Frame did not load.")
            return False

        # Getting Recaptcha Tiles
        recaptcha_tiles = captcha_frame.locator("[class='rc-imageselect-tile']").all()
        # Checking if Captcha Loaded Properly
        for _ in range(10):
            if len(recaptcha_tiles) in (9, 16):
                break

            self.page.wait_for_timeout(1000)
            recaptcha_tiles = captcha_frame.locator("[class='rc-imageselect-tile']").all()

        # Detecting Images and Clicking right Coordinates
        area_captcha = len(recaptcha_tiles) == 16
        not_yet_passed = self.detect_tiles(prompt, area_captcha, area_captcha, captcha_frame)

        if self.dynamic and not area_captcha:
            while not_yet_passed:
                self.page.wait_for_timeout(5000)
                not_yet_passed = self.detect_tiles(prompt, area_captcha, True, captcha_frame)

        # Resetting value if challenge fails
        # Submit challenge
        submit_button = captcha_frame.locator("#recaptcha-verify-button")
        submit_button.click()

        self.page.wait_for_timeout(1000)
        if captcha_token := self.check_result():
            return captcha_token
        else:
            # Retrying
            self.retried += 1
            if self.retried >= self.retry_times:
                raise RecursionError(f"Exceeded maximum retry times of {self.retry_times}")

            return self.handle_recaptcha()

    def solve_recaptcha(self) -> Union[str, bool]:
        """
        Solve a hcaptcha-challenge on the specified Playwright Page

        Returns:
            str/bool: The result of the challenge
        Raises:
            RecursionError: If the challenger doesn´t succeed in the given retry times
        """
        # Resetting Values
        if not self.click_checkbox():
            return False

        self.page.wait_for_timeout(2000)
        return self.handle_recaptcha()
