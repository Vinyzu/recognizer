# -*- coding: utf-8 -*-
# Author     : Vinyzu
# GitHub     : https://github.com/Vinyzu
# Description:
from __future__ import annotations
from contextlib import suppress
from typing import List
import random

from recaptcha_challenger import Detector
from loguru import logger
from playwright.sync_api import Page as SyncPage, FrameLocator as SyncFrameLocator, Locator as SyncLocator, Response as SyncResponse
from playwright.sync_api import TimeoutError as PlaywrightSyncTimeout

from playwright.async_api import Page as AsyncPage, FrameLocator as AsyncFrameLocator, Locator as AsyncLocator, Request as AsyncRequest
from playwright.async_api import TimeoutError as PlaywrightAsyncTimeout

class AsyncChallenger:
    def __init__(self, page: AsyncPage) -> None:
        self.page = page
        self.detector = Detector()

        self.dynamic = None
        self.images = []
        self.coordinates = {}

        self.page.on('request', self.request_handler)

    async def request_handler(self, request: AsyncRequest) -> None:
        if request.url.startswith("https://www.google.com/recaptcha/"):
            if "reload" in request.url:
                with suppress(Exception):
                    response = await request.response()
                    self.dynamic = "dynamic" in await response.text()
            elif "payload" in request.url and request.resource_type == "image":
                with suppress(Exception):
                    response = await request.response()
                    self.images.append(await response.body())

    async def click_checkbox(self) -> bool:
        # Clicking Captcha Checkbox
        try:
            checkbox = self.page.frame_locator("//iframe[@title='reCAPTCHA']")
            await checkbox.locator(".recaptcha-checkbox-border").click()
            return True
        except PlaywrightAsyncTimeout:
            logger.error("Could not click reCaptcha Checkbox.")
            return False

    async def check_result(self) -> str | None:
        with suppress(ReferenceError):
            captcha_token = await self.page.evaluate("grecaptcha.getResponse()")
            return captcha_token

        with suppress(ReferenceError):
            captcha_token = await self.page.evaluate("grecaptcha.enterprise.getResponse()")
            return captcha_token

        return None

    async def detect_tiles(self, prompt: str, area_captcha: bool, verify: bool, captcha_frame: AsyncFrameLocator):
        if not any(self.images):
            logger.error("Images List is empty.")
            return await self.retry(captcha_frame, verify=verify)

        detector_results = self.detector.detect(prompt, self.images, area_captcha=area_captcha)

        if not any(detector_results):
            if not verify:
                logger.error("Detector did not return any Results.")
            return await self.retry(captcha_frame, verify=verify)
        elif len(detector_results) != len(self.coordinates):
            logger.error(f"Detector Results Length does not match Tiles Amount. Results Length: {len(detector_results)}. Tiles Amount: {len(self.coordinates)}")
            print("Error Coords:", self.coordinates)
            return await self.retry(captcha_frame, verify=verify)
        # elif len(detector_results) != len(recaptcha_tiles):
        #     logger.error(f"Detector Results Length does not match Tiles Amount. Results Length: {len(detector_results)}. Tiles Amount: {len(recaptcha_tiles)}")
        #     return await self.retry(captcha_frame)

        # Resetting Images before Clicking to clean log the next ones
        self.images = []
        # Getting right coordinates and clickingf
        self.coordinates = {coordinate: result for coordinate, result in zip(self.coordinates, detector_results) if result}
        for (coord_x, coord_y) in self.coordinates:
            await self.page.mouse.click(x=coord_x, y=coord_y)

    async def retry(self, captcha_frame, verify=False):
        if not verify:
            logger.info("Reloading Captcha and Retrying")

        # Resetting Values
        self.dynamic = None
        self.coordinates = {}
        self.images = []
        # Clicking Reload Button
        if verify:
            reload_button = captcha_frame.locator("#recaptcha-verify-button")
        else:
            reload_button = captcha_frame.locator("#recaptcha-reload-button")
        await reload_button.click()
        return await self.handle_recaptcha()

    async def handle_recaptcha(self):
        try:
            # Getting the Captcha Frame
            captcha_frame = self.page.frame_locator("//iframe[contains(@src,'bframe')]")
            label_obj = captcha_frame.locator("//strong")
            prompt = await label_obj.text_content()
        except PlaywrightAsyncTimeout:
            # Checking if Captcha Token is available
            if captcha_token := await self.check_result():
                return captcha_token

            logger.error("reCaptcha Frame did not load.")
            return False

        # Getting Recaptcha Tiles
        recaptcha_tiles = await captcha_frame.locator("[class='rc-imageselect-tile']").all()
        for tile in recaptcha_tiles:
            # Getting Bounding Box
            try:
                bounding_box = await tile.bounding_box()
            except (PlaywrightAsyncTimeout, ValueError):
                # Something really went wrong, hard resetting
                await self.page.mouse.click(0, 0)
                return await self.solve_recaptcha()

            x, y, width, height = bounding_box.values()
            # Calculating Center of Tile and random Offset
            center_x, center_y = x+(width//2), y+(height//2)
            offset_x, offset_y = random.randint(-width//4, width//4), random.randint(-height//4, height//4)
            # Getting Final Coordinate Clicking Point
            self.coordinates[(center_x + offset_x, center_y + offset_y)] = None

        # Detecting Images and Clicking right Coordinates
        area_captcha = len(recaptcha_tiles) == 16
        await self.detect_tiles(prompt, area_captcha, area_captcha, captcha_frame)
        # detector_results = self.detector.detect(prompt, self.images, area_captcha=area_captcha)
        #
        # if not any(recaptcha_tiles):
        #     logger.error("Detector did not return any Results.")
        #     return await self.retry(captcha_frame)
        # elif len(detector_results) != len(recaptcha_tiles):
        #     logger.error(f"Detector Results Length does not match Tiles Amount. Results Length: {len(detector_results)}. Tiles Amount: {len(recaptcha_tiles)}")
        #     return await self.retry(captcha_frame)
        #
        # # Resetting Images before Clicking to clean log the next ones
        # self.images = []
        # # Getting right coordinates and clicking
        # self.coordinates = {coordinate: result for coordinate, result in zip(self.coordinates, detector_results) if result}
        # for (coord_x, coord_y) in self.coordinates:
        #     await self.page.mouse.click(x=coord_x, y=coord_y)

        if self.dynamic and not area_captcha:
            while any(self.coordinates):
                # Todo: Check timeout
                await self.page.wait_for_timeout(4000)
                await self.detect_tiles(prompt, area_captcha, True, captcha_frame)

        # Resetting value if challenge fails
        self.images = []
        self.coordinates = {}
        # Submit challenge
        submit_button = captcha_frame.locator("#recaptcha-verify-button")
        await submit_button.click()

        await self.page.wait_for_timeout(1000)
        if captcha_token := await self.check_result():
            return captcha_token
        else:
            # Retrying
            return await self.handle_recaptcha()

    async def solve_recaptcha(self):
        # Resetting Values
        self.coordinates = {}
        if not await self.click_checkbox():
            return False

        return await self.handle_recaptcha()
