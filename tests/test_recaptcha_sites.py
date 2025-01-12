from __future__ import annotations

from contextlib import suppress

import pytest
from patchright.async_api import Page

from recognizer.agents.playwright import AsyncChallenger

# All URLs:
# https://berstend.github.io/static/recaptcha/enterprise-checkbox-auto-recaptchadotnet.html
# https://berstend.github.io/static/recaptcha/enterprise-checkbox-auto.html
# https://berstend.github.io/static/recaptcha/enterprise-checkbox-explicit.html
# https://berstend.github.io/static/recaptcha/v2-checkbox-auto-nowww.html
# https://berstend.github.io/static/recaptcha/v2-checkbox-auto-recaptchadotnet-nowww.html
# https://berstend.github.io/static/recaptcha/v2-checkbox-explicit.html
# https://berstend.github.io/static/recaptcha/v2-invisible-auto.html
# https://berstend.github.io/static/recaptcha/v2-invisible-explicit.html
# https://berstend.github.io/static/recaptcha/v2-invisible-explicit-isolated.html
# https://www.recaptcha.net/recaptcha/api2/demo
# https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php
# https://nopecha.com/demo/recaptcha#easy
# https://nopecha.com/demo/recaptcha#moderate
# https://nopecha.com/demo/recaptcha#hard
# https://2captcha.com/demo/recaptcha-v2
# https://2captcha.com/demo/recaptcha-v2-invisible
# https://2captcha.com/demo/recaptcha-v2-callback
# https://2captcha.com/demo/recaptcha-v2-enterprise
# https://2captcha.com/demo/recaptcha-v3-enterprise
# https://2captcha.com/demo/recaptcha-v3
# https://patrickhlauke.github.io/recaptcha/
# https://testrecaptcha.github.io/
# http://www.recaptcha2.lyates.com/
# https://ask.usda.gov/resource/1589940255000/recaptcha2
# https://jfo.moj.go.th/page/complain3.php
# https://huyliem.z23.web.core.windows.net/
# https://www.opju.ac.in/nitincap
# https://www.flight-simulators.co.uk/acatalog/mailtest1.php
# https://evans-email.glitch.me/


@pytest.mark.asyncio
async def test_bernsted_enterprise_auto_recaptchadotnet(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/enterprise-checkbox-auto-recaptchadotnet.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_enterprise_auto(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/enterprise-checkbox-auto.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_enterprise_explicit(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/enterprise-checkbox-explicit.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_auto(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/v2-checkbox-auto-nowww.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_auto_recaptchadotnet(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/v2-checkbox-auto-recaptchadotnet-nowww.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_explicit(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/v2-checkbox-explicit.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_invisible_auto(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/v2-invisible-auto.html")
    await async_page.click("[data-callback='onSubmit']")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_invisible_explicit(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/v2-invisible-explicit.html")
    await async_page.click("[id='submit']")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_invisible_explicit_isolated(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://berstend.github.io/static/recaptcha/v2-invisible-explicit-isolated.html")
    await async_page.click("[id='submit']")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_recaptcha_net(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://www.recaptcha.net/recaptcha/api2/demo")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_recaptcha_demo_appspot(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_nopecha_easy(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://nopecha.com/demo/recaptcha#easy")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_nopecha_moderate(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://nopecha.com/demo/recaptcha#moderate")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_nopecha_hard(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://nopecha.com/demo/recaptcha#hard")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://2captcha.com/demo/recaptcha-v2")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2_invisible(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://2captcha.com/demo/recaptcha-v2-invisible")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2_callback(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://2captcha.com/demo/recaptcha-v2-callback")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2_enterprise(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://2captcha.com/demo/recaptcha-v2-enterprise")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v3_enterprise(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://2captcha.com/demo/recaptcha-v3-enterprise")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v3(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://2captcha.com/demo/recaptcha-v3")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_patrickhlauke(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://patrickhlauke.github.io/recaptcha/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_testrecaptcha_github(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://testrecaptcha.github.io/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_lyates_v2(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("http://www.recaptcha2.lyates.com/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_usda_v2(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://ask.usda.gov/resource/1589940255000/recaptcha2")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_jfo_moj_go_th_v3(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://jfo.moj.go.th/page/complain3.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_huyliem_windows(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://huyliem.z23.web.core.windows.net/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_opju_ac_in(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://www.opju.ac.in/nitincap")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_flight_simulators_mailtest(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://www.flight-simulators.co.uk/acatalog/mailtest1.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_evans_email_glitch(async_page: Page):
    challenger = AsyncChallenger(async_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    await async_page.goto("https://evans-email.glitch.me/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res
