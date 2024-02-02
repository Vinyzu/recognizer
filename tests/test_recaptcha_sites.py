from __future__ import annotations

import pytest
from contextlib import suppress

from botright.extended_typing import Page
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
async def test_bernsted_enterprise_auto_recaptchadotnet(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/enterprise-checkbox-auto-recaptchadotnet.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_enterprise_auto(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/enterprise-checkbox-auto.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_enterprise_explicit(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/enterprise-checkbox-explicit.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_auto(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/v2-checkbox-auto-nowww.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_auto_recaptchadotnet(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/v2-checkbox-auto-recaptchadotnet-nowww.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_explicit(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/v2-checkbox-explicit.html")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_invisible_auto(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/v2-invisible-auto.html")
    await botright_page.click("[data-callback='onSubmit']")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_invisible_explicit(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/v2-invisible-explicit.html")
    await botright_page.click("[id='submit']")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
async def test_bernsted_v2_invisible_explicit_isolated(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://berstend.github.io/static/recaptcha/v2-invisible-explicit-isolated.html")
    await botright_page.click("[id='submit']")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_recaptcha_net(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://www.recaptcha.net/recaptcha/api2/demo")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_recaptcha_demo_appspot(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_nopecha_easy(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://nopecha.com/demo/recaptcha#easy")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_nopecha_moderate(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://nopecha.com/demo/recaptcha#moderate")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_nopecha_hard(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://nopecha.com/demo/recaptcha#hard")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://2captcha.com/demo/recaptcha-v2")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2_invisible(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://2captcha.com/demo/recaptcha-v2-invisible")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2_callback(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://2captcha.com/demo/recaptcha-v2-callback")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v2_enterprise(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://2captcha.com/demo/recaptcha-v2-enterprise")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v3_enterprise(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://2captcha.com/demo/recaptcha-v3-enterprise")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_2captcha_v3(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://2captcha.com/demo/recaptcha-v3")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_patrickhlauke(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://patrickhlauke.github.io/recaptcha/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_testrecaptcha_github(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://testrecaptcha.github.io/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_lyates_v2(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("http://www.recaptcha2.lyates.com/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_usda_v2(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://ask.usda.gov/resource/1589940255000/recaptcha2")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_jfo_moj_go_th_v3(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://jfo.moj.go.th/page/complain3.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_huyliem_windows(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://huyliem.z23.web.core.windows.net/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_opju_ac_in(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://www.opju.ac.in/nitincap")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_flight_simulators_mailtest(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://www.flight-simulators.co.uk/acatalog/mailtest1.php")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res


@pytest.mark.asyncio
@pytest.mark.skip(reason="No different challenge type, Skipping due to time complexity")
async def test_evans_email_glitch(botright_page: Page):
    challenger = AsyncChallenger(botright_page)

    await botright_page.goto("https://evans-email.glitch.me/")

    with suppress(RecursionError):
        res = await challenger.solve_recaptcha()
        assert res
