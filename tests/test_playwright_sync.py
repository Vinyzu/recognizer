from __future__ import annotations

from contextlib import suppress

from playwright.sync_api import Page

from recognizer.agents.playwright import SyncChallenger


def test_sync_challenger(sync_page: Page):
    challenger = SyncChallenger(sync_page, click_timeout=1000)
    # For slow Pytest Loading
    challenger.detector.detection_models.check_loaded()

    sync_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")

    with suppress(RecursionError):
        res = challenger.solve_recaptcha()
        assert res
