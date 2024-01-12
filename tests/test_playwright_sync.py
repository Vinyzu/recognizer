from __future__ import annotations

import pytest

from playwright.sync_api import Page
from recognizer.agents.playwright import SyncChallenger


@pytest.mark.xfail
def test_sync_challenger(sync_page: Page):
    challenger = SyncChallenger(sync_page)

    sync_page.goto("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox-explicit.php")
    res = challenger.solve_recaptcha()
    assert res
