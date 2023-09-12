# -*- coding: utf-8 -*-
# Author     : Vinyzu
# GitHub     : https://github.com/Vinyzu
# Description:
import typing


class ArmorException(Exception):
    """Armor module basic exception"""

    def __init__(
        self,
        msg: typing.Optional[str] = None,
        stacktrace: typing.Optional[typing.Sequence[str]] = None,
    ):
        self.msg = msg
        self.stacktrace = stacktrace
        super().__init__()

    def __str__(self) -> str:
        exception_msg = f"Message: {self.msg}\n"
        if self.stacktrace:
            stacktrace = "\n".join(self.stacktrace)
            exception_msg += f"Stacktrace:\n{stacktrace}"
        return exception_msg


class ChallengeException(ArmorException):
    """hCAPTCHA Challenge basic exceptions"""


class ChallengePassed(ChallengeException):
    """Challenge not popping up"""


class LoadImageTimeout(ChallengeException):
    """Loading challenge image timed out"""


class LabelNotFoundException(ChallengeException):
    """Get an empty image label name"""