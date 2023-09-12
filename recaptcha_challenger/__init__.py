# -*- coding: utf-8 -*-
# Author     : Vinyzu
# GitHub     : https://github.com/Vinyzu
# Description:
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hcaptcha_challenger as solver
from hcaptcha_challenger.onnx.modelhub import ModelHub
from .utils import init_log
from .components.detector import Detector

__all__ = ["Detector"]
# solver.install(upgrade=True, flush_yolo=True)


@dataclass
class Project:
    at_dir = Path(__file__).parent
    logs = at_dir.joinpath("logs")


project = Project()

init_log(
    runtime=project.logs.joinpath("runtime.log"),
    error=project.logs.joinpath("error.log"),
    serialize=project.logs.joinpath("serialize.log"),
)


def install(
    upgrade: bool | None = False,
    username: str = "QIN2DIM",
    lang: str = "en",
    flush_yolo: bool = False,
):
    modelhub = ModelHub.from_github_repo(username=username, lang=lang)
    modelhub.pull_objects(upgrade=upgrade)
    modelhub.assets.flush_runtime_assets(upgrade=upgrade)
    if flush_yolo:
        from hcaptcha_challenger.onnx.modelhub import DEFAULT_KEYPOINT_MODEL

        modelhub.pull_model(focus_name=DEFAULT_KEYPOINT_MODEL)

