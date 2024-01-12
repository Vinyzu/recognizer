from pathlib import Path

from recognizer import Detector

image_dir = Path(__file__).parent.joinpath("images")


def test_full_page_screenshot(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == 3
    assert len(coordinates) == 3


def test_only_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("only_captcha.png").read_bytes()
    response, coordinates = detector.detect("motorcycle", img_bytes, area_captcha=True)

    # General Checks
    assert response, coordinates
    assert len(response) == 16

    # Response Correctness
    assert sum(response) == 6
    assert len(coordinates) == 6


def test_area_yolo_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("area_image.png").read_bytes()
    response, coordinates = detector.detect("fire hydrant", img_bytes, area_captcha=True)

    # General Checks
    assert response, coordinates
    assert len(response) == 16

    # Response Correctness
    assert sum(response) == 8
    assert len(coordinates) == 8


def test_area_clip_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("area_no_yolo.png").read_bytes()
    response, coordinates = detector.detect("chimney", img_bytes, area_captcha=True)

    # General Checks
    assert response, coordinates
    assert len(response) == 16

    # Response Correctness
    assert sum(response) == 4
    assert len(coordinates) == 4


def test_classify_yolo_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("classify_image.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == 3
    assert len(coordinates) == 3


def test_classify_clip_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("classify_no_yolo.png").read_bytes()
    response, coordinates = detector.detect("stairs", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == 4
    assert len(coordinates) == 4
