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
    assert sum(response) == len(coordinates)
    assert response == [True, True, False, False, True, False, False, False, False]


def test_only_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("only_captcha.png").read_bytes()
    response, coordinates = detector.detect("motorcycle", img_bytes, area_captcha=True)

    # General Checks
    assert response, coordinates
    assert len(response) == 16

    # Response Correctness
    assert sum(response) == len(coordinates)
    assert response == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_area_yolo_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("area_image.png").read_bytes()
    response, coordinates = detector.detect("fire hydrant", img_bytes, area_captcha=True)

    # General Checks
    assert response, coordinates
    assert len(response) == 16

    # Response Correctness
    assert sum(response) == len(coordinates)
    assert response == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_area_clip_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("area_no_yolo.png").read_bytes()
    response, coordinates = detector.detect("chimney", img_bytes, area_captcha=True)

    # General Checks
    assert response, coordinates
    assert len(response) == 16

    # Response Correctness
    assert sum(response) == len(coordinates)
    assert response == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_classify_yolo_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("classify_image.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)
    assert response == [False, False, False, True, False, False, True, True, False]


def test_classify_clip_captcha(detector: Detector):
    img_bytes = image_dir.joinpath("classify_no_yolo.png").read_bytes()
    response, coordinates = detector.detect("stairs", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)
    assert response == [True, True, False, True, False, False, False, False, True]
