from pathlib import Path

from recognizer import Detector

image_dir = Path(__file__).parent.joinpath("images")
splitted_image_dir = image_dir.joinpath("splitted")
splitted_images = list(splitted_image_dir.iterdir())


def test_single_pathlib_input(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png")
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)
    print(f"Path: [full_page.png], Task: [Bicycle], Result: {response}; Coordinates: {coordinates}")

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_one_pathlib_input(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png")
    response, coordinates = detector.detect("bicycle", [img_bytes], area_captcha=False)
    print(f"Path: [full_page.png], Task: [Bicycle], Result: {response}; Coordinates: {coordinates}")

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_pathlibs_input(detector: Detector):
    response, coordinates = detector.detect("bicycle", splitted_images, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_single_path_input(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png")
    response, coordinates = detector.detect("bicycle", str(img_bytes), area_captcha=False)
    print(f"Path: [full_page.png], Task: [Bicycle], Result: {response}; Coordinates: {coordinates}")

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_one_path_input(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png")
    response, coordinates = detector.detect("bicycle", [str(img_bytes)], area_captcha=False)
    print(f"Path: [full_page.png], Task: [Bicycle], Result: {response}; Coordinates: {coordinates}")

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_paths_input(detector: Detector):
    splitted_paths = [str(splitted_path) for splitted_path in splitted_images]
    response, coordinates = detector.detect("bicycle", splitted_paths, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_single_bytes_input(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_one_bytes_input(detector: Detector):
    img_bytes = image_dir.joinpath("full_page.png").read_bytes()
    response, coordinates = detector.detect("bicycle", [img_bytes], area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)


def test_bytes_input(detector: Detector):
    img_bytes = [img.read_bytes() for img in splitted_images]
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)

    # General Checks
    assert response, coordinates
    assert len(response) == 9

    # Response Correctness
    assert sum(response) == len(coordinates)
