# -*- coding: utf-8 -*-
# Author     : Vinyzu
# GitHub     : https://github.com/Vinyzu
# Description:
from pathlib import Path

from recaptcha_challenger import Detector

image_dir = Path(__file__).parent.joinpath("images")

if __name__ == '__main__':
    detector = Detector()

    # Classification
    print("-- CLASSIFICATION --")
    result = detector.detect("bicycle", image_dir.joinpath("full_page.png").read_bytes(), area_captcha=False)
    print(f"Path: [full_page.png], Task: [Bicycle], Result: {result}")

    result = detector.detect("bicycle", image_dir.joinpath("classify_image.png").read_bytes(), area_captcha=False)
    print(f"Path: [classify_image.png], Task: [Bicycle], Result: {result}")

    # Area Detection
    print("-- AREA DETECTION --")
    result = detector.detect("motorcycle", image_dir.joinpath("only_captcha.png").read_bytes(), area_captcha=True)
    print(f"Path: [only_captcha.png], Task: [Motorcycle], Result: {result}")

    result = detector.detect("fire hydrant", image_dir.joinpath("area_image.png").read_bytes(), area_captcha=True)
    print(f"Path: [area_image.png], Task: [Fire Hydrant], Result: {result}")

    result = detector.detect("chimney", image_dir.joinpath("area_no_yolo.png").read_bytes(), area_captcha=True)
    print(f"Path: [area_no_yolo.png], Task: [Chimney], Result: {result}")
