from pathlib import Path

import cv2
from imageio.v2 import imread
import matplotlib.pyplot as plt

from recognizer import Detector

image_dir = Path(__file__).parent.joinpath("images")
show_results = True


def draw_coordinates(img_bytes, coordinates):
    if show_results:
        image = imread(img_bytes)
        for (x, y) in coordinates:
            image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    detector = Detector()

    # Classification
    print("-- CLASSIFICATION --")
    img_bytes = image_dir.joinpath("full_page.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)
    print(f"Path: [full_page.png], Task: [Bicycle], Result: {response}; Coordinates: {coordinates}")
    draw_coordinates(img_bytes, coordinates)

    img_bytes = image_dir.joinpath("classify_image.png").read_bytes()
    response, coordinates = detector.detect("bicycle", img_bytes, area_captcha=False)
    print(f"Path: [classify_image.png], Task: [Bicycle], Result: {response}; Coordinates: {coordinates}")
    draw_coordinates(img_bytes, coordinates)

    img_bytes = image_dir.joinpath("classify_no_yolo.png").read_bytes()
    response, coordinates = detector.detect("stairs", img_bytes, area_captcha=False)
    print(f"Path: [classify_no_yolo.png], Task: [Stairs], Result: {response}; Coordinates: {coordinates}")
    draw_coordinates(img_bytes, coordinates)

    # Area Detection
    # print("-- AREA DETECTION --")
    # img_bytes = image_dir.joinpath("only_captcha.png").read_bytes()
    # response, coordinates = detector.detect("motorcycle", img_bytes, area_captcha=True)
    # print(f"Path: [only_captcha.png], Task: [Motorcycle], Result: {response}; Coordinates: {coordinates}")
    # draw_coordinates(img_bytes, coordinates)
    #
    # img_bytes = image_dir.joinpath("area_image.png").read_bytes()
    # response, coordinates = detector.detect("fire hydrant", img_bytes, area_captcha=True)
    # print(f"Path: [area_image.png], Task: [Fire Hydrant], Result: {response}; Coordinates: {coordinates}")
    # draw_coordinates(img_bytes, coordinates)
    #
    # img_bytes = image_dir.joinpath("area_no_yolo.png").read_bytes()
    # response, coordinates = detector.detect("chimney", img_bytes, area_captcha=True)
    # print(f"Path: [area_no_yolo.png], Task: [Chimney], Result: {response}; Coordinates: {coordinates}")
    # draw_coordinates(img_bytes, coordinates)
    #
    # img_bytes = image_dir.joinpath("area_no_yolo1.png").read_bytes()
    # response, coordinates = detector.detect("crosswalks", img_bytes, area_captcha=True)
    # print(f"Path: [area_no_yolo.png], Task: [Crosswalks], Result: {response}; Coordinates: {coordinates}")
    # draw_coordinates(img_bytes, coordinates)
