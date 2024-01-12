from __future__ import annotations

import math
from typing import List, Tuple

import cv2
from numpy.typing import NDArray
from numpy import uint8, concatenate


def get_captcha_fields(img: NDArray[uint8]) -> Tuple[List[bytes], List[Tuple[int, int]]]:
    captcha_fields: List[Tuple[bytes, int, int]] = []

    # Turn image to grayscale and Apply a white threshold to it
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 254, 255, cv2.CHAIN_APPROX_NONE)
    # Find Countours of the white threshold
    try:
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # type: ignore
    except ValueError:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # type: ignore

    for contour in contours:
        # Checking Image area to only get large captcha areas (no image details)
        area = cv2.contourArea(contour)
        if area > 1000:
            # IDK what this does i copy pasted it :)
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            # Calculating x, y, width, height, aspectRatio
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h

            if 0.95 <= aspectRatio <= 1.05:
                # Cropping Image area to Captcha Field
                crop_img = img[y:y+h, x:x+w]
                # Cv2 to Image Bytes
                image_bytes = cv2.imencode('.jpg', crop_img)[1].tobytes()
                # Getting Center of Captcha Field
                center_x, center_y = x+(w//2), y+(h//2)
                captcha_fields.append((image_bytes, center_x, center_y))

    sorted_captcha_fields: List[Tuple[bytes, int, int]] = sorted(captcha_fields, key=lambda element: [element[2], element[1]])
    # return sorted_captcha_fields
    return (
        [field[0] for field in sorted_captcha_fields],
        [(field[1], field[2]) for field in sorted_captcha_fields]
    )


def split_image_into_tiles(img: NDArray[uint8], tile_count: int) -> List[bytes]:
    tiles = []

    # Get the dimensions of the image
    height, width, _ = img.shape

    # Calculate the size of each tile
    tile_width = width // tile_count
    tile_height = height // tile_count

    # Iterate through the image and crop it into tiles
    for i in range(tile_count):
        for j in range(tile_count):
            x_start = j * tile_width
            x_end = (j + 1) * tile_width
            y_start = i * tile_height
            y_end = (i + 1) * tile_height

            # Crop the image to create a tile
            tile = img[y_start:y_end, x_start:x_end]
            image_bytes = cv2.imencode('.jpg', tile)[1].tobytes()
            tiles.append(image_bytes)

    return tiles


def create_image_grid(images: List[NDArray[uint8]]) -> NDArray[uint8]:
    cv2_images = images
    tile_count_per_row = int(math.sqrt(len(cv2_images)))

    # Combining horizontal layers together
    layers = []
    for i in range(tile_count_per_row):
        layer_images = [cv2_images[i*tile_count_per_row + j] for j in range(tile_count_per_row)]
        layer = concatenate(layer_images, axis=1)
        layers.append(layer)

    # Combining layers verticly to one image
    combined_img = concatenate(layers, axis=0)

    return combined_img
