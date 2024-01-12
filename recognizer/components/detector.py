from __future__ import annotations

import base64
import binascii
import math
import random
from pathlib import Path
from typing import List, Tuple, Union, Optional
from contextlib import suppress

import cv2
from imageio.v2 import imread

from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

from numpy.typing import NDArray
from numpy import generic, concatenate

from .prompt_handler import split_prompt_message
from .image_processor import get_captcha_fields, split_image_into_tiles, create_image_grid


class YoloDetector:
    yolo_alias = {
        "bicycle": ["bicycle"],
        "car": ["car", "truck"],
        "bus": ["bus", "truck"],
        "motorcycle": ["motorcycle"],
        "boat": ["boat"],
        "fire hydrant": ["fire hydrant", "parking meter"],
        "parking meter": ["fire hydrant", "parking meter"],
        "traffic light": ["traffic light"],
    }

    def __init__(self) -> None:
        self.model = YOLO("yolov8m-seg.pt")
        self.yolo_classes = list(self.model.names.values())

    def get_tiles_in_bounding_box(self, img: NDArray[generic], tile_amount: int, point_start: Tuple[int, int], point_end: Tuple[int, int]) -> List[bool]:
        tiles_in_bbox = []
        # Define the size of the original image
        height, width, _ = img.shape
        tiles_per_row = int(math.sqrt(tile_amount))

        # Calculate the width and height of each tile
        tile_width = width // tiles_per_row
        tile_height = height // tiles_per_row

        for i in range(tiles_per_row):
            for j in range(tiles_per_row):
                # Calculate the coordinates of the current tile
                tile_x1 = j * tile_height
                tile_y1 = i * tile_width
                tile_x2 = (j + 1) * tile_height
                tile_y2 = (i + 1) * tile_width

                # Calculate Tile Area
                tile_area = (tile_x2 - tile_x1) * (tile_y2 - tile_y1)

                # Calculate the intersection area
                intersection_x1 = max(tile_x1, point_start[0])
                intersection_x2 = min(tile_x2, point_end[0])
                intersection_y1 = max(tile_y1, point_start[1])
                intersection_y2 = min(tile_y2, point_end[1])

                # Check if the current tile intersects with the bounding box
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    # Getting intersection area coordinates and calculating Tile Coverage
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    if (intersection_area / tile_area) == 1:
                        tiles_in_bbox.append(True)
                    else:
                        tiles_in_bbox.append(False)
                else:
                    tiles_in_bbox.append(False)

        return tiles_in_bbox

    def detect_image(self, image: NDArray[generic], tile_amount: int, task_type: str) -> List[bool]:
        response = [False for _ in range(tile_amount)]
        height, width, _ = image.shape
        tiles_per_row = int(math.sqrt(tile_amount))
        tile_width, tile_height = width // tiles_per_row, height // tiles_per_row

        outputs = self.model.predict(image, verbose=False, conf=0.2, iou=0.3)  # , save=True
        results = outputs[0]

        for result in results:
            assert result
            # Check if correct task type
            class_index = int(result.boxes.cls[0])
            if self.yolo_classes[class_index] not in self.yolo_alias[task_type]:
                continue

            masks = result.masks
            mask = masks.xy[0]

            for coord in mask:
                mask_point_x, mask_point_y = tuple(coord)

                # Calculate the column and row of the tile based on the coordinate
                col = int(mask_point_x // tile_width)
                row = int(mask_point_y // tile_height)

                # Ensure the column and row are within the valid range
                col = min(col, tiles_per_row - 1)
                row = min(row, tiles_per_row - 1)

                tile_index = row * tiles_per_row + col
                if response[tile_index]:
                    continue

                # Calculate the boundary for the 5% area within the tile
                boundary_x = tile_width * 0.05
                boundary_y = tile_height * 0.05

                # Check if the point is within the 5% boundary of the tile area
                within_boundary = (
                        boundary_x <= coord[1] % tile_width <= tile_width - boundary_x
                        and boundary_y <= coord[0] % tile_height <= tile_height - boundary_y
                )

                if within_boundary:
                    response[tile_index] = True

            # In AreaCaptcha Mode, Calculate Tiles inside of boundary, which arent covered by mask point
            if tile_amount == 16:
                coords = result.boxes.xyxy.flatten().tolist()
                points_start, point_end = coords[:2], coords[2:]
                tiles_in_bbox = self.get_tiles_in_bounding_box(image, tile_amount, tuple(points_start), tuple(point_end))
                # Appending True Tiles to Response but not making True ones False again
                response = [x or y for x, y in zip(response, tiles_in_bbox)]

        return response


class ClipDetector:
    plain_labels = ["bicycle", "boat", "bus", "car", "fire hydrant", "motorcycle", "parking meter", "traffic light",  # YOLO TASKS
                    "bridge", "chimney", "crosswalk", "mountain", "palm tree", "stair", "tractor", "taxi"]

    all_labels = ["a bicycle", "a boat", "a bus", "a car", "a fire hydrant", "a motorcycle", "a parking meter", "a traffic light",  # YOLO TASKS
                  "a concrete bridge or concrete pillars", "a chimney on top of a house", "white stripes or lines of a crosswalk on a gray ground of a street",  # of a crosswalk
                  "a green or gray mountain or grassy hill in the landscape", "a palm tree", "a stairway or stairs or steps",
                  "a tractor or agricultural vehicle", "a taxi or a yellow car"]

    thresholds = {
        "bridge": 0.7285372716747225,
        "chimney": 0.7918647485226393,
        # "crosswalk": 0.9508330273628235,
        "crosswalk": 0.8879293048381806,
        # "mountain": 0.4551278884819476,
        "mountain": 0.5551278884819476,
        "palm tree": 0.8093279512040317,
        "stair": 0.7312694561691023,
        "tractor": 0.9385110986077537,
        "taxi": 0.7967491503432393,
    }

    area_captcha_labels = {
        "bridge": ["a bridge", "a street", "a road", "empty sky or air or clouds"],
        "chimney": ["a chimney", "a house", "a rooftop", "a factory", "a empty sky", "air", "clouds"],
        "crosswalk": ["white stripes or white lines of a crosswalk", "empty grey ground of a street", "empty grey ground of a road", "cars", "traffic"],  # , "an empty grey ground of a floor"
        "mountain": ["a nature mountain or grassy hill in the background", "an empty sky", "an empty street", "an empty road", "cars or traffic"],
        "palm tree": ["a palm tree", "a empty sky", "empty air with clouds", "a street", "a road", "cars or traffic", "a beach"],
        "stair": ["a stairway or stairs or steps", "a street", "a road", "a building", "a door", "an empty area", "an empty floor", "an empty tiling"],
        "tractor": ["a tractor or agricultural vehicle", "a street", "a road", "an area", "an empty landscape", "empty nature"],
        "taxi": ["a yellow car or taxi", "a street", "a road", "an area"],
    }

    def __init__(self) -> None:
        self.vit_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vit_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def detect_vit(self, images: List[NDArray[generic]], task_type: str, area_captcha: Optional[bool] = False) -> List[bool]:
        response = []
        # labels = self.all_labels if not area_captcha else self.area_captcha_labels[task_type]
        labels = self.area_captcha_labels[task_type] if area_captcha else self.all_labels

        inputs = self.vit_processor(text=labels, images=images, return_tensors="pt", padding=True)
        outputs = self.vit_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        results = probs.tolist()

        for result in results:
            if area_captcha:
                max_index = result.index(max(result))
                choice = max_index == 0
            # if area_captcha and self.area_captcha_max_choice[task_type]:
            #     max_index = result.index(max(result))
            #     choice = self.plain_labels[max_index] == task_type
            else:
                task_index = self.plain_labels.index(task_type)
                prediction = result[task_index]
                choice = prediction >= (self.thresholds[task_type] - 0.3)

            response.append(choice)

        return response

    def detect_image(self, images: List[NDArray[generic]], task_type: str) -> List[bool]:
        if len(images) == 9:
            return self.detect_vit(images, task_type, area_captcha=False)

        indexes_4x4 = [0, 2, 8, 10]
        response_4x4 = [True, True, True, True]

        images_2x2 = []
        for i, res in enumerate(response_4x4):
            if not res:
                continue

            start_index = indexes_4x4[i]
            grid_2x2_tile0 = concatenate([images[start_index], images[start_index+1]], axis=1)
            grid_2x2_tile1 = concatenate([images[start_index+4], images[start_index+5]], axis=1)

            images_2x2.append(grid_2x2_tile0)
            images_2x2.append(grid_2x2_tile1)

        response_2x2 = self.detect_vit(images_2x2, task_type, area_captcha=True)

        response = [False for _ in range(len(images))]
        for i, res in enumerate(response_4x4):
            start_index = indexes_4x4[i]

            if not res:
                response[start_index], response[start_index+1] = False, False
                response[start_index+4], response[start_index+5] = False, False

            else:
                response[start_index], response[start_index + 1] = response_2x2[0], response_2x2[0]
                response_2x2.pop(0)

                response[start_index + 4], response[start_index + 5] = response_2x2[0], response_2x2[0]
                response_2x2.pop(0)

        return response


class Detector:
    challenge_alias = {
        "car": "car", "cars": "car", "vehicles": "car",
        "taxis": "taxi", "taxi": "taxi",
        "bus": "bus", "buses": "bus",
        "motorcycle": "motorcycle", "motorcycles": "motorcycle",
        "bicycle": "bicycle", "bicycles": "bicycle",
        "boats": "boat", "boat": "boat",
        "tractors": "tractor", "tractor": "tractor",
        "stairs": "stair", "stair": "stair",
        "palm trees": "palm tree", "palm tree": "palm tree",
        "fire hydrants": "fire hydrant", "a fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
        "parking meters": "parking meter", "parking meter": "parking meter",
        "crosswalks": "crosswalk", "crosswalk": "crosswalk",
        "traffic lights": "traffic light", "traffic light": "traffic light",
        "bridges": "bridge", "bridge": "bridge",
        "mountains or hills": "mountain", "mountain or hill": "mountain", "mountain": "mountain", "mountains": "mountain", "hills": "mountain", "hill": "mountain",
        "chimney": "chimney", "chimneys": "chimney"
    }

    def __init__(self) -> None:
        self.yolo_detector = YoloDetector()
        self.clip_detector = ClipDetector()

    def calculate_approximated_coords(self, grid_width: int, grid_height: int, tile_amount: int) -> List[Tuple[int, int]]:
        # Calculate the middle points of the images within the grid
        middle_points = []

        for y in range(tile_amount):
            for x in range(tile_amount):
                # Calculate the coordinates of the middle point of each image
                middle_x = (x * grid_width) + (grid_width // 2)
                middle_y = (y * grid_height) + (grid_height // 2)

                # Append the middle point coordinates to the list
                middle_points.append((middle_x, middle_y))

        return middle_points

    def handle_single_image(self, single_image: Union[Path, bytes], area_captcha: bool) -> Tuple[List[bytes], List[Tuple[int, int]]]:
        if isinstance(single_image, bytes):
            with suppress(binascii.Error):
                single_image = base64.b64decode(single_image, validate=True)

        # Image Bytes to Cv2
        rgba_img = imread(single_image)
        img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)

        # Image Splitting Presuming has white barriers
        images, coords = get_captcha_fields(img)

        if len(images) == 1:
            # Turning bytes from get_captcha_fields back to Cv2
            rgba_img = imread(images[0])
            img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)

            # Either it is just a single image or no white barriers
            height, width, _ = img.shape

            if height > 200 and width > 200:
                tiles_amount = 4 if area_captcha else 3
                images = split_image_into_tiles(img, tiles_amount)
                coords = self.calculate_approximated_coords(height//tiles_amount, width//tiles_amount, tiles_amount)

        return images, coords

    def handle_multiple_images(self, images: List[bytes]) -> List[NDArray[generic]]:
        cv2_images = []
        for image in images:
            try:
                byte_image = base64.b64decode(image, validate=True)
            except binascii.Error:
                byte_image = image

            # Image Bytes to Cv2
            rgba_img = imread(byte_image)
            cv2_img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)
            cv2_images.append(cv2_img)

        return cv2_images

    def detect(self, prompt: str, images: Union[Path, bytes, List[Path], List[bytes]], area_captcha: Optional[bool] = None) -> Tuple[List[bool], List[Tuple[int, int]]]:
        response = []
        coordinates: List[Tuple[int, int]] = []
        # Making best guess if its area_captcha if user did not specify
        area_captcha = "square" in prompt if area_captcha is None else area_captcha
        label = split_prompt_message(prompt)

        if label not in self.challenge_alias:
            print(f"[ERROR] Types of challenges of label {label} not yet scheduled (Prompt: {prompt}).")
            return [], []
        label = self.challenge_alias[label]

        # Image Splitting if Image-Bytes is provided, not list of Images
        if isinstance(images, bytes) or isinstance(images, Path):
            images, coordinates = self.handle_single_image(images, area_captcha)  # type: ignore

        if isinstance(images, list):
            if len(images) == 1:
                byte_images, coordinates = self.handle_single_image(images[0], area_captcha)
            else:
                byte_images = [img.read_bytes() if isinstance(img, Path) else img for img in images]

            if len(byte_images) not in (9, 16):
                print(f"[ERROR] Images amount must equal 9 or 16. Is: {len(byte_images)}")
                return [], []

            cv2_images = self.handle_multiple_images(byte_images)

            if not any(coordinates):
                height, width, _ = cv2_images[0].shape
                tiles_amount = 4 if area_captcha else 3
                coordinates = self.calculate_approximated_coords(height, width, tiles_amount)

            if label in self.yolo_detector.yolo_classes:
                # Creating Image Grid from List of Images
                cv2_image = create_image_grid(cv2_images)
                response = self.yolo_detector.detect_image(cv2_image, len(byte_images), label)
            else:
                response = self.clip_detector.detect_image(cv2_images, label)

        good_coordinates: List[Tuple[int, int]] = []
        for i, result in enumerate(response):
            if result:
                x, y = coordinates[i]
                good_coordinates.append((x+random.randint(-25, 25), y+random.randint(-25, 25)))

        return response, good_coordinates
