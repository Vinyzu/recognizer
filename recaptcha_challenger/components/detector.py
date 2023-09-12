# -*- coding: utf-8 -*-
# Author     : Vinyzu
# GitHub     : https://github.com/Vinyzu
# Description:
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union, Optional
import base64
import binascii
from contextlib import suppress
import math

import cv2
from cv2.dnn import Net
from onnxruntime import InferenceSession
from loguru import logger
from imageio import imread

from numpy.typing import NDArray
from numpy import uint8

from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8

from .prompt_handler import split_prompt_message
from .image_processor import get_captcha_fields, split_image_into_tiles, create_image_grid

CHALLENGE_ALIAS = {"car": "car", "cars": "car", "vehicles": "car",
                   "taxis": "taxi", "taxi": "taxi",
                   "bus": "bus", "buses": "bus",
                   "motorcycle": "motorcycle", "motorcycles": "motorcycle",
                   "bicycle": "bicycle", "bicycles": "bicycle",
                   "boats": "boat", "boat": "boat",
                   "tractors": "tractor", "tractor": "tractor",
                   "stairs": "stair", "stair": "stair",
                   "palm trees": "palm trees", "palm tree": "palm tree",
                   "fire hydrants": "fire hydrant", "a fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
                   "parking meters": "parking meter", "parking meter": "parking meter",
                   "crosswalks": "crosswalk", "crosswalk": "crosswalk",
                   "traffic lights": "traffic light", "traffic light": "traffic light",
                   "bridges": "bridge", "bridge": "bridge",
                   "mountains or hills": "mountain", "mountain or hill": "mountain", "mountain": "mountain", "mountains": "mountain", "hills": "mountain", "hill": "mountain",
                   "chimney": "factory with chimney", "chimneys": "factory with chimney"}

YOLO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

class Detector:
    def __init__(self):
        self.modelhub = ModelHub.from_github_repo()
        self.modelhub.parse_objects()

    def load_model(self, label: str, prompt: str) -> Net | InferenceSession | None:
        # Loading Models
        if label in YOLO_CLASSES:
            focus_name, classes = self.modelhub.apply_ash_of_war(ash=label)
            session = self.modelhub.match_net(focus_name=focus_name)
            if not session:
                logger.error(
                    f"ModelNotFound, please upgrade assets and flush yolo model", focus_name=focus_name
                )
                return None
            model = YOLOv8.from_pluggable_model(session, classes)
            # Lowering Thresholds for better results
            model.conf_threshold, model.iou_threshold = 0.2, 0.2
        else:
            focus_label = self.modelhub.label_alias.get(label)
            if not focus_label:
                logger.debug("Types of challenges not yet scheduled", label=label, prompt=prompt)
                return None

            focus_name = focus_label if focus_label.endswith(".onnx") else f"{focus_label}.onnx"
            net = self.modelhub.match_net(focus_name)
            model = ResNetControl.from_pluggable_model(net)

        return model

    def get_tiles_in_bounding_box(self, img: NDArray[uint8], tile_amount: int, point_start: Tuple[int, int], point_end: Tuple[int, int]) -> List[bool | None]:
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
                    tile_coverage_percentage = int(intersection_area / tile_area * 100)
                    # Checking if tile coverage is above 5% (human eyes can not detect object boundaries in this range)
                    if tile_coverage_percentage > 10:
                        tiles_in_bbox.append(True)
                    else:
                        tiles_in_bbox.append(None)

                else:
                    tiles_in_bbox.append(None)

        return tiles_in_bbox

    def handle_single_image(self, images: bytes, area_captcha: bool) -> List[bytes | None]:
        with suppress(binascii.Error):
            images = base64.b64decode(images, validate=True)

        # Image Bytes to Cv2
        rgba_img = imread(images)
        img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)

        # Image Splitting Presuming has white barriers
        images = get_captcha_fields(img)

        if len(images) == 1:
            # Turning bytes from get_captcha_fields back to Cv2
            rgba_img = imread(images[0])
            img = cv2.cvtColor(rgba_img, cv2.COLOR_BGR2RGB)

            # Either it is just a single image or no white barriers
            height, width, _ = img.shape

            if height > 200 and width > 200:
                tiles_amount = 4 if area_captcha else 3
                images = split_image_into_tiles(img, tiles_amount)

        return images

    def detect(self, prompt: str, images: Union[List[Path | bytes], bytes], area_captcha: Optional[bool] = None) -> List[bool | None]:
        response = []
        # Making best guess if its area_captcha if user did not specify
        area_captcha = "square" in prompt if area_captcha is None else area_captcha
        label = split_prompt_message(prompt)

        if label not in CHALLENGE_ALIAS:
            logger.debug("Types of challenges not yet scheduled", label=label, prompt=prompt)
        label = CHALLENGE_ALIAS[label]

        # Loading Models
        if label in YOLO_CLASSES:
            model = self.load_model(label, prompt)
            yolo_model = True
        else:
            model = self.load_model(label, prompt)
            yolo_model = False

        if not model:
            return response

        # Image Splitting if Image-Bytes is provided, not list of Images
        if isinstance(images, bytes):
            images = self.handle_single_image(images, area_captcha)

        if isinstance(images, list):
            if len(images) == 1:
                images = self.handle_single_image(images[0], area_captcha)

            if area_captcha and yolo_model:
                response = [None for _ in range(16)]
                # Turning Paths into Image Bytes
                images = [image.read_bytes() if isinstance(image, Path) and image.exists() else image for image in images]
                if len(images) != 16:
                    logger.error(f"Images amount must equal 16. Is: {len(images)}")
                    return response

                # Creating Image Grid from List of Images
                image_bytes, cv2_image = create_image_grid(images, 16)

                results = model(image=image_bytes, shape_type="bounding_box")

                for result in results:
                    res_label, point_start, point_end, certainty = result
                    # Checking if right result
                    if res_label == label:
                        tiles_in_bbox = self.get_tiles_in_bounding_box(cv2_image, len(images), point_start, point_end)
                        # Appending True Tiles to Response but not making True ones False again
                        response = [x or y for x, y in zip(response, tiles_in_bbox)]
            else:
                for image in images:
                    try:
                        if isinstance(image, Path):
                            if not image.exists():
                                response.append(None)
                                continue
                            image = image.read_bytes()

                        if isinstance(image, bytes):
                            with suppress(binascii.Error):
                                image = base64.b64decode(image, validate=True)

                            if yolo_model:
                                results = model(image=image, shape_type="point")
                                if any([result[0] == label for result in results]):
                                    response.append(True)
                                else:
                                    response.append(None)
                            else:
                                result = model.binary_classify(image)
                                response.append(result)
                        else:
                            response.append(None)
                    except Exception as err:
                        logger.debug(str(err), label=label, prompt=prompt)
                        response.append(None)

        return response
