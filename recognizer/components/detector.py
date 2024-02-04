from __future__ import annotations

import math
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
from numpy import generic, uint8
from numpy.typing import NDArray
from torch import no_grad, set_num_threads

from .detection_processor import calculate_approximated_coords, calculate_segmentation_response, get_tiles_in_bounding_box
from .image_processor import create_image_grid, handle_multiple_images, handle_single_image
from .prompt_handler import split_prompt_message

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


class DetectionModels:
    def __init__(self) -> None:
        # Preloading: Loading Models takes ~9 seconds
        set_num_threads(2)
        self.executor = ThreadPoolExecutor(max_workers=2)

        try:
            self.yolo_loading_feature = self.executor.submit(self._load_yolo_detector)
            self.clip_loading_feature = self.executor.submit(self._load_clip_detector)
        except Exception as e:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise e

    def _load_yolo_detector(self):
        from ultralytics import YOLO

        self.yolo_model = YOLO("yolov8m-seg.pt")

    def _load_clip_detector(self):
        from transformers import CLIPModel, CLIPProcessor, CLIPSegForImageSegmentation, CLIPSegProcessor

        self.vit_model = CLIPModel.from_pretrained("flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")
        self.vit_processor = CLIPProcessor.from_pretrained("flavour/CLIP-ViT-B-16-DataComp.XL-s13B-b90K")

        self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    def check_loaded(self):
        try:
            if not self.yolo_loading_feature.done() or not self.clip_loading_feature.done():
                self.yolo_loading_feature.result()
                self.clip_loading_feature.result()

            assert self.yolo_model
            assert self.seg_model
            assert self.vit_model
        except Exception as e:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise e


detection_models = DetectionModels()


class YoloDetector:
    # fmt: off
    yolo_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # fmt: on

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
        pass

    def detect_image(self, image: NDArray[generic], tile_amount: int, task_type: str) -> List[bool]:
        response = [False for _ in range(tile_amount)]
        height, width, _ = image.shape
        tiles_per_row = int(math.sqrt(tile_amount))
        tile_width, tile_height = width // tiles_per_row, height // tiles_per_row

        outputs = detection_models.yolo_model.predict(image, verbose=False, conf=0.2, iou=0.3)  # , save=True
        results = outputs[0]

        for result in results:
            assert result
            # Check if correct task type
            class_index = int(result.boxes.cls[0])
            if self.yolo_classes[class_index] not in self.yolo_alias[task_type]:
                continue

            masks = result.masks
            mask = masks.xy[0]
            response = calculate_segmentation_response(mask, response, tile_width, tile_height, tiles_per_row)

            # In AreaCaptcha Mode, Calculate Tiles inside of boundary, which arent covered by mask point
            if tile_amount == 16:
                coords = result.boxes.xyxy.flatten().tolist()
                points_start, point_end = coords[:2], coords[2:]
                tiles_in_bbox = get_tiles_in_bounding_box(image, tile_amount, tuple(points_start), tuple(point_end))
                # Appending True Tiles to Response but not making True ones False again
                response = [x or y for x, y in zip(response, tiles_in_bbox)]

        return response


class ClipDetector:
    # fmt: off
    plain_labels = ["bicycle", "boat", "bus", "car", "fire hydrant", "motorcycle", "traffic light",  # YOLO TASKS
                    "bridge", "chimney", "crosswalk", "mountain", "palm tree", "stair", "tractor", "taxi"]

    all_labels = ["a bicycle", "a boat", "a bus", "a car", "a fire hydrant", "a motorcycle", "a traffic light",  # YOLO TASKS
                  "the front or bottom or side of a concrete or steel bridge supported by concrete pillars over a street or highway",
                  "a chimney on the roof or top of a house or building",
                  "white/yellow stripes or white/yellow lines of a crosswalk stretching over a gray ground of a street",
                  "a green or gray mountain or grassy hill in the background of a landscape",  # Quite Bad Accuracy, might need another approach
                  "a palm tree behind a rooftop of a house or next to a street stretching into the sky",
                  "a concrete or steel or wooden stairway or stairs or steps with railings on its side in front of a house or building leading to the street",
                  "a tractor or agricultural vehicle driving on a street or field",
                  "a taxi or a yellow car"]
    # fmt: on

    thresholds = {
        "bridge": 0.7285372716747225,
        "chimney": 0.7918647485226393,
        "crosswalk": 0.8879293048381806,
        "mountain": 0.5551278884819476,
        "palm tree": 0.8093279512040317,
        "stair": 0.7312694561691023,
        "tractor": 0.9385110986077537,
        "taxi": 0.7967491503432393,
    }

    area_captcha_labels = {
        "bridge": "the front or bottom or side of a concrete or steel bridge supported by concrete pillars",
        "chimney": "a chimney on the roof or top of a house or building",
        "crosswalk": "white/yellow stripes or white/yellow lines of a crosswalk",
        "mountain": "a green or gray mountain or grassy hill",
        "palm tree": "a palm tree",
        "stair": "a concrete or steel or wooden stairway or stairs or steps with railings on its side",
        "tractor": "a tractor or agricultural vehicle",
        "taxi": "a yellow car or taxi",
    }

    def __init__(self) -> None:
        pass

    def clip_detect_vit(self, images: List[NDArray[generic]], task_type: str) -> List[bool]:
        response = []

        inputs = detection_models.vit_processor(text=self.all_labels, images=images, return_tensors="pt", padding=True)
        with no_grad():
            outputs = detection_models.vit_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        results = probs.tolist()

        for result in results:
            task_index = self.plain_labels.index(task_type)
            prediction = result[task_index]
            choice = prediction >= (self.thresholds[task_type] - 0.4)

            response.append(choice)

        return response

    def clipseg_detect_rd64(self, image: NDArray[generic], task_type: str, tiles_amount: int) -> List[bool]:
        response = [False for _ in range(tiles_amount)]
        segment_label = self.area_captcha_labels[task_type]

        inputs = detection_models.seg_processor(text=segment_label, images=[image], padding="max_length", return_tensors="pt")
        with no_grad():
            outputs = detection_models.seg_model(**inputs)

        heatmap = outputs.logits
        # Get the normalized adjusted threshold for the heatmap
        adjusted_normalized_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 2.5
        threshold = heatmap.max() - adjusted_normalized_heatmap.mean()
        # Create the Threshold mask for the Heatmap
        threshold_mask = (heatmap > threshold).float()

        # Getting Tile Size from threshold mask
        tiles_per_row = int(math.sqrt(tiles_amount))
        mask_width, mask_height = heatmap.shape
        tile_width, tile_height = (mask_width // tiles_per_row, mask_height // tiles_per_row)

        # Creating Contours of Threshold Mask
        threshold_image = threshold_mask.numpy().astype(uint8)
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_contours = []
        for contour in contours:
            # Checking Image area to only get large captcha areas (no image details)
            area = cv2.contourArea(contour)
            if area > 100:
                new_contours.append(contour)
                mask = contour.squeeze()
                response = calculate_segmentation_response(mask, response, tile_width, tile_height, tiles_per_row)

        return response

    def detect_image(self, images: List[NDArray[generic]], task_type: str) -> List[bool]:
        if len(images) == 9:
            return self.clip_detect_vit(images, task_type)

        tile_height, tile_width, _ = images[0].shape
        combined_image = create_image_grid(images)
        return self.clipseg_detect_rd64(combined_image, task_type, len(images))


class Detector:
    # fmt: off
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

    # fmt: on

    def __init__(self) -> None:
        """
        Spawn a new reCognizer Detector Instance
        """
        self.yolo_detector = YoloDetector()
        self.clip_detector = ClipDetector()

    def detect(
        self,
        prompt: str,
        images: Union[Path, Union[PathLike[str], str], bytes, Sequence[Path], Sequence[Union[PathLike[str], str]], Sequence[bytes]],
        area_captcha: Optional[bool] = None,
    ) -> Tuple[List[bool], List[Tuple[int, int]]]:
        """
        Create a new Botright browser instance with specified configurations.

        Args:
            prompt (str): The prompt name/sentence of the captcha (e.g. "Select all images with crosswalks" / "crosswalk").
            images (Path | PathLike | bytes | Sequence[Path] | Sequence[PathLike] | Sequence[bytes]): The Image(s) to reCognize.
            area_captcha (bool, optional): Whether the Captcha Task is an area-captcha.

        Returns:
            List[bool], List[Tuple[int, int]]: The reCognizer Response and calculated click-coordinates for the response
        """
        detection_models.check_loaded()

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
        if isinstance(images, (PathLike, str)):
            images = Path(images)

        if isinstance(images, bytes) or isinstance(images, Path):
            images, coordinates = handle_single_image(images, area_captcha)  # type: ignore

        if isinstance(images, list):
            if len(images) == 1:
                if isinstance(images[0], (PathLike, str)):
                    pathed_image = Path(images[0])
                    byte_images, coordinates = handle_single_image(pathed_image, area_captcha)
                else:
                    byte_images, coordinates = handle_single_image(images[0], area_captcha)
            else:
                byte_images = []
                for image in images:
                    if isinstance(image, Path):
                        byte_images.append(image.read_bytes())
                    elif isinstance(image, (PathLike, str)):
                        pathed_image = Path(image)
                        byte_images.append(pathed_image.read_bytes())
                    else:
                        byte_images.append(image)

            if len(byte_images) not in (9, 16):
                print(f"[ERROR] Images amount must equal 9 or 16. Is: {len(byte_images)}")
                return [], []

            cv2_images = handle_multiple_images(byte_images)

            if not any(coordinates):
                height, width, _ = cv2_images[0].shape
                tiles_amount = 4 if area_captcha else 3
                coordinates = calculate_approximated_coords(height, width, tiles_amount)

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
                good_coordinates.append((x + random.randint(-25, 25), y + random.randint(-25, 25)))

        return response, good_coordinates
