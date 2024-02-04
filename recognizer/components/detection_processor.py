import math
from typing import List, Tuple, Union

from numpy import generic
from numpy.typing import NDArray


def calculate_segmentation_response(mask: Union[NDArray[generic], List[Tuple[int, int]]], response: List[bool], tile_width: int, tile_height: int, tiles_per_row: int) -> List[bool]:
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
        # fmt: off
        within_boundary = (
                boundary_x <= coord[1] % tile_width <= tile_width - boundary_x
                and boundary_y <= coord[0] % tile_height <= tile_height - boundary_y
        )
        # fmt: on

        if within_boundary:
            response[tile_index] = True

    return response


def get_tiles_in_bounding_box(img: NDArray[generic], tile_amount: int, point_start: Tuple[int, int], point_end: Tuple[int, int]) -> List[bool]:
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


def calculate_approximated_coords(grid_width: int, grid_height: int, tile_amount: int) -> List[Tuple[int, int]]:
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
