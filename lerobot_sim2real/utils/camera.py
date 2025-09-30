import numpy as np


def scale_intrinsics(
    intrinsics, original_width, original_height, new_width, new_height
):
    """
    Given an 3x3 intrinsics matrix rescale it to a new width and height. Will rescale such that the center
    of the image stays the same and there is no additional distortion, effectively equivalent to a center crop.
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # First we crop only as little as possible on just one side to match the new width/height ratio
    # cx, cy are linearly reduced cropping from the left and top
    if original_width / new_width > original_height / new_height:
        crop_width = original_height * (new_width / new_height)
        cx = cx - (original_width - crop_width) / 2
    else:
        crop_height = original_width * (new_height / new_width)
        cy = cy - (original_height - crop_height) / 2
    factor = new_height / original_height
    fx = fx * factor
    fy = fy * factor
    cx = cx * factor
    cy = cy * factor
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
