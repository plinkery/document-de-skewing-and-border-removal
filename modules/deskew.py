"""
deskew.py - Person 3
======================
Responsibility: Detect the skew angle and rotate the image to straighten it.

INPUT  received from: main.py (both raw image and cleaned binary image)
OUTPUT sent to:       border_removal.py via remove_borders()

CONTRACT - do not change without telling the team:
    deskew_image(raw_img, binary_img) -> tuple[np.ndarray, float]
        IN:  raw_img    - original BGR image,  shape (H, W, 3),  dtype uint8
             binary_img - cleaned binary,       shape (H, W),     dtype uint8
        OUT: (rotated_image, angle)
             rotated_image - BGR image, straightened,  shape (H, W, 3), dtype uint8
             angle         - the skew angle in degrees (float)

        IMPORTANT: return order is (image, angle) - not (angle, image)

Your topics to implement:
    - Canny edge detection
    - Hough Transform (cv2.HoughLinesP)
    - Custom angle-filtering logic to pick the dominant skew angle
    - Image rotation using cv2.warpAffine
"""

import cv2
import numpy as np


def deskew_image(raw_img: np.ndarray, binary_img: np.ndarray) -> tuple:
    """
    Detect skew angle and rotate the image to straighten the document.

    Parameters
    ----------
    raw_img : np.ndarray
        Original BGR image. Shape: (H, W, 3).
    binary_img : np.ndarray
        Cleaned binary image from apply_morphology(). Shape: (H, W).

    Returns
    -------
    tuple[np.ndarray, float]
        (rotated_image, angle)
        rotated_image: Straightened BGR image. Shape: (H, W, 3).
        angle: The skew angle in degrees that was corrected. Type: float.
    """
    # STARTER CODE (returns unchanged image, angle = 0.0) - Person 3: replace with real implementation
    return raw_img, 0.0
