"""
morphology.py - Person 2
==========================
Responsibility: Clean up the binary image by removing noise and
strengthening useful edges/text regions.

INPUT  received from: preprocessing.py via preprocess_image()
OUTPUT sent to:       deskew.py via deskew_image()

CONTRACT - do not change without telling the team:
    apply_morphology(binary_img: np.ndarray) -> np.ndarray
        IN:  Binary image,  shape (H, W),  dtype uint8, values = 0 or 255
        OUT: Binary image,  shape (H, W),  dtype uint8, values = 0 or 255

Your topics to implement:
    - Structuring elements
    - Erosion, dilation, opening, closing
    - At least one custom morphological operation (not just cv2.morphologyEx)
"""

import cv2
import numpy as np


def apply_morphology(binary_img: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean the binary image.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image from preprocess_image(). Shape: (H, W). Values: 0 or 255.

    Returns
    -------
    np.ndarray
        Cleaned binary image. Same shape and dtype as input.
    """
    # STARTER CODE (passthrough placeholder) - Person 2: replace with real implementation
    return binary_img
