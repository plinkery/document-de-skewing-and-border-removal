"""
preprocessing.py - Person 1
============================
Responsibility: Convert the raw input image into a clean binary image.

INPUT  received from: main.py (raw cv2.imread result)
OUTPUT sent to:       morphology.py via apply_morphology()

CONTRACT - do not change without telling the team:
    preprocess_image(img: np.ndarray) -> np.ndarray
        IN:  BGR colour image,  shape (H, W, 3),  dtype uint8
        OUT: Binary image,      shape (H, W),     dtype uint8, values = 0 or 255

Your topics to implement:
    - Grayscale conversion
    - Histogram analysis for threshold selection (custom algorithm)
    - Thresholding
"""

import cv2
import numpy as np


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR image to a binary (black-and-white) image.

    Parameters
    ----------
    img : np.ndarray
        Raw BGR image loaded by cv2.imread(). Shape: (H, W, 3)

    Returns
    -------
    np.ndarray
        Binary image. Shape: (H, W). Dtype: uint8. Values: 0 or 255.
    """
    # STARTER CODE (placeholder) - Person 1: replace this with your real implementation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
