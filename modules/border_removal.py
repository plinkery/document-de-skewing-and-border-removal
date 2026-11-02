"""
border_removal.py - Person 4
==============================
Responsibility:
  1. Detect the document boundary in the deskewed image and crop it.
  2. Manage main.py and integration testing.

INPUT  received from: deskew.py via deskew_image()
OUTPUT returned to:   main.py (the final result)

CONTRACT - do not change without telling the team:
    remove_borders(img: np.ndarray) -> np.ndarray
        IN:  Deskewed BGR image,  shape (H, W, 3),  dtype uint8
        OUT: Cropped BGR image,   shape (h, w, 3),  dtype uint8   (h <= H, w <= W)

Your topics to implement:
    - Row and column projection scanning (your own algorithm)
    - Finding crop coordinates from intensity sums
    - Cropping with numpy slicing
"""

import cv2
import numpy as np


def remove_borders(img: np.ndarray) -> np.ndarray:
    """
    Detect and remove the surrounding border from a deskewed document image.

    Parameters
    ----------
    img : np.ndarray
        Deskewed BGR image from deskew_image(). Shape: (H, W, 3).

    Returns
    -------
    np.ndarray
        Cropped image containing only the document. Shape: (h, w, 3).
    """
    # STARTER CODE (returns image unchanged) - Person 4: replace with real implementation
    return img
