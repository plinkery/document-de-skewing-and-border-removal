"""
main.py - Integration Pipeline
================================
This file connects all four modules into a single pipeline.
Person 4 is the gatekeeper of this file.
Nobody else should edit main.py directly.

Run:
    python3 main.py data/samples/your_image.jpg
"""

import cv2
import numpy as np
import sys
import os

from modules.preprocessing  import preprocess_image
from modules.morphology     import apply_morphology
from modules.deskew         import deskew_image
from modules.border_removal import remove_borders


def run_pipeline(image_path: str) -> np.ndarray:
    """
    Runs the full document de-skewing and border removal pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image file.

    Returns
    -------
    np.ndarray
        The final corrected image as a NumPy array (BGR format).
    """
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print("[1/4] Preprocessing image...")
    binary_img = preprocess_image(raw_img)

    print("[2/4] Applying morphological operations...")
    cleaned_img = apply_morphology(binary_img)

    print("[3/4] Detecting skew and rotating...")
    deskewed_img, angle = deskew_image(raw_img, cleaned_img)
    print(f"      Detected angle: {angle:.2f} degrees")

    print("[4/4] Removing borders...")
    final_output = remove_borders(deskewed_img)

    return final_output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_path = "data/samples/sample1.jpg"
        print(f"No path given. Using default: {image_path}")
    else:
        image_path = sys.argv[1]

    result = run_pipeline(image_path)

    os.makedirs("data/outputs", exist_ok=True)
    output_path = "data/outputs/result.jpg"
    cv2.imwrite(output_path, result)
    print(f"Done. Saved to: {output_path}")
