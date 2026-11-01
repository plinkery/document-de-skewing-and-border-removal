"""
main.py - Integration Pipeline
================================
This file connects all four modules into a single pipeline.
Person 4 is the gatekeeper of this file, but Person 2 has set it up
with a complete working skeleton.

Run (basic):
    python3 main.py data/samples/your_image.jpg

Run (debug mode - saves intermediate images after every step):
    python3 main.py data/samples/your_image.jpg --debug

How the pipeline works
----------------------
raw image
    |
    v  modules/preprocessing.py  (Person 1)
binary image
    |
    v  modules/morphology.py     (Person 2)
cleaned binary image
    |
    v  modules/deskew.py         (Person 3)   <-- also receives raw image
deskewed colour image + angle
    |
    v  modules/border_removal.py (Person 4)
final cropped image  -->  saved to data/outputs/result.jpg
"""

import cv2
import numpy as np
import sys
import os
import argparse


# ---- Import all four modules -----------------------------------------------
from modules.preprocessing  import preprocess_image
from modules.morphology      import apply_morphology
from modules.deskew          import deskew_image
from modules.border_removal  import remove_borders


# ---- Helper: save an intermediate debug image ------------------------------

def _save_debug(img: np.ndarray, name: str, out_dir: str) -> None:
    """Save an intermediate image to out_dir for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    # Binary images have shape (H, W); colour images have shape (H, W, 3)
    cv2.imwrite(path, img)
    print(f"      [debug] saved: {path}")


# ---- Main pipeline function ------------------------------------------------

def run_pipeline(image_path: str, debug: bool = False) -> np.ndarray:
    """
    Runs the full document de-skewing and border removal pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    debug : bool
        If True, intermediate images are saved after each step so you can
        inspect exactly what each module does.

    Returns
    -------
    np.ndarray
        The final corrected image as a NumPy array (BGR format, 3 channels).
    """
    debug_dir = "data/outputs/debug"

    # ---- Load the original image -------------------------------------------
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}\n"
            f"Make sure the path is correct and the file exists."
        )
    print(f"Loaded: {image_path}  shape={raw_img.shape}")
    if debug:
        _save_debug(raw_img, "0_original.jpg", debug_dir)

    # ---- Step 1: Preprocessing (Person 1) ----------------------------------
    print("[1/4] Preprocessing image (grayscale + threshold)...")
    binary_img = preprocess_image(raw_img)
    _validate_binary(binary_img, step="preprocessing")
    if debug:
        _save_debug(binary_img, "1_binary.jpg", debug_dir)

    # ---- Step 2: Morphology (Person 2) -------------------------------------
    print("[2/4] Applying morphological operations (noise removal)...")
    cleaned_img = apply_morphology(binary_img)
    _validate_binary(cleaned_img, step="morphology")
    if debug:
        _save_debug(cleaned_img, "2_cleaned_binary.jpg", debug_dir)

    # ---- Step 3: Deskew (Person 3) -----------------------------------------
    print("[3/4] Detecting skew angle and rotating image...")
    deskewed_img, angle = deskew_image(raw_img, cleaned_img)
    print(f"      Detected skew angle: {angle:.2f} degrees")
    _validate_colour(deskewed_img, step="deskew")
    if debug:
        _save_debug(deskewed_img, "3_deskewed.jpg", debug_dir)

    # ---- Step 4: Border removal (Person 4) ---------------------------------
    print("[4/4] Detecting document boundary and removing borders...")
    final_output = remove_borders(deskewed_img)
    _validate_colour(final_output, step="border_removal")
    if debug:
        _save_debug(final_output, "4_final.jpg", debug_dir)

    return final_output


# ---- Validation helpers (catch wrong return types early) -------------------

def _validate_binary(img: np.ndarray, step: str) -> None:
    """Raise a clear error if a module did not return a valid binary image."""
    if img is None:
        raise ValueError(f"[{step}] returned None - did you forget 'return'?")
    if img.ndim != 2:
        raise ValueError(
            f"[{step}] expected a 2-D binary image (H, W), got shape {img.shape}.\n"
            f"Make sure you return a grayscale/binary array, not a colour one."
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"[{step}] expected dtype uint8, got {img.dtype}.\n"
            f"Convert with: img = img.astype(np.uint8)"
        )
    unique = set(np.unique(img).tolist())
    if not unique.issubset({0, 255}):
        raise ValueError(
            f"[{step}] binary image must contain only 0 and 255. Got: {unique}.\n"
            f"Run a threshold after your operation to ensure binary values."
        )


def _validate_colour(img: np.ndarray, step: str) -> None:
    """Raise a clear error if a module did not return a valid BGR colour image."""
    if img is None:
        raise ValueError(f"[{step}] returned None - did you forget 'return'?")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"[{step}] expected a 3-channel colour image (H, W, 3), got shape {img.shape}.\n"
            f"Make sure you return the BGR colour image, not the binary one."
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"[{step}] expected dtype uint8, got {img.dtype}."
        )


# ---- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document De-skew and Border Removal Pipeline"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="data/samples/sample1.jpg",
        help="Path to the input image (default: data/samples/sample1.jpg)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images after each pipeline step"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Document De-skew & Border Removal Pipeline")
    print("=" * 55)

    result = run_pipeline(args.image_path, debug=args.debug)

    os.makedirs("data/outputs", exist_ok=True)
    output_path = "data/outputs/result.jpg"
    cv2.imwrite(output_path, result)

    print("=" * 55)
    print(f"Done. Final output saved to: {output_path}")
    if args.debug:
        print("Debug images saved to: data/outputs/debug/")
    print("=" * 55)
