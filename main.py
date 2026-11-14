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
from pathlib import Path


# ---- Import all four modules -----------------------------------------------
from modules.preprocessing  import preprocess_image
from modules.morphology      import apply_morphology
from modules.deskew          import deskew_image
from modules.border_removal  import remove_borders
from modules.interactive_crop import interactive_refine_crop


SAMPLE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def discover_sample_images(samples_dir: str = "data/samples") -> list[str]:
    """Return supported sample-image paths recursively in stable order."""
    root = Path(samples_dir)
    if not root.is_dir():
        return []

    paths = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SAMPLE_EXTENSIONS
    ]
    return sorted(paths, key=str.casefold)


def _resolve_image_path(image_path: str) -> str:
    """Find the image path even if relative to data/samples/ or case differs."""
    if os.path.exists(image_path):
        return image_path

    # Try common subfolder locations
    candidates = [
        os.path.join("data/samples", image_path),
        os.path.join("data/samples/original", image_path),
        os.path.join("data/samples/original", os.path.basename(image_path)),
        os.path.join("data/samples", os.path.basename(image_path)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Case-insensitive / recursive search in data/samples
    target_base = os.path.basename(image_path).lower()
    for sample in discover_sample_images():
        if os.path.basename(sample).lower() == target_base:
            return sample

    return image_path


# ---- Helper: save an intermediate debug image ------------------------------

def _save_debug(img: np.ndarray, name: str, out_dir: str) -> None:
    """Save an intermediate image to out_dir for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    # Binary images have shape (H, W); colour images have shape (H, W, 3)
    cv2.imwrite(path, img)
    print(f"      [debug] saved: {path}")


def generate_comparison_panel(
    raw_img: np.ndarray,
    binary_img: np.ndarray,
    cleaned_img: np.ndarray,
    deskewed_img: np.ndarray,
    auto_cropped_img: np.ndarray,
    angle: float,
    out_path: str,
    user_cropped_img: np.ndarray | None = None,
    gray_img: np.ndarray | None = None,
    target_h: int = 600,
) -> str:
    """
    Generate and save a multi-panel comparison strip:
    - Automated mode: 4 panels:
        [1. Grayscale | 2. Binary (P1) | 3. Cleaned (P2) | 4. Deskewed + Border Removed (P3+P4)]
    - Interactive mode: 5 panels:
        [1. Grayscale | 2. Binary (P1) | 3. Cleaned (P2) | 4. Deskewed + Border Removed (P3+P4) | 5. Custom Crop]
    """
    # Prepare grayscale image if not provided
    if gray_img is None:
        if raw_img.ndim == 3:
            gray_display = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            gray_display = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR)
        else:
            gray_display = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    else:
        gray_display = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) if gray_img.ndim == 2 else gray_img

    images = [
        (gray_display, f"1. Grayscale ({gray_display.shape[1]}x{gray_display.shape[0]})"),
        (cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR), "2. Binary (P1)"),
        (cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR), "3. Cleaned (P2)"),
        (auto_cropped_img, f"4. Deskewed + Border Removed ({auto_cropped_img.shape[1]}x{auto_cropped_img.shape[0]})"),
    ]

    if user_cropped_img is not None:
        images.append(
            (user_cropped_img, f"5. Custom Crop ({user_cropped_img.shape[1]}x{user_cropped_img.shape[0]})")
        )

    panels = []
    for img, title in images:
        h, w = img.shape[:2]
        new_w = max(int(w * (target_h / h)), 100)
        resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        # Header banner
        banner_h = 36
        banner = np.full((banner_h, new_w, 3), (35, 35, 35), dtype=np.uint8)
        cv2.putText(banner, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.vstack([banner, resized])
        panels.append(panel)

    divider = np.full((panels[0].shape[0], 4, 3), (180, 180, 180), dtype=np.uint8)
    combined = []
    for i, p in enumerate(panels):
        combined.append(p)
        if i < len(panels) - 1:
            combined.append(divider)

    full_strip = np.hstack(combined)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, full_strip)
    return out_path


# ---- Main pipeline function ------------------------------------------------

def run_pipeline(image_path: str, debug: bool = False, interactive: bool = False) -> np.ndarray:
    """
    Runs the full document de-skewing and border removal pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    debug : bool
        If True, intermediate images are saved after each step.
    interactive : bool
        If True, launches an interactive GUI window allowing the user to
        manually drag and adjust the crop corners before perspective transformation.

    Returns
    -------
    np.ndarray
        The final corrected image as a NumPy array (BGR format, 3 channels).
    """
    debug_dir = "data/outputs/debug"

    # ---- Load the original image -------------------------------------------
    resolved_path = _resolve_image_path(image_path)
    raw_img = cv2.imread(resolved_path)
    if raw_img is None:
        raise FileNotFoundError(
            f"Could not load image from '{image_path}' (resolved to '{resolved_path}').\n"
            f"Make sure the path is correct and the file exists in data/samples/."
        )
    print(f"Loaded: {resolved_path}  shape={raw_img.shape}")
    # Save separate full images of each step when debug is enabled
    gray_img = None
    if debug:
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        _save_debug(raw_img, "0_original.jpg", debug_dir)
        _save_debug(gray_img, "0_grayscale.jpg", debug_dir)

    # Determine total steps depending on interactive mode
    total_steps = 5 if interactive else 4

    # ---- Step 1: Preprocessing (Person 1) ----------------------------------
    print(f"[1/{total_steps}] Preprocessing image (grayscale + threshold)...")
    binary_img = preprocess_image(raw_img)
    _validate_binary(binary_img, step="preprocessing")
    if debug:
        _save_debug(binary_img, "1_binary.jpg", debug_dir)

    # ---- Step 2: Morphology (Person 2) -------------------------------------
    print(f"[2/{total_steps}] Applying morphological operations (noise removal)...")
    cleaned_img = apply_morphology(binary_img)
    _validate_binary(cleaned_img, step="morphology")
    if debug:
        _save_debug(cleaned_img, "2_cleaned_binary.jpg", debug_dir)

    # ---- Step 3: Deskew (Person 3) -----------------------------------------
    print(f"[3/{total_steps}] Detecting skew angle and rotating image...")
    deskewed_img, angle = deskew_image(raw_img, cleaned_img)
    print(f"      Detected skew angle: {angle:.2f} degrees")
    _validate_colour(deskewed_img, step="deskew")
    if debug:
        _save_debug(deskewed_img, "3_deskewed.jpg", debug_dir)

    # ---- Step 4: Border removal (Person 4) ---------------------------------
    print(f"[4/{total_steps}] Detecting document boundary and removing borders...")
    auto_cropped = remove_borders(deskewed_img)
    _validate_colour(auto_cropped, step="border_removal")

    user_cropped = None
    if interactive:
        print(f"[5/{total_steps}] Launching interactive crop tool for user refinement...")
        user_cropped, _ = interactive_refine_crop(deskewed_img)
        _validate_colour(user_cropped, step="interactive_crop")

    final_output = user_cropped if user_cropped is not None else auto_cropped

    if debug:
        _save_debug(auto_cropped, "4_final_auto.jpg", debug_dir)
        if user_cropped is not None:
            _save_debug(user_cropped, "5_final_user_refined.jpg", debug_dir)

        compare_path = os.path.join(debug_dir, "pipeline_comparison.png")
        generate_comparison_panel(
            raw_img,
            binary_img,
            cleaned_img,
            deskewed_img,
            auto_cropped,
            angle,
            compare_path,
            user_cropped_img=user_cropped,
            gray_img=gray_img
        )
        print(f"      [debug] saved comparison strip: {compare_path}")

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
        default=None,
        help="Path to the input image (defaults to the first image under data/samples/)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images after each pipeline step"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive GUI to drag corners and manually refine document boundaries"
    )
    args = parser.parse_args()
    if args.image_path is None:
        sample_images = discover_sample_images()
        if not sample_images:
            parser.error(
                "no sample images found under data/samples; provide an image path"
            )
        args.image_path = sample_images[0]

    print("=" * 55)
    print("  Document De-skew & Border Removal Pipeline")
    print("=" * 55)

    result = run_pipeline(args.image_path, debug=args.debug, interactive=args.interactive)

    os.makedirs("data/outputs", exist_ok=True)
    output_path = "data/outputs/result.jpg"
    cv2.imwrite(output_path, result)

    print("=" * 55)
    print(f"Done. Final output saved to: {output_path}")
    if args.debug:
        print("Debug images saved to: data/outputs/debug/")
    print("=" * 55)
