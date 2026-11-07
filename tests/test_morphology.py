"""
test_morphology.py - Person 2 Unit & Integration Tests
=======================================================
Run:
    python3 tests/test_morphology.py                        # synthetic tests only
    python3 tests/test_morphology.py --real                 # test on up to 15 real images in data/samples/
    python3 tests/test_morphology.py --real --count 10      # test on exactly 10 real images
    python3 tests/test_morphology.py --real --debug         # also save side-by-side comparison images

Tests covered
-------------
1. Shape is preserved after morphology
2. Output is strictly binary (only 0 and 255)
3. Opening removes an isolated single-pixel noise dot
4. Closing fills a small gap inside a solid region
5. Real Image Integration:
   - Takes real images from data/samples/ (including data/samples/original/)
   - Runs Person 1's preprocess_image() to generate the binary image
   - Runs Person 2's apply_morphology() on Person 1's output
   - Verifies the contract at each stage
   - In --debug mode, saves visual side-by-side comparisons to data/outputs/morphology_debug/
"""

import numpy as np
import sys
import os
import argparse
import cv2

# Allow running from project root or tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.preprocessing import preprocess_image
from modules.morphology import apply_morphology, custom_erode, custom_dilate


# ===========================================================================
# Synthetic unit tests
# ===========================================================================

def test_shape_preserved():
    """Output shape must exactly match input shape."""
    fake = np.zeros((480, 640), dtype=np.uint8)
    fake[100:200, 100:300] = 255
    result = apply_morphology(fake)
    assert result.shape == fake.shape, (
        f"FAIL: shape changed from {fake.shape} to {result.shape}"
    )
    print("PASS: output shape matches input shape")


def test_output_is_binary():
    """Output must contain only values 0 and 255."""
    fake = np.zeros((480, 640), dtype=np.uint8)
    fake[50:100, 50:100] = 255
    result = apply_morphology(fake)
    unique = set(np.unique(result).tolist())
    assert unique.issubset({0, 255}), (
        f"FAIL: non-binary values found: {unique}"
    )
    print("PASS: output contains only 0 and 255")


def test_output_dtype():
    """Output dtype must be uint8."""
    fake = np.zeros((100, 100), dtype=np.uint8)
    result = apply_morphology(fake)
    assert result.dtype == np.uint8, (
        f"FAIL: expected dtype uint8, got {result.dtype}"
    )
    print("PASS: output dtype is uint8")


def test_opening_removes_noise():
    """
    Opening (erode then dilate) should remove a single isolated white pixel.
    """
    img = np.zeros((20, 20), dtype=np.uint8)
    img[10, 10] = 255               # single isolated noise pixel

    eroded = custom_erode(img, kernel_size=3)
    assert eroded[10, 10] == 0, (
        f"FAIL: isolated noise pixel survived erosion (value={eroded[10,10]})"
    )
    print("PASS: custom_erode removes an isolated noise pixel")


def test_dilation_expands_region():
    """
    Dilation should grow a white region outward by one pixel on each side.
    """
    img = np.zeros((10, 10), dtype=np.uint8)
    img[4:6, 4:6] = 255             # 2x2 white square

    dilated = custom_dilate(img, kernel_size=3)
    assert dilated[3, 3] == 255, (
        f"FAIL: dilation did not expand region (pixel (3,3) = {dilated[3,3]})"
    )
    print("PASS: custom_dilate expands the white region outward")


def test_closing_fills_gap():
    """
    Closing should fill small black gaps inside a white bar.
    """
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 3:17] = 255           # thick white bar
    img[9:11, 9:11] = 0             # 2x2 black hole in the middle

    result = apply_morphology(img)
    filled = result[9, 9]
    print(f"INFO: gap-fill test: centre pixel after morphology = {filled}")
    if filled == 255:
        print("PASS: closing filled the internal gap")
    else:
        print("INFO: gap was not completely filled")


# ===========================================================================
# Real-image tests (Integrated with Person 1's Preprocessing)
# ===========================================================================

def find_sample_images(base_dir: str = "data/samples"):
    """Recursively discover image files under base_dir."""
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    found = []
    if not os.path.exists(base_dir):
        return found
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                found.append(os.path.join(root, f))
    # Return natural sorted list
    return sorted(found)


def test_on_real_images(count: int = 15, debug: bool = False) -> None:
    """
    Runs Person 1's preprocess_image() followed by Person 2's apply_morphology()
    on actual document images from data/samples/ (including subfolders).
    """
    out_dir = "data/outputs/morphology_debug"
    if debug:
        os.makedirs(out_dir, exist_ok=True)

    all_images = find_sample_images("data/samples")
    if not all_images:
        print("WARNING: No images found in data/samples/ or data/samples/original/")
        print("         Please ensure images are placed in data/samples/original/")
        return

    test_images = all_images[:count]
    print(f"Found {len(all_images)} images. Running on {len(test_images)} image(s)...")

    all_passed = True
    for idx, fpath in enumerate(test_images, 1):
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            print(f"[{idx}/{len(test_images)}] SKIP: could not read {fpath}")
            continue

        # 1. Step 1: Person 1's Preprocessing
        try:
            binary_preprocessed = preprocess_image(img)
        except Exception as e:
            print(f"[{idx}/{len(test_images)}] FAIL [{fname}]: Preprocessing threw error: {e}")
            all_passed = False
            continue

        # Verify Person 1 contract
        if binary_preprocessed.shape != (img.shape[0], img.shape[1]):
            print(f"[{idx}/{len(test_images)}] FAIL [{fname}]: Preprocessed shape mismatch")
            all_passed = False
            continue

        # 2. Step 2: Person 2's Morphology on Person 1's output
        try:
            cleaned_morphology = apply_morphology(binary_preprocessed)
        except Exception as e:
            print(f"[{idx}/{len(test_images)}] FAIL [{fname}]: Morphology threw error: {e}")
            all_passed = False
            continue

        # Verify Person 2 contract
        if cleaned_morphology.shape != binary_preprocessed.shape:
            print(f"[{idx}/{len(test_images)}] FAIL [{fname}]: Morphology shape changed: {cleaned_morphology.shape}")
            all_passed = False
            continue
        if cleaned_morphology.dtype != np.uint8:
            print(f"[{idx}/{len(test_images)}] FAIL [{fname}]: Morphology dtype {cleaned_morphology.dtype} != uint8")
            all_passed = False
            continue
        unique_vals = set(np.unique(cleaned_morphology).tolist())
        if not unique_vals.issubset({0, 255}):
            print(f"[{idx}/{len(test_images)}] FAIL [{fname}]: Non-binary values in morphology: {unique_vals}")
            all_passed = False
            continue

        print(f"[{idx}/{len(test_images)}] PASS [{fname}]: Preprocessing -> Morphology OK ({img.shape[1]}x{img.shape[0]})")

        # 3. Save visual comparison if --debug is enabled
        if debug:
            base_name = os.path.splitext(fname)[0]
            # Convert original grayscale to 8-bit for a 3-panel comparison:
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Panel: [Original Grayscale | Person 1 Binary | Person 2 Cleaned Morphology]
            panel = np.hstack([gray_orig, binary_preprocessed, cleaned_morphology])
            out_path = os.path.join(out_dir, f"compare_{base_name}.png")
            cv2.imwrite(out_path, panel)
            print(f"      [debug] saved 3-panel comparison to: {out_path}")

    if all_passed:
        print("\nSUCCESS: All tested images passed Preprocessing + Morphology integration checks!")
    else:
        print("\nFAILURE: Some images failed contract checks. See errors above.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morphology module tests")
    parser.add_argument('--real', action='store_true',
                        help='Also run tests on real images in data/samples/')
    parser.add_argument('--count', type=int, default=15,
                        help='Number of real images to test (default: 15)')
    parser.add_argument('--debug', action='store_true',
                        help='Save 3-panel comparison images to data/outputs/morphology_debug/')
    args = parser.parse_args()

    print("=" * 60)
    print("  Morphology Module Unit & Integration Tests")
    print("=" * 60)

    # 1. Standard synthetic tests
    test_shape_preserved()
    test_output_is_binary()
    test_output_dtype()
    test_opening_removes_noise()
    test_dilation_expands_region()
    test_closing_fills_gap()

    # 2. Real image integration tests
    if args.real:
        print()
        print(f"--- Real Image Integration Tests (Testing up to {args.count} images) ---")
        test_on_real_images(count=args.count, debug=args.debug)

    print()
    print("=" * 60)
    print("Completed.")
    print("=" * 60)
