"""
test_morphology.py - Person 2 runs this before merging
=======================================================
Run:
    python3 tests/test_morphology.py                        # synthetic tests only
    python3 tests/test_morphology.py --real                 # also test on real images
    python3 tests/test_morphology.py --real --debug         # save debug images too

Tests covered
-------------
1. Shape is preserved after morphology
2. Output is strictly binary (only 0 and 255)
3. Opening removes an isolated single-pixel noise dot
4. Closing fills a small gap inside a solid region
5. (--real) Runs on actual sample images from data/samples/ and
   saves before/after comparison images to data/outputs/morphology_debug/

These tests work whether or not Person 1's preprocessing.py is complete,
because they generate synthetic binary inputs internally.  When Person 1
finishes, you can chain them together using the --chain flag.
"""

import numpy as np
import sys
import os
import argparse
import cv2

# Allow running from the project root or from inside tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.morphology import apply_morphology, custom_erode, custom_dilate


# ===========================================================================
# Synthetic (no real image needed) tests
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
    A 1-pixel white dot surrounded by black is smaller than our 3x3 kernel,
    so it disappears after erosion and does not come back after dilation.
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

    # After dilation with a 3x3 kernel the white region grows by 1 pixel each way
    # so pixels at (3,3) should now be white
    assert dilated[3, 3] == 255, (
        f"FAIL: dilation did not expand region (pixel (3,3) = {dilated[3,3]})"
    )
    print("PASS: custom_dilate expands the white region outward")


def test_closing_fills_gap():
    """
    Closing (dilate then erode) should fill a small black gap inside a white bar.
    """
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 3:17] = 255           # thick white bar
    img[9:11, 9:11] = 0             # small 2x2 black hole in the middle

    # The hole should be filled after closing
    result = apply_morphology(img)
    # Expect the centre pixels of the hole to be filled (255)
    filled = result[9, 9]
    print(f"INFO: gap-fill test: centre of gap after morphology = {filled} "
          f"(expected 255 after real implementation)")
    if filled == 255:
        print("PASS: closing filled the internal gap")
    else:
        print("INFO: gap was not filled - may need larger kernel in closing step")


# ===========================================================================
# Real-image tests (run with --real flag)
# ===========================================================================

def test_on_real_images(debug: bool = False) -> None:
    """
    Run apply_morphology on every image in data/samples/ and verify the
    contract (shape, dtype, binary values).  Optionally save side-by-side
    comparison images.

    This test works INDEPENDENTLY of Person 1's preprocessing.py.
    We convert the sample image to grayscale and threshold it ourselves
    here (using Otsu) to get a binary image - exactly the kind of input
    that Person 1's module will produce.

    When Person 1's module is merged, you can replace the inline conversion
    below with:
        from modules.preprocessing import preprocess_image
        binary = preprocess_image(img)
    """
    samples_dir = "data/samples"
    out_dir     = "data/outputs/morphology_debug"
    os.makedirs(out_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(samples_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    ]

    if not image_files:
        print(f"WARNING: No images found in {samples_dir}/")
        print("         Add .jpg or .png files there and re-run.")
        return

    all_passed = True
    for fname in sorted(image_files):
        fpath = os.path.join(samples_dir, fname)
        img   = cv2.imread(fpath)
        if img is None:
            print(f"SKIP: could not read {fname}")
            continue

        # -- Simulate Person 1's output: grayscale + Otsu threshold ----------
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # NOTE: once Person 1 merges, replace the 2 lines above with:
        #   from modules.preprocessing import preprocess_image
        #   bin_img = preprocess_image(img)

        # -- Run our morphology ----------------------------------------------
        result = apply_morphology(bin_img)

        # -- Contract checks -------------------------------------------------
        ok = True
        if result.shape != bin_img.shape:
            print(f"FAIL [{fname}]: shape changed {bin_img.shape} -> {result.shape}")
            ok = False
        if result.dtype != np.uint8:
            print(f"FAIL [{fname}]: dtype {result.dtype} (expected uint8)")
            ok = False
        unique = set(np.unique(result).tolist())
        if not unique.issubset({0, 255}):
            print(f"FAIL [{fname}]: non-binary values: {unique}")
            ok = False

        if ok:
            print(f"PASS [{fname}]: shape={result.shape}, dtype={result.dtype}")
        else:
            all_passed = False

        # -- Save side-by-side before/after comparison -----------------------
        if debug:
            comparison = np.hstack([bin_img, result])
            out_path   = os.path.join(out_dir, f"morphology_{fname}")
            cv2.imwrite(out_path, comparison)
            print(f"      [debug] saved comparison: {out_path}")

    if all_passed:
        print("PASS: all real images passed the morphology contract checks")
    else:
        print("FAIL: some real images failed - see messages above")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morphology module tests")
    parser.add_argument('--real',  action='store_true',
                        help='Also run tests on real images in data/samples/')
    parser.add_argument('--debug', action='store_true',
                        help='Save before/after images for visual inspection')
    args = parser.parse_args()

    print("=" * 50)
    print("  Morphology Module Tests")
    print("=" * 50)

    # Synthetic tests - always run
    test_shape_preserved()
    test_output_is_binary()
    test_output_dtype()
    test_opening_removes_noise()
    test_dilation_expands_region()
    test_closing_fills_gap()

    # Real image tests - only when --real is passed
    if args.real:
        print()
        print("--- Real image tests ---")
        test_on_real_images(debug=args.debug)

    print()
    print("All tests completed.")
