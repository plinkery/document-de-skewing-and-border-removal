"""
test_preprocessing.py - Person 1 Unit & Integration Tests
==========================================================
Run:
    python tests/test_preprocessing.py                # Run all standard unit tests
    python tests/test_preprocessing.py --real         # Test on real images in data/samples/
    python tests/test_preprocessing.py --real --debug # Also save visual comparison images
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.preprocessing import (
    preprocess_image,
    custom_compute_histogram,
    custom_contrast_stretch,
    custom_otsu_threshold,
    custom_apply_threshold,
)


def test_output_is_2d():
    """Verify that preprocess_image returns a 2-D array (H, W)."""
    fake = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(fake)
    assert result.ndim == 2, f"Expected 2D image, got shape {result.shape}"
    assert result.shape == (480, 640), f"Shape mismatch: expected (480, 640), got {result.shape}"
    print("PASS: output is 2D and matches (H, W)")


def test_output_is_binary():
    """Verify that output contains strictly values in {0, 255}."""
    fake = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(fake)
    unique = set(np.unique(result).tolist())
    assert unique.issubset({0, 255}), f"Expected only values in {{0, 255}}, got {unique}"
    print("PASS: output values are strictly 0 and 255")


def test_output_dtype():
    """Verify that output array has dtype uint8."""
    fake = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result = preprocess_image(fake)
    assert result.dtype == np.uint8, f"Expected dtype uint8, got {result.dtype}"
    print("PASS: output dtype is uint8")


def test_custom_histogram():
    """Verify that custom_compute_histogram correctly tallies all 256 intensity bins."""
    gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    hist = custom_compute_histogram(gray)
    assert hist.shape == (256,), f"Expected 256 bins, got shape {hist.shape}"
    assert int(np.sum(hist)) == gray.size, f"Histogram sum ({np.sum(hist)}) != total pixels ({gray.size})"
    
    # Compare against numpy bincount
    expected_bincount = np.bincount(gray.ravel(), minlength=256)
    assert np.array_equal(hist, expected_bincount), "Custom histogram does not match np.bincount"
    print("PASS: custom histogram accurately counts all 256 bins")


def test_custom_contrast_stretch():
    """Verify that linear contrast stretching maps min to 0 and max to 255."""
    # Synthetic image with compressed dynamic range [50, 150]
    gray = np.random.randint(50, 151, (100, 100), dtype=np.uint8)
    gray[0, 0] = 50
    gray[0, 1] = 150
    stretched = custom_contrast_stretch(gray)
    assert int(np.min(stretched)) == 0, f"Expected min 0 after stretching, got {np.min(stretched)}"
    assert int(np.max(stretched)) == 255, f"Expected max 255 after stretching, got {np.max(stretched)}"
    print("PASS: custom contrast stretch normalizes intensity range to [0, 255]")


def test_custom_otsu_synthetic_bimodal():
    """Verify that custom_otsu_threshold correctly finds the valley between two peaks."""
    # Create clear bimodal image: half pixels at 40 (text), half at 200 (background)
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:50, :] = 40
    gray[50:, :] = 200

    custom_t = custom_otsu_threshold(gray)
    # The optimal threshold must lie strictly between the two peaks
    assert 40 < custom_t <= 200, f"Otsu threshold {custom_t} is outside expected range (40, 200]"

    # Compare against OpenCV's built-in Otsu threshold
    cv_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OpenCV's threshold and custom Otsu should match within 1 intensity level
    diff = abs(custom_t - int(cv_t))
    assert diff <= 1, f"Custom Otsu ({custom_t}) diverged from OpenCV ({cv_t}) by {diff}"
    print(f"PASS: custom Otsu ({custom_t}) matches OpenCV Otsu ({int(cv_t)}) on bimodal test")


def test_edge_case_uniform_image():
    """Verify that uniform / single-color images do not crash with division by zero."""
    flat_black = np.zeros((50, 50, 3), dtype=np.uint8)
    res_black = preprocess_image(flat_black)
    assert res_black.shape == (50, 50)

    flat_white = np.full((50, 50, 3), 255, dtype=np.uint8)
    res_white = preprocess_image(flat_white)
    assert res_white.shape == (50, 50)
    print("PASS: uniform black and white images handled without division errors")


def test_real_samples(debug: bool = False):
    """Test on real sample document images in data/samples/."""
    candidates = [
        "data/samples/sample1.jpg",
        "data/samples/original/38_1.JPG",
        "data/samples/original/38_2.JPG",
        "data/samples/original/39_1.JPG",
    ]
    found = [p for p in candidates if os.path.exists(p)]

    if not found:
        # Search anywhere in data/samples/
        for root, _, files in os.walk("data/samples"):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    found.append(os.path.join(root, f))
                    if len(found) >= 3:
                        break
            if len(found) >= 3:
                break

    if not found:
        print("SKIP: no sample images found in data/samples/")
        return

    out_dir = "data/outputs/preprocessing_debug"
    if debug:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Running real sample tests on {len(found)} image(s)...")
    for img_path in found:
        img = cv2.imread(img_path)
        if img is None:
            continue
        binary = preprocess_image(img)
        assert binary.ndim == 2, f"Failed 2D check on {img_path}"
        assert binary.shape == (img.shape[0], img.shape[1]), f"Shape changed on {img_path}"
        assert set(np.unique(binary).tolist()).issubset({0, 255}), f"Non-binary values on {img_path}"

        if debug:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Save side-by-side or individual debug outputs
            debug_path = os.path.join(out_dir, f"{base_name}_binary.png")
            cv2.imwrite(debug_path, binary)
            print(f"      [debug] saved binarized output to: {debug_path}")

        print(f"PASS: real sample {os.path.basename(img_path)} ({img.shape[1]}x{img.shape[0]}) binarized successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Person 1 Preprocessing Tests")
    parser.add_argument("--real", action="store_true", help="Run tests on real sample images")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images to data/outputs/")
    args = parser.parse_args()

    print("=" * 60)
    print("  Person 1 (Preprocessing) Unit & Integration Tests")
    print("=" * 60)

    test_output_is_2d()
    test_output_is_binary()
    test_output_dtype()
    test_custom_histogram()
    test_custom_contrast_stretch()
    test_custom_otsu_synthetic_bimodal()
    test_edge_case_uniform_image()

    if args.real:
        test_real_samples(debug=args.debug)

    print("=" * 60)
    print("All preprocessing tests passed successfully!")
    print("=" * 60)
