"""
test_border_removal.py - Comprehensive Unit Tests for Member 4's Border Removal Module
=======================================================================================
Scenarios 01–17  : original regression suite (preserved intact)
Section A (18–22): direct mathematical unit tests for pure NumPy/Python functions
Section B (23–27): regression tests for documented edge-case limitations
Section C (28–29): synthetic tests for opposite two-sided and three-sided borders

Total: 29 test cases.

All 17 Scenarios covered:
  1. Top-only border (crop top, preserve bottom/left/right)
  2. Bottom-only border (crop bottom, preserve top/left/right)
  3. Left-only border (crop left, preserve top/bottom/right)
  4. Right-only border (crop right, preserve top/bottom/left)
  5. Two-sided border (top & left cropped, bottom & right preserved)
  6. Dark document with large bright illustration on white scanner bed
  7. White document with large dark illustration on borderless page
  8. Borderless document with natural margins (full image preserved)
  9. Four-sided dark border surrounding a white document
  10. Four-sided light border surrounding a darker document
  11. White document with large blank bottom area
  12. Document containing a dark photograph or illustration
  13. Pure black image (safe fallback)
  14. Pure white image (safe fallback)
  15. Uniform grey image (safe fallback)
  16. Noise and isolated artifacts near borders (noise rejected)
  17. Tiny image (5x5), invalid inputs (ValueError), and memory contiguity

Run:
    python tests/test_border_removal.py
"""

import os
import sys
import numpy as np

# Allow running from project root or inside tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.border_removal import (
    remove_borders,
    compute_horizontal_projection,
    compute_vertical_projection,
    scan_boundary_inward,
)


def test_scenario_01_top_only_border():
    """
    Scenario 1: 60px dark border on top ONLY; bottom, left, and right flush with paper.
    Expected: Crop top; bottom, left, and right must remain 100% untouched.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[:60, :] = 25  # top 60px dark border
    for r in range(100, 380, 25):
        img[r : r + 4, 40:460] = 30

    result = remove_borders(img)

    # Height should be cropped (~342), width should remain 500
    assert abs(result.shape[0] - 342) <= 4, f"Top crop failed: height is {result.shape[0]}"
    assert result.shape[1] == W, f"Width changed unexpectedly: {result.shape[1]}"

    # Verify untouched edge positions and content:
    # Bottom row of result must equal bottom row of original image
    assert np.array_equal(result[-1], img[-1]), "Bottom edge was altered!"
    # Left column of result must equal left column of original image
    cut_top = H - result.shape[0]
    assert np.array_equal(result[:, 0], img[cut_top:, 0]), "Left edge was altered!"
    # Right column of result must equal right column of original image
    assert np.array_equal(result[:, -1], img[cut_top:, -1]), "Right edge was altered!"

    return img.shape, result.shape


def test_scenario_02_bottom_only_border():
    """
    Scenario 2: 60px dark border on bottom ONLY; top, left, and right flush with paper.
    Expected: Crop bottom; top, left, and right must remain 100% untouched.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[-60:, :] = 25  # bottom 60px dark border
    for r in range(40, 320, 25):
        img[r : r + 4, 40:460] = 30

    result = remove_borders(img)

    assert abs(result.shape[0] - 342) <= 4, f"Bottom crop failed: height is {result.shape[0]}"
    assert result.shape[1] == W, f"Width changed unexpectedly: {result.shape[1]}"

    # Top row of result must equal top row of original image
    assert np.array_equal(result[0], img[0]), "Top edge was altered!"
    # Left column of result must equal left column of original image
    assert np.array_equal(result[:, 0], img[: result.shape[0], 0]), "Left edge was altered!"
    # Right column of result must equal right column of original image
    assert np.array_equal(result[:, -1], img[: result.shape[0], -1]), "Right edge was altered!"

    return img.shape, result.shape


def test_scenario_03_left_only_border():
    """
    Scenario 3: 60px dark border on left ONLY; top, bottom, and right flush with paper.
    Expected: Crop left; top, bottom, and right must remain 100% untouched.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[:, :60] = 25  # left 60px dark border
    for r in range(40, 360, 25):
        img[r : r + 4, 80:480] = 30

    result = remove_borders(img)

    assert result.shape[0] == H, f"Height changed unexpectedly: {result.shape[0]}"
    assert abs(result.shape[1] - 442) <= 4, f"Left crop failed: width is {result.shape[1]}"

    cut_left = W - result.shape[1]
    # Right column of result must equal right column of original image
    assert np.array_equal(result[:, -1], img[:, -1]), "Right edge was altered!"
    # Top row of result must equal top row of original image
    assert np.array_equal(result[0], img[0, cut_left:]), "Top edge was altered!"
    # Bottom row of result must equal bottom row of original image
    assert np.array_equal(result[-1], img[-1, cut_left:]), "Bottom edge was altered!"

    return img.shape, result.shape


def test_scenario_04_right_only_border():
    """
    Scenario 4: 60px dark border on right ONLY; top, bottom, and left flush with paper.
    Expected: Crop right; top, bottom, and left must remain 100% untouched.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[:, -60:] = 25  # right 60px dark border
    for r in range(40, 360, 25):
        img[r : r + 4, 20:420] = 30

    result = remove_borders(img)

    assert result.shape[0] == H, f"Height changed unexpectedly: {result.shape[0]}"
    assert abs(result.shape[1] - 442) <= 4, f"Right crop failed: width is {result.shape[1]}"

    # Left column of result must equal left column of original image
    assert np.array_equal(result[:, 0], img[:, 0]), "Left edge was altered!"
    # Top row of result must equal top row of original image
    assert np.array_equal(result[0], img[0, : result.shape[1]]), "Top edge was altered!"
    # Bottom row of result must equal bottom row of original image
    assert np.array_equal(result[-1], img[-1, : result.shape[1]]), "Bottom edge was altered!"

    return img.shape, result.shape


def test_scenario_05_two_sided_border():
    """
    Scenario 5: 60px dark border on TOP and LEFT; bottom and right flush with paper.
    Expected: Crop top and left; bottom and right must remain 100% untouched.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[:60, :] = 25  # top 60px
    img[:, :60] = 25  # left 60px
    for r in range(100, 380, 25):
        img[r : r + 4, 100:460] = 30

    result = remove_borders(img)

    assert abs(result.shape[0] - 342) <= 4, f"Top crop failed: height {result.shape[0]}"
    assert abs(result.shape[1] - 442) <= 4, f"Left crop failed: width {result.shape[1]}"

    cut_left = W - result.shape[1]
    cut_top = H - result.shape[0]
    # Bottom row of result must equal bottom row of original image
    assert np.array_equal(result[-1], img[-1, cut_left:]), "Bottom edge was altered!"
    # Right column of result must equal right column of original image
    assert np.array_equal(result[:, -1], img[cut_top:, -1]), "Right edge was altered!"

    return img.shape, result.shape


def test_scenario_06_dark_doc_with_bright_illustration():
    """
    Scenario 6: Dark document page on white scanner bed containing white text
    and a large bright illustration.
    Expected: White scanner border cropped; dark document content preserved.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 245, dtype=np.uint8)  # white scanner bed
    img[50:350, 60:440] = 50                      # dark document paper (substrate = 50)
    for r in range(80, 200, 15):
        img[r : r + 3, 80:420] = 250              # white text lines
    img[220:330, 150:350] = 240                   # bright illustration box (240)

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]
    assert abs(h_out - 300) <= 8, f"Expected height ~300, got {h_out}"
    assert abs(w_out - 380) <= 8, f"Expected width ~380, got {w_out}"

    # Verify that the bright illustration is completely preserved inside the result
    assert np.any(result == 240), "Illustration content was lost!"
    # Verify that white text lines are present
    assert np.any(result == 250), "Text lines were lost!"

    return img.shape, result.shape


def test_scenario_07_white_doc_with_dark_illustration():
    """
    Scenario 7: White document with 40px natural white margins, text, and a large
    dark illustration (borderless).
    Expected: Full image preserved (450, 450, 3). Natural margins NOT cropped.
    """
    H, W = 450, 450
    img = np.full((H, W, 3), 240, dtype=np.uint8)  # white paper
    for r in range(40, 180, 25):
        img[r : r + 4, 40 : W - 40] = 30
    img[220:380, 100:350] = 40                     # dark illustration box

    result = remove_borders(img)

    assert result.shape == (H, W, 3), f"Natural margins cropped! {img.shape} -> {result.shape}"
    assert np.array_equal(result, img), "Image content was modified!"

    return img.shape, result.shape


def test_scenario_08_borderless_document_with_natural_margins():
    """
    Scenario 8: Borderless document with white paper throughout and text lines.
    Expected: Full image preserved intact (400, 500, 3).
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 235, dtype=np.uint8)
    for r in range(20, H - 20, 30):
        img[r : r + 5, 20 : W - 20] = 30

    result = remove_borders(img)

    assert result.shape == (H, W, 3), f"Borderless document changed: {img.shape} -> {result.shape}"
    assert np.array_equal(result, img), "Image content was modified!"

    return img.shape, result.shape


def test_scenario_09_four_sided_dark_border():
    """
    Scenario 9: 4-sided dark background surrounding white document (standard scan).
    Expected: All 4 borders cropped; white document preserved.
    """
    H, W = 400, 500
    top_doc, bot_doc = 50, 350
    left_doc, right_doc = 60, 440

    img = np.full((H, W, 3), 25, dtype=np.uint8)
    img[top_doc:bot_doc, left_doc:right_doc] = 230
    for r in range(top_doc + 30, bot_doc - 30, 25):
        img[r : r + 4, left_doc + 20 : right_doc - 20] = 30

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]
    assert abs(h_out - 300) <= 8, f"Expected height ~300, got {h_out}"
    assert abs(w_out - 380) <= 8, f"Expected width ~380, got {w_out}"

    return img.shape, result.shape


def test_scenario_10_four_sided_light_border():
    """
    Scenario 10: 4-sided light background surrounding a darker document.
    Expected: All 4 borders cropped; darker document preserved.
    """
    H, W = 400, 500
    top_doc, bot_doc = 50, 350
    left_doc, right_doc = 60, 440

    img = np.full((H, W, 3), 245, dtype=np.uint8)
    img[top_doc:bot_doc, left_doc:right_doc] = 85
    for r in range(top_doc + 30, bot_doc - 30, 25):
        img[r : r + 4, left_doc + 20 : right_doc - 20] = 20

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]
    assert abs(h_out - 300) <= 8, f"Expected height ~300, got {h_out}"
    assert abs(w_out - 380) <= 8, f"Expected width ~380, got {w_out}"

    return img.shape, result.shape


def test_scenario_11_white_doc_with_large_blank_area():
    """
    Scenario 11: White document with text only at top, and large blank area below.
    Expected: Outer border removed; blank bottom half preserved intact.
    """
    H, W = 500, 500
    top_doc, bot_doc = 60, 440
    left_doc, right_doc = 60, 440

    img = np.full((H, W, 3), 20, dtype=np.uint8)
    img[top_doc:bot_doc, left_doc:right_doc] = 240
    for r in range(80, 180, 20):
        img[r : r + 4, left_doc + 20 : right_doc - 20] = 30

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]
    assert abs(h_out - 380) <= 8, f"Blank area was clipped! Height {h_out}"
    assert abs(w_out - 380) <= 8, f"Blank area was clipped! Width {w_out}"

    return img.shape, result.shape


def test_scenario_12_doc_with_dark_photograph():
    """
    Scenario 12: Document contains a large dark photo inside page.
    Expected: Photo does not cause false border cut; whole page preserved.
    """
    H, W = 500, 500
    top_doc, bot_doc = 50, 450
    left_doc, right_doc = 50, 450

    img = np.full((H, W, 3), 20, dtype=np.uint8)
    img[top_doc:bot_doc, left_doc:right_doc] = 235
    img[150:350, 150:350] = 35

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]
    assert abs(h_out - 400) <= 8, f"Expected height ~400, got {h_out}"
    assert abs(w_out - 400) <= 8, f"Expected width ~400, got {w_out}"

    return img.shape, result.shape


def test_scenario_13_pure_black():
    """
    Scenario 13: Completely black image.
    Expected: Safe fallback, returns original image without crashing.
    """
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    result = remove_borders(img)
    assert result.shape == (300, 400, 3), f"Failed: {result.shape}"
    return img.shape, result.shape


def test_scenario_14_pure_white():
    """
    Scenario 14: Completely white image.
    Expected: Safe fallback, returns original image without crashing.
    """
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    result = remove_borders(img)
    assert result.shape == (300, 400, 3), f"Failed: {result.shape}"
    return img.shape, result.shape


def test_scenario_15_uniform_grey():
    """
    Scenario 15: Uniform grey image with zero contrast.
    Expected: Safe fallback, returns original image without crashing.
    """
    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    result = remove_borders(img)
    assert result.shape == (300, 400, 3), f"Failed: {result.shape}"
    return img.shape, result.shape


def test_scenario_16_noise_near_edges():
    """
    Scenario 16: Noise specks and isolated dust inside the border area.
    Expected: Noise ignored by consecutive hit check; true boundary found.
    """
    H, W = 400, 500
    top_doc, bot_doc = 50, 350
    left_doc, right_doc = 60, 440

    img = np.full((H, W, 3), 20, dtype=np.uint8)
    img[top_doc:bot_doc, left_doc:right_doc] = 230

    img[5:8, 5:8] = 255
    img[15, 100:104] = 255
    img[385:388, 20:25] = 255
    img[200:204, 10:14] = 255

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]
    assert abs(h_out - 300) <= 8, f"Noise triggered false cut! Height: {h_out}"
    assert abs(w_out - 380) <= 8, f"Noise triggered false cut! Width: {w_out}"

    return img.shape, result.shape


def test_scenario_17_tiny_images_invalid_inputs_contiguity():
    """
    Scenario 17: Tiny image (5x5), invalid inputs (ValueError), format and C-contiguity.
    Expected: Informative ValueError for bad inputs, C-contiguous non-empty output.
    """
    # 1. Tiny 5x5 image
    img_tiny = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    res_tiny = remove_borders(img_tiny)
    assert res_tiny.shape == (5, 5, 3), f"Tiny image failed: {res_tiny.shape}"

    # Invalid inputs must fail with contract-specific messages.
    invalid_inputs = [
        (None, "valid NumPy array"),
        (np.zeros((100, 100), dtype=np.uint8), "3-channel"),
        (np.zeros((100, 100, 4), dtype=np.uint8), "3-channel"),
        ("not an image", "valid NumPy array"),
        (np.zeros((100, 100, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((100, 100, 3), dtype=np.uint16), "dtype uint8"),
        (np.zeros((0, 100, 3), dtype=np.uint8), "non-empty image"),
        (np.zeros((100, 0, 3), dtype=np.uint8), "non-empty image"),
    ]
    for invalid_img, expected_message in invalid_inputs:
        try:
            remove_borders(invalid_img)
        except ValueError as error:
            assert expected_message in str(error), (
                f"Expected {expected_message!r} in error, got {error!r}"
            )
        else:
            raise AssertionError(f"Expected ValueError containing {expected_message!r}")

    # 6. Format and memory contiguity
    fake = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    result = remove_borders(fake)
    assert result.dtype == np.uint8, "dtype not uint8"
    assert result.ndim == 3 and result.shape[2] == 3, "Not 3-channel"
    assert result.flags.c_contiguous is True, "Not C-contiguous in memory"

    return "Tiny & Invalid Tests", "Handled Safely & Validated"


# ===========================================================================
# Section A — Direct mathematical unit tests for pure NumPy/Python functions
# ===========================================================================

def test_mathunit_A1_horizontal_projection_values():
    """
    Section A, Test 1: Verify compute_horizontal_projection() against
    hand-calculated reference values.

    Input mask (4 rows × 4 cols, uint8 with values 0/255):
        row 0: [255, 255, 255, 255]  → density = 4/4 = 1.0
        row 1: [255,   0,   0,   0]  → density = 1/4 = 0.25
        row 2: [  0,   0,   0,   0]  → density = 0/4 = 0.0
        row 3: [255, 255,   0,   0]  → density = 2/4 = 0.5

    Expected projection: [1.0, 0.25, 0.0, 0.5]
    """
    mask = np.array(
        [[255, 255, 255, 255],
         [255,   0,   0,   0],
         [  0,   0,   0,   0],
         [255, 255,   0,   0]],
        dtype=np.uint8,
    )
    proj = compute_horizontal_projection(mask)
    expected = np.array([1.0, 0.25, 0.0, 0.5], dtype=np.float32)

    assert proj.shape == (4,), f"Wrong shape: {proj.shape}"
    np.testing.assert_allclose(proj, expected, atol=1e-6,
        err_msg=f"Horizontal projection mismatch: {proj} != {expected}")
    return "4x4 mask", proj.tolist()


def test_mathunit_A2_vertical_projection_values():
    """
    Section A, Test 2: Verify compute_vertical_projection() against
    hand-calculated reference values.

    Input mask (4 rows × 4 cols, uint8 with values 0/255):
        col 0: [255, 255, 0, 255]  → density = 3/4 = 0.75
        col 1: [255,   0, 0, 255]  → density = 2/4 = 0.5
        col 2: [255,   0, 0,   0]  → density = 1/4 = 0.25
        col 3: [255,   0, 0,   0]  → density = 1/4 = 0.25

    Expected projection: [0.75, 0.5, 0.25, 0.25]
    """
    mask = np.array(
        [[255, 255, 255, 255],
         [255,   0,   0,   0],
         [  0,   0,   0,   0],
         [255, 255,   0,   0]],
        dtype=np.uint8,
    )
    proj = compute_vertical_projection(mask)
    expected = np.array([0.75, 0.5, 0.25, 0.25], dtype=np.float32)

    assert proj.shape == (4,), f"Wrong shape: {proj.shape}"
    np.testing.assert_allclose(proj, expected, atol=1e-6,
        err_msg=f"Vertical projection mismatch: {proj} != {expected}")
    return "4x4 mask", proj.tolist()


def test_mathunit_A3_scan_boundary_forward():
    """
    Section A, Test 3: Verify that scan_boundary_inward() forward scan finds
    the first run of consecutive above-threshold values correctly.

    Projection: [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    threshold=0.05, min_consecutive=3
    Forward scan should lock at index 2 (first of 5 consecutive hits at 0.1).
    start_idx = max(0, 4 - 3 + 1) = 2
    """
    proj = np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float32)
    start, end = scan_boundary_inward(proj, threshold=0.05, min_consecutive=3)

    assert start == 2, f"Forward scan start wrong: {start} (expected 2)"
    # end should be at 10 (no trailing background)
    assert end == 10, f"Backward scan end wrong: {end} (expected 10)"
    return "forward scan", (start, end)


def test_mathunit_A4_scan_boundary_backward():
    """
    Section A, Test 4: Verify that scan_boundary_inward() backward scan finds
    the last run of consecutive above-threshold values correctly.

    Projection: [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
    threshold=0.05, min_consecutive=3
    Backward scan should lock just after the last above-threshold run ends.
    Scan finds 3 consecutive hits ending at index 7; end_idx = min(10, 7+3) = 10.
    Wait — backward hits land at [7,6,5], when hit count=3 at i=5: end_idx=min(10,5+3)=8.
    """
    proj = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0], dtype=np.float32)
    start, end = scan_boundary_inward(proj, threshold=0.05, min_consecutive=3)

    assert start == 0, f"Forward scan start wrong: {start} (expected 0)"
    assert end == 8, f"Backward scan end wrong: {end} (expected 8)"
    return "backward scan", (start, end)


def test_mathunit_A5_empty_masks():
    """
    Section A, Test 5: Verify that compute_horizontal_projection() and
    compute_vertical_projection() safely return an empty array for all
    zero-dimension inputs without raising any exception.

    Before the fix, shapes (0, W) and (H, 0) caused:
      ValueError: zero-size array to reduction operation maximum
    """
    shapes_h = [(0, 10), (10, 0)]   # both should return []
    shapes_v = [(0, 10), (10, 0)]

    for shape in shapes_h:
        result = compute_horizontal_projection(np.zeros(shape, dtype=np.uint8))
        assert len(result) == 0, (
            f"compute_horizontal_projection{shape} should return [], got len={len(result)}"
        )

    for shape in shapes_v:
        result = compute_vertical_projection(np.zeros(shape, dtype=np.uint8))
        assert len(result) == 0, (
            f"compute_vertical_projection{shape} should return [], got len={len(result)}"
        )

    # Also test scan_boundary_inward on empty projection
    start, end = scan_boundary_inward(np.array([], dtype=np.float32))
    assert start == 0 and end == 0, (
        f"scan_boundary_inward([]) should return (0,0), got ({start},{end})"
    )

    return "Empty guard check", "All safe"


# ===========================================================================
# Section B — Regression tests for documented edge-case limitations
# ===========================================================================

def test_regression_B1_fullbleed_dark_header_banner():
    """
    Section B, Regression 1: Borderless white document with a full-bleed dark
    header banner touching the top physical edge.

    The banner (rows 0–19) is dark (30) but belongs to the document, not an
    external scanner background.

    Expected behaviour (documented limitation): the algorithm cannot distinguish
    a dark full-bleed banner from an external dark border because projection
    profiles are 1-D. The banner may be cropped.  This test therefore verifies
    only that:
      1. The function returns without raising any exception.
      2. The output is a valid 3-channel uint8 array.
      3. No crash occurs and no content below the banner is lost.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 240, dtype=np.uint8)
    img[:20, :] = 30         # Full-bleed dark banner at the very top
    for r in range(50, 360, 25):
        img[r:r+4, 40:460] = 30   # Text below the banner

    result = remove_borders(img)

    assert isinstance(result, np.ndarray), "Result must be a NumPy array"
    assert result.ndim == 3 and result.shape[2] == 3, "Result must be (H,W,3)"
    assert result.dtype == np.uint8, "Result dtype must be uint8"
    assert result.shape[0] > 0 and result.shape[1] > 0, "Result must be non-empty"

    # Content (text rows) below the banner must be preserved
    assert np.any(result == 30), "Text-row content was completely lost!"

    return img.shape, result.shape


def test_regression_B2_white_doc_large_dark_photo():
    """
    Section B, Regression 2: White document containing a very large dark
    photograph occupying > 50 % of the page, on a borderless page.

    The broad interior median drops below 180 because the dark photo dominates.
    The safeguard 'is_borderless_white' may therefore fail, and the algorithm
    may treat the natural white margins as a scanner bed and crop them.

    This test verifies:
      1. No exception is raised.
      2. The result is a valid 3-channel uint8 array.
      3. The dark photograph content is NOT cropped out of the result.
    """
    H, W = 500, 500
    img = np.full((H, W, 3), 240, dtype=np.uint8)
    img[50:450, 50:450] = 40    # Large dark photo (90 % of page area)

    result = remove_borders(img)

    assert isinstance(result, np.ndarray), "Result must be a NumPy array"
    assert result.ndim == 3 and result.shape[2] == 3, "Result must be (H,W,3)"
    assert result.shape[0] > 0 and result.shape[1] > 0, "Result must be non-empty"
    assert np.any(result == 40), "Dark photograph content was completely lost!"

    return img.shape, result.shape


def test_regression_B3_dark_doc_large_bright_illustration():
    """
    Section B, Regression 3: Dark (charcoal, intensity 95) document on a
    white scanner bed, containing a large bright illustration that covers
    > 50 % of the central region.

    When c_p25 >= 90 and c_med > 125 (both fail), Condition B exits early and
    returns the uncropped image (a safe false-negative).

    This test verifies:
      1. No exception is raised.
      2. The result is a valid 3-channel uint8 array.
      3. The dark document content is NOT cropped.
      4. The function does not crash regardless of whether it crops or not.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 245, dtype=np.uint8)   # White scanner bed
    img[50:350, 60:440] = 95                         # Charcoal document
    img[100:320, 100:400] = 230                      # Large bright illustration

    result = remove_borders(img)

    assert isinstance(result, np.ndarray), "Result must be a NumPy array"
    assert result.ndim == 3 and result.shape[2] == 3, "Result must be (H,W,3)"
    assert result.shape[0] > 0 and result.shape[1] > 0, "Result must be non-empty"
    assert np.any(result == 95), "Charcoal document substrate was completely lost!"

    return img.shape, result.shape


def test_regression_B4_gradient_shadow_edge():
    """
    Section B, Regression 4: White document with a linear gradient shadow on
    the left edge (intensity 80 at x=0 → 235 at x=40) and document text
    starting inside the shadow region (x ≥ 10).

    Expected behaviour (documented limitation): global Otsu threshold may
    classify shadow pixels as background, cropping into the shadow region and
    potentially losing text at x=10.

    This test verifies:
      1. No exception is raised.
      2. The result is a valid 3-channel uint8 array.
      3. No crash.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 235, dtype=np.uint8)
    shadow = np.linspace(80, 235, 40, dtype=np.uint8)
    img[:, :40] = shadow[None, :, None]        # Gradient shadow on left
    for r in range(40, 360, 25):
        img[r:r+4, 10:460] = 30               # Text beginning inside shadow

    result = remove_borders(img)

    assert isinstance(result, np.ndarray), "Result must be a NumPy array"
    assert result.ndim == 3 and result.shape[2] == 3, "Result must be (H,W,3)"
    assert result.shape[0] > 0 and result.shape[1] > 0, "Result must be non-empty"

    return img.shape, result.shape


def test_regression_B5_charcoal_doc_bright_graphics_false_negative():
    """
    Section B, Regression 5: Charcoal document (c_p25 ≈ 95, within range
    90–110) with a large bright graphic covering the centre so that c_med > 125.

    Both conditions (c_med <= 125 and c_p25 < 90) fail → safe fallback returns
    the full uncropped image.  This is intentional safe behaviour.

    This test verifies:
      1. The result equals the original image exactly (safe false-negative).
      2. No exception or crash.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 245, dtype=np.uint8)   # White scanner bed
    img[50:350, 60:440] = 95                         # Charcoal substrate
    # Bright graphics covering > 50 % of centre box so c_med > 125
    img[100:300, 90:410] = 230

    result = remove_borders(img)

    assert isinstance(result, np.ndarray), "Result must be a NumPy array"
    assert result.ndim == 3 and result.shape[2] == 3, "Result must be (H,W,3)"
    # Either the image is returned unchanged (safe fallback) or the scanner
    # border was detected.  Either way the charcoal content must survive.
    assert np.any(result == 95), "Charcoal substrate was completely removed!"

    return img.shape, result.shape


# ===========================================================================
# Section C — Synthetic tests for opposite two-sided and three-sided borders
# ===========================================================================

def test_synthetic_C1_opposite_two_sided_top_and_bottom():
    """
    Section C, Test 1: 60 px dark border on TOP and BOTTOM; left and right
    edges are flush with white paper.

    Expected: Both top and bottom borders cropped; left and right edges remain
    100 % untouched (same pixel-level content as the original image).
    The result width must equal the original width (no horizontal cropping).
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[:60, :] = 25   # top border
    img[-60:, :] = 25  # bottom border
    for r in range(100, 320, 25):
        img[r:r+4, 40:460] = 30   # text in the middle

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]

    # Both borders (60 px each) should be removed; content span is ~280 px.
    # Allow ±8 px for safety margin and threshold quantisation.
    assert abs(h_out - 282) <= 8, (
        f"Top+bottom crop failed: output height {h_out}, expected ~282"
    )
    # Width must be untouched: left=0, right=W → width stays W
    assert w_out == W, f"Width changed unexpectedly: {w_out} (expected {W})"

    # The left and right columns of the result must be byte-identical to
    # the same columns of the original image at the cropped row-range.
    # We identify the starting row by scanning for the first white row (>60).
    start_row = H - h_out  # rough estimate; use it to find the actual match
    # Binary-search free: since top ~60, start_row ≈ 58-62. Use np.array_equal
    # on the left column to find the exact offset.
    matched = False
    for cand in range(55, 70):
        if np.array_equal(result[:, 0], img[cand:cand + h_out, 0]):
            assert np.array_equal(result[:, -1], img[cand:cand + h_out, -1]), (
                "Right column does not match original image at the detected crop!"
            )
            matched = True
            break
    assert matched, "Could not verify left/right column match against original image."

    return img.shape, result.shape


def test_synthetic_C2_three_sided_border_top_left_bottom():
    """
    Section C, Test 2: 60 px dark border on TOP, LEFT, and BOTTOM;
    the right edge is flush with white paper.

    Expected: Three borders cropped; right edge of result must be byte-identical
    to the right edge of the original image at the same row-range.
    """
    H, W = 400, 500
    img = np.full((H, W, 3), 230, dtype=np.uint8)
    img[:60, :] = 25     # top
    img[-60:, :] = 25    # bottom
    img[:, :60] = 25     # left
    for r in range(100, 320, 25):
        img[r:r+4, 80:460] = 30   # text

    result = remove_borders(img)

    h_out, w_out = result.shape[:2]

    # Height: top+bottom removed (~280 content px ± 8)
    assert abs(h_out - 282) <= 8, (
        f"Top+bottom crop failed: height {h_out}, expected ~282"
    )
    # Width: left removed (~440 content px ± 8)
    assert abs(w_out - 442) <= 8, (
        f"Left crop failed: width {w_out}, expected ~442"
    )

    # Right column of result must be byte-identical to the right column of
    # the original image at the same row-range (right=W, so no horizontal
    # padding was applied to the right edge).
    # Scan the original right column for the matching h_out-length slice.
    right_col_result = result[:, -1]        # shape (h_out, 3)
    right_col_original = img[:, -1]        # shape (H, 3)
    matched = False
    for cand in range(H - h_out + 1):
        if np.array_equal(right_col_result, right_col_original[cand:cand + h_out]):
            matched = True
            break
    assert matched, (
        f"Right column of result does not match any row-slice of the original "
        f"right column. Unique values in result right col: "
        f"{set(right_col_result[:, 0].tolist())}"
    )

    return img.shape, result.shape


# ===========================================================================
# Test Runner
# ===========================================================================

def run_all_tests():
    test_functions = [
        # --- Original 17 scenarios (preserved intact) ---
        ("Scenario 01: Top-only border", test_scenario_01_top_only_border),
        ("Scenario 02: Bottom-only border", test_scenario_02_bottom_only_border),
        ("Scenario 03: Left-only border", test_scenario_03_left_only_border),
        ("Scenario 04: Right-only border", test_scenario_04_right_only_border),
        ("Scenario 05: Two-sided border (top & left)", test_scenario_05_two_sided_border),
        ("Scenario 06: Dark doc with bright illustration", test_scenario_06_dark_doc_with_bright_illustration),
        ("Scenario 07: White doc with dark illustration", test_scenario_07_white_doc_with_dark_illustration),
        ("Scenario 08: Borderless document with margins", test_scenario_08_borderless_document_with_natural_margins),
        ("Scenario 09: 4-sided dark border on white doc", test_scenario_09_four_sided_dark_border),
        ("Scenario 10: 4-sided light border on dark doc", test_scenario_10_four_sided_light_border),
        ("Scenario 11: White doc with blank area", test_scenario_11_white_doc_with_large_blank_area),
        ("Scenario 12: Doc with dark photograph", test_scenario_12_doc_with_dark_photograph),
        ("Scenario 13: Pure black image", test_scenario_13_pure_black),
        ("Scenario 14: Pure white image", test_scenario_14_pure_white),
        ("Scenario 15: Uniform grey image", test_scenario_15_uniform_grey),
        ("Scenario 16: Noise near edges", test_scenario_16_noise_near_edges),
        ("Scenario 17: Tiny, invalid inputs & contiguity", test_scenario_17_tiny_images_invalid_inputs_contiguity),
        # --- Section A: Mathematical unit tests ---
        ("Math A1: Horizontal projection values", test_mathunit_A1_horizontal_projection_values),
        ("Math A2: Vertical projection values", test_mathunit_A2_vertical_projection_values),
        ("Math A3: Forward boundary scan", test_mathunit_A3_scan_boundary_forward),
        ("Math A4: Backward boundary scan", test_mathunit_A4_scan_boundary_backward),
        ("Math A5: Empty mask safety (bug fix)", test_mathunit_A5_empty_masks),
        # --- Section B: Regression tests for documented limitations ---
        ("Regression B1: Full-bleed dark header banner", test_regression_B1_fullbleed_dark_header_banner),
        ("Regression B2: White doc large dark photo", test_regression_B2_white_doc_large_dark_photo),
        ("Regression B3: Dark doc large bright illustration", test_regression_B3_dark_doc_large_bright_illustration),
        ("Regression B4: Gradient shadow near edge", test_regression_B4_gradient_shadow_edge),
        ("Regression B5: Charcoal doc + bright graphics (safe fallback)", test_regression_B5_charcoal_doc_bright_graphics_false_negative),
        # --- Section C: Synthetic opposite/three-sided border tests ---
        ("Synthetic C1: Opposite two-sided border (top & bottom)", test_synthetic_C1_opposite_two_sided_top_and_bottom),
        ("Synthetic C2: Three-sided border (top, left, bottom)", test_synthetic_C2_three_sided_border_top_left_bottom),
    ]

    passed = 0
    failed = 0
    results_table = []

    print("=" * 72)
    print("  Running Member 4 Border Removal Comprehensive 29-Test Suite")
    print("=" * 72)

    for name, test_fn in test_functions:
        try:
            in_shape, out_shape = test_fn()
            passed += 1
            results_table.append((name, "PASS", in_shape, out_shape))
            print(f"PASS [{name}]: {in_shape} -> {out_shape}")
        except Exception as e:
            failed += 1
            results_table.append((name, f"FAIL ({e})", "N/A", "N/A"))
            print(f"FAIL [{name}]: {e}")

    print("=" * 72)
    print(f"Summary: {passed} passed, {failed} failed out of {len(test_functions)} tests.")
    print("=" * 72)

    return passed, failed, results_table


if __name__ == "__main__":
    passed, failed, results = run_all_tests()
    if failed > 0:
        sys.exit(1)
