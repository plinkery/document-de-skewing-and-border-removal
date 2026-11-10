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

Implemented topics (Person 4's own work for 50/50 balance):
    1. Grayscale conversion and polarity-aware thresholding (cv2)
    2. compute_horizontal_projection() - mathematical row projection (pure NumPy)
    3. compute_vertical_projection()   - mathematical column projection (pure NumPy)
    4. scan_boundary_inward()          - custom inward scanning with noise filtering (pure Python/NumPy)
    5. detect_document_boundaries()    - independent 4-edge margin analysis, dual-statistic substrate
                                         verification (c_p75, c_p25, c_med), safety margin padding,
                                         and safe fallbacks
    6. remove_borders()                - safe slicing of the original BGR image
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Module-level named constants (replaces inline magic numbers)
# ---------------------------------------------------------------------------

# Minimum pixel contrast (max - min) below which the whole image is considered
# uniform and no border detection is attempted.
MIN_CONTRAST = 15

# Fraction of image height/width sampled as the outer edge strip (2%).
EDGE_STRIP_FRACTION = 0.02

# Interior sampling box: central 20%–80% region used to estimate document substrate.
INTERIOR_BOX_FRACTION_INNER = 0.20
INTERIOR_BOX_FRACTION_OUTER = 0.80

# Broad interior box: 10%–90% region used for overall page background verification.
BROAD_BOX_FRACTION_INNER = 0.10
BROAD_BOX_FRACTION_OUTER = 0.90

# Light document substrate: 75th percentile of central pixels must exceed this
# value to classify the document as a light (white-paper) document.
LIGHT_DOC_P75_MIN = 140

# Dark document substrate: 25th percentile of central pixels must be below this
# value to classify the document substrate as dark.
DARK_DOC_P25_MAX = 110

# Minimum intensity difference between an outer edge median and the document
# substrate percentile before a genuine external border is confirmed.
BORDER_TRANSITION_DELTA = 30.0

# Broad interior median threshold below which the page is NOT considered a
# borderless white document (used to protect natural white margins from being
# cropped when the page contains large dark illustrations).
WHITE_PAGE_BROAD_MED_MIN = 180

# Maximum allowed absolute difference between any edge median and the broad
# interior median for the borderless-white-document safeguard to trigger.
WHITE_PAGE_EDGE_TOLERANCE = 20

# Dark document core: median of central pixels must be at or below this value
# to confirm a genuine dark-substrate document.
DARK_DOC_MEDIAN_MAX = 125

# Deep dark substrate: 25th percentile below this value unconditionally
# confirms a dark document regardless of the central median.
DARK_DOC_P25_DEEP = 90

# Minimum projection density (fraction of row/column occupied by document
# pixels) the strongest row/column must achieve for detection to proceed.
CONFIDENCE_MIN_PROJ_DENSITY = 0.20

# Fraction of the peak row/column projection density used as the adaptive
# scanning threshold (clamped to a minimum floor of ADAPTIVE_PROJ_FLOOR).
ADAPTIVE_PROJ_SCALE = 0.08

# Floor value for the adaptive projection threshold so sparse-content rows
# still register (e.g. a document with only one line of text per page).
ADAPTIVE_PROJ_FLOOR = 0.04

# Fraction of image height/width added as safety margin after cropping to
# prevent cutting content at the border edge (0.5 %).
SAFETY_MARGIN_PCT = 0.005

# Fraction of image height/width used to derive the minimum consecutive-hit
# count for the inward scan (clamped to a minimum of MIN_CONSECUTIVE_HITS).
CONSECUTIVE_FRACTION = 0.002

# Absolute minimum number of consecutive rows/columns above the projection
# threshold before a document boundary is confirmed.
MIN_CONSECUTIVE_HITS = 3

# Minimum size (pixels) of either image dimension; images smaller than this
# in either dimension are returned unchanged without attempting detection.
TINY_IMAGE_MIN_DIM = 10

# Any crop that leaves the document occupying less than this fraction of the
# original dimension is considered unreliable and the original is returned.
MIN_CROP_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Custom mathematical projection functions (pure NumPy - 50/50 proof)
# ---------------------------------------------------------------------------

def compute_horizontal_projection(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the horizontal projection profile of a 2D binary mask.

    Mathematically, for an image of height H and width W:
        P_h(y) = (1 / W) * sum_{x=0}^{W-1} M(y, x)
    where M(y, x) is 1 for document pixels and 0 for border/background.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary array of shape (H, W), dtype float or bool/uint8.
        Values: 1.0 (or 255) for document content, 0 for border.

    Returns
    -------
    np.ndarray
        1D array of length H, representing the fraction of document
        pixels along each row (values in [0.0, 1.0]).
    """
    # Guard: must be a 2D array with at least one element in every dimension.
    # shape[1] == 0 was the original check; size == 0 also catches shape (0, W)
    # which previously caused np.max() to raise ValueError on empty arrays.
    if binary_mask.ndim != 2 or binary_mask.size == 0 or binary_mask.shape[1] == 0:
        return np.array([], dtype=np.float32)

    # Normalize mask to {0.0, 1.0} if it contains 255
    if binary_mask.dtype == np.uint8 and np.max(binary_mask) > 1:
        mask = (binary_mask > 0).astype(np.float32)
    else:
        mask = binary_mask.astype(np.float32)

    # Row-wise sum divided by width gives the row density
    row_sums = np.sum(mask, axis=1)
    width = mask.shape[1]
    return row_sums / float(width)


def compute_vertical_projection(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the vertical projection profile of a 2D binary mask.

    Mathematically, for an image of height H and width W:
        P_v(x) = (1 / H) * sum_{y=0}^{H-1} M(y, x)
    where M(y, x) is 1 for document pixels and 0 for border/background.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary array of shape (H, W), dtype float or bool/uint8.
        Values: 1.0 (or 255) for document content, 0 for border.

    Returns
    -------
    np.ndarray
        1D array of length W, representing the fraction of document
        pixels along each column (values in [0.0, 1.0]).
    """
    # Guard: must be a 2D array with at least one element in every dimension.
    # shape[0] == 0 was the original check; size == 0 also catches shape (H, 0)
    # which previously caused np.max() to raise ValueError on empty arrays.
    if binary_mask.ndim != 2 or binary_mask.size == 0 or binary_mask.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # Normalize mask to {0.0, 1.0} if it contains 255
    if binary_mask.dtype == np.uint8 and np.max(binary_mask) > 1:
        mask = (binary_mask > 0).astype(np.float32)
    else:
        mask = binary_mask.astype(np.float32)

    # Column-wise sum divided by height gives the column density
    col_sums = np.sum(mask, axis=0)
    height = mask.shape[0]
    return col_sums / float(height)


# ---------------------------------------------------------------------------
# Custom inward scanning algorithm (pure Python/NumPy - 50/50 proof)
# ---------------------------------------------------------------------------

def scan_boundary_inward(
    projection: np.ndarray,
    threshold: float = 0.05,
    min_consecutive: int = 5
) -> tuple[int, int]:
    """
    Scan a 1D projection profile inward from both ends to find the
    start and end boundaries of the document.

    To eliminate isolated noise lines (e.g. scanner dust or punch holes),
    the density must meet or exceed `threshold` for at least `min_consecutive`
    consecutive indices before a boundary is accepted.

    Parameters
    ----------
    projection : np.ndarray
        1D array representing density profile along an axis.
    threshold : float
        Minimum density ratio (default 0.05, i.e., 5% of row/col must be document).
    min_consecutive : int
        Number of consecutive points required to confirm document boundary.

    Returns
    -------
    tuple[int, int]
        (start_index, end_index) where end_index is exclusive.
    """
    length = len(projection)
    if length == 0:
        return 0, 0

    # Forward scan from beginning (top or left)
    start_idx = 0
    consecutive_hits = 0
    for i in range(length):
        if projection[i] >= threshold:
            consecutive_hits += 1
            if consecutive_hits >= min_consecutive:
                start_idx = max(0, i - min_consecutive + 1)
                break
        else:
            consecutive_hits = 0

    # Backward scan from end (bottom or right)
    end_idx = length
    consecutive_hits = 0
    for i in range(length - 1, -1, -1):
        if projection[i] >= threshold:
            consecutive_hits += 1
            if consecutive_hits >= min_consecutive:
                end_idx = min(length, i + min_consecutive)
                break
        else:
            consecutive_hits = 0

    return start_idx, end_idx


# ---------------------------------------------------------------------------
# Full boundary detection with independent edge analysis & substrate checks
# ---------------------------------------------------------------------------

def detect_document_boundaries(
    gray: np.ndarray,
    safety_margin_pct: float = 0.005,
    transition_delta: float = 30.0
) -> tuple[int, int, int, int]:
    """
    Detect the [top, bottom, left, right] boundaries of the document in a
    grayscale image.

    Key improvements:
      1. Independent per-edge analysis: evaluates top, bottom, left, and right
         margins separately. Prevents single-sided or two-sided borders from
         being diluted or masked by borderless edges.
      2. Dual-statistic substrate estimation:
         - c_p75 represents the paper substrate for light documents (immune to text).
         - c_p25 represents the paper substrate for dark documents.
         - c_med represents the central core median.
      3. White document with illustration safeguard:
         Verifies whether bright edges belong to an external scanner bed or
         are simply the natural white margins of a borderless page.
      4. Independent projection scanning:
         Edges with a border advance inward to the document boundary;
         flush edges without a border stop immediately at index 0 or H/W.
      5. Safety padding and comprehensive fallback protections.

    Parameters
    ----------
    gray : np.ndarray
        2D grayscale image.
    safety_margin_pct : float
        Safety padding percentage added to each side (default 0.5%).
    transition_delta : float
        Minimum contrast difference required between external border and
        document substrate to confirm that a genuine external border exists.

    Returns
    -------
    tuple[int, int, int, int]
        (top, bottom, left, right) slice coordinates.
    """
    H, W = gray.shape

    # Safety check: tiny image
    if H < TINY_IMAGE_MIN_DIM or W < TINY_IMAGE_MIN_DIM:
        return 0, H, 0, W

    # Step 1: Contrast check (uniform black, white, or grey images)
    min_val, max_val = int(np.min(gray)), int(np.max(gray))
    if max_val - min_val < MIN_CONTRAST:
        # Uniform image with no discernible borders
        return 0, H, 0, W

    # Step 2: Sample 4 outer edges independently
    edge_h = max(2, int(H * EDGE_STRIP_FRACTION))
    edge_w = max(2, int(W * EDGE_STRIP_FRACTION))

    m_top = float(np.median(gray[:edge_h, :]))
    m_bot = float(np.median(gray[-edge_h:, :]))
    m_left = float(np.median(gray[:, :edge_w]))
    m_right = float(np.median(gray[:, -edge_w:]))
    edge_medians = [m_top, m_bot, m_left, m_right]

    # Sample interior core (INTERIOR_BOX_FRACTION_INNER to INTERIOR_BOX_FRACTION_OUTER box)
    center_y1 = int(H * INTERIOR_BOX_FRACTION_INNER)
    center_y2 = int(H * INTERIOR_BOX_FRACTION_OUTER)
    center_x1 = int(W * INTERIOR_BOX_FRACTION_INNER)
    center_x2 = int(W * INTERIOR_BOX_FRACTION_OUTER)
    center_pixels = gray[center_y1:center_y2, center_x1:center_x2].ravel()

    c_p75 = float(np.percentile(center_pixels, 75))
    c_p25 = float(np.percentile(center_pixels, 25))
    c_med = float(np.median(center_pixels))

    # Broad interior for overall page substrate verification
    broad_y1 = int(H * BROAD_BOX_FRACTION_INNER)
    broad_y2 = int(H * BROAD_BOX_FRACTION_OUTER)
    broad_x1 = int(W * BROAD_BOX_FRACTION_INNER)
    broad_x2 = int(W * BROAD_BOX_FRACTION_OUTER)
    broad_pixels = gray[broad_y1:broad_y2, broad_x1:broad_x2].ravel()
    broad_med = float(np.median(broad_pixels))

    # Step 3: Independent edge transition analysis
    # Differences relative to light document paper (c_p75) and dark document paper (c_p25)
    dark_diffs = [c_p75 - m for m in edge_medians]
    light_diffs = [m - c_p25 for m in edge_medians]

    # Slight Gaussian blur to suppress camera sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Condition A: Light Document with Dark Border (on 1, 2, 3, or 4 sides)
    # The document has a light paper substrate (c_p75 > LIGHT_DOC_P75_MIN), and at least one
    # edge is significantly darker than the paper substrate (c_p75 - m >= BORDER_TRANSITION_DELTA)
    if c_p75 > LIGHT_DOC_P75_MIN and max(dark_diffs) >= transition_delta:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Condition B: Dark Document with Light Border (on 1, 2, 3, or 4 sides)
    # The document substrate is dark (c_p25 < DARK_DOC_P25_MAX), at least one edge is bright
    # scanner glass (m - c_p25 >= BORDER_TRANSITION_DELTA), and safeguards confirm it is a genuine
    # dark document rather than a white document with a dark illustration.
    elif (c_p25 < DARK_DOC_P25_MAX) and (max(light_diffs) >= transition_delta):
        # Safeguard: in a borderless white document with illustrations, the broad
        # interior paper is white (broad_med >= WHITE_PAGE_BROAD_MED_MIN) and
        # all edge medians are close to the broad interior (within WHITE_PAGE_EDGE_TOLERANCE).
        is_borderless_white = (broad_med >= WHITE_PAGE_BROAD_MED_MIN) and all(
            abs(m - broad_med) < WHITE_PAGE_EDGE_TOLERANCE for m in edge_medians
        )

        # In a genuine dark document, the dark substrate is dominant (c_med <= DARK_DOC_MEDIAN_MAX)
        # or the dark paper extends deep into the substrate (c_p25 < DARK_DOC_P25_DEEP).
        if (c_med <= DARK_DOC_MEDIAN_MAX or c_p25 < DARK_DOC_P25_DEEP) and not is_borderless_white:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            return 0, H, 0, W
    else:
        # No significant transition on any edge: borderless document or natural margins
        return 0, H, 0, W

    # Step 4: Compute horizontal and vertical projection profiles (custom NumPy functions)
    horizontal_proj = compute_horizontal_projection(thresh)
    vertical_proj = compute_vertical_projection(thresh)

    max_h_proj = float(np.max(horizontal_proj)) if len(horizontal_proj) > 0 else 0.0
    max_v_proj = float(np.max(vertical_proj)) if len(vertical_proj) > 0 else 0.0

    # Step 5: Confidence & reliability check
    if max_h_proj < CONFIDENCE_MIN_PROJ_DENSITY or max_v_proj < CONFIDENCE_MIN_PROJ_DENSITY:
        return 0, H, 0, W

    # Adaptive threshold: at least ADAPTIVE_PROJ_FLOOR or ADAPTIVE_PROJ_SCALE of max density
    h_thresh = max(ADAPTIVE_PROJ_FLOOR, ADAPTIVE_PROJ_SCALE * max_h_proj)
    v_thresh = max(ADAPTIVE_PROJ_FLOOR, ADAPTIVE_PROJ_SCALE * max_v_proj)

    # Number of consecutive rows/cols to confirm true document boundary
    min_consec_h = max(MIN_CONSECUTIVE_HITS, int(H * CONSECUTIVE_FRACTION))
    min_consec_v = max(MIN_CONSECUTIVE_HITS, int(W * CONSECUTIVE_FRACTION))

    top, bottom = scan_boundary_inward(horizontal_proj, threshold=h_thresh, min_consecutive=min_consec_h)
    left, right = scan_boundary_inward(vertical_proj, threshold=v_thresh, min_consecutive=min_consec_v)

    # Step 6: Add safety margin padding to avoid cutting any text near the edge
    pad_h = int(H * safety_margin_pct)
    pad_w = int(W * safety_margin_pct)

    # Only pad inward boundaries; if an edge was not cropped (0 or H/W), keep it at the image limit
    if top > 0:
        top = max(0, top - pad_h)
    if bottom < H:
        bottom = min(H, bottom + pad_h)
    if left > 0:
        left = max(0, left - pad_w)
    if right < W:
        right = min(W, right + pad_w)

    # Step 7: Validation and fallback protection
    crop_h = bottom - top
    crop_w = right - left

    # Sanity checks:
    # - Inverted or degenerate crop
    # - Over-aggressive crop (less than MIN_CROP_FRACTION of original dimensions)
    if crop_h <= 0 or crop_w <= 0:
        return 0, H, 0, W
    if crop_h < int(MIN_CROP_FRACTION * H) or crop_w < int(MIN_CROP_FRACTION * W):
        # Reject unreasonable crops and return original bounds
        return 0, H, 0, W

    return top, bottom, left, right


# ---------------------------------------------------------------------------
# Main public function - matches the team contract exactly
# ---------------------------------------------------------------------------

def remove_borders(img: np.ndarray) -> np.ndarray:
    """
    Detect and remove the surrounding border from a deskewed document image.

    Parameters
    ----------
    img : np.ndarray
        Deskewed BGR colour image from deskew_image(). Shape: (H, W, 3), dtype: uint8.

    Returns
    -------
    np.ndarray
        Cropped BGR colour image containing only the document area.
        Shape: (h, w, 3) where 0 < h <= H and 0 < w <= W, dtype: uint8.
        If detection is unreliable or no border exists, returns the original image.
    """
    # Guard against invalid inputs
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("[border_removal] input image must be a valid NumPy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"[border_removal] expected a 3-channel colour image (H, W, 3), got shape {img.shape}"
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"[border_removal] expected dtype uint8, got {img.dtype}"
        )
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(
            f"[border_removal] expected a non-empty image, got shape {img.shape}"
        )

    H, W = img.shape[:2]

    # Convert to grayscale internally for projection profile analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect boundary coordinates
    top, bottom, left, right = detect_document_boundaries(gray)

    # Crop original BGR colour image using NumPy slicing
    cropped = img[top:bottom, left:right]

    # Final safety check: ensure non-empty valid 3-channel image
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return img

    # Ensure contiguous array in memory
    return np.ascontiguousarray(cropped)
