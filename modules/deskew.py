"""
deskew.py - Person 3
======================
Responsibility: Detect the skew angle and rotate the image to straighten it.

INPUT  received from: main.py (both raw image and cleaned binary image)
OUTPUT sent to:       border_removal.py via remove_borders()

CONTRACT - do not change without telling the team:
    deskew_image(raw_img, binary_img) -> tuple[np.ndarray, float]
        IN:  raw_img    - original BGR image,  shape (H, W, 3),  dtype uint8
             binary_img - cleaned binary,       shape (H, W),     dtype uint8
        OUT: (rotated_image, angle)
             rotated_image - BGR image, straightened,  shape (H, W, 3), dtype uint8
             angle         - the skew angle in degrees (float)

        IMPORTANT: return order is (image, angle) - not (angle, image)

Algorithm overview (hybrid 50/50 approach):
-------------------------------------------
LIBRARY-BASED (OpenCV) — ~50%:
    1. cv2.Canny          : Edge detection on the binary image to find sharp transitions.
    2. cv2.HoughLinesP    : Probabilistic Hough Transform to detect line segments in the
                            edge map (each line segment carries an implicit angle).
    3. cv2.getRotationMatrix2D + cv2.warpAffine : Rotate the raw BGR image by the
                            estimated skew angle with high-quality INTER_CUBIC interpolation
                            and constant border fill to avoid cropping content.

CUSTOM HAND-CODED — ~50%:
    4. compute_line_angle()     : Compute the angle (degrees) of a detected line segment
                                  from its (x1,y1,x2,y2) coordinates using atan2.
    5. filter_near_horizontal() : Keep only angles in [-max_skew, +max_skew] because
                                  document skew is small; steep lines are page content,
                                  not the document tilt.
    6. build_angle_histogram()  : Bucket the filtered angles into discrete bins (resolution
                                  = 0.5°) and count how many lines fall in each bin.
                                  This is a custom 1-D histogram built with a plain loop.
    7. find_dominant_angle()    : Walk the histogram to find the bin with the highest count.
                                  Ties are broken by a weighted mean over the top-scoring
                                  bins — also hand-coded without any scipy/numpy stat call.
    8. estimate_skew_angle()    : Orchestrates steps 4-7; also applies a minimum-angle
                                  threshold to avoid rotating nearly-straight images.

Why this approach is appropriate:
----------------------------------
The input images are photographs/scans of documents with text lines.  Document
skew is typically small (< 10°) and the dominant orientation of text lines
produces a strong cluster of near-horizontal angles.  The Hough Transform
efficiently extracts these line segments from the clean binary image produced by
morphology.  The custom angle-histogram then finds the peak orientation robustly,
even when several spurious lines (vertical, diagonal) are present.  This avoids
the brittleness of relying on a single line or a simple mean, while remaining
simple and transparent.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Custom (hand-coded) angle-estimation helpers  <- 50/50 proof: no cv2 here
# ---------------------------------------------------------------------------

def compute_line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Compute the angle of a line segment in degrees using atan2.

    The angle is measured from the positive X-axis (horizontal right),
    so a perfectly horizontal line gives 0°, and a line tilted slightly
    upward to the right gives a small positive angle.

    Parameters
    ----------
    x1, y1 : int   Start point of the line segment.
    x2, y2 : int   End point of the line segment.

    Returns
    -------
    float   Angle in degrees, in the range (-90, +90].

    How it works
    ------------
    We compute angle = atan2(dy, dx) in radians, then convert to degrees.
    Because we only care about the line's orientation (not direction), we
    fold angles outside [-90, 90] back by adding or subtracting 180°.
    """
    dx = x2 - x1
    dy = y2 - y1  # Note: y increases downward in image coordinates

    # atan2 returns angle in (-pi, pi]; convert to degrees
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Fold into (-90, +90] so both directions of the same line map identically
    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0

    return angle_deg


def filter_near_horizontal(angles: list, max_skew: float = 15.0) -> list:
    """
    Retain only angles that could represent document skew.

    Document skew is usually a small tilt (< 10°), and horizontal text lines
    produce near-horizontal Hough line segments.  Steep lines (table borders,
    vertical strokes) are part of the page content and should not influence
    the skew estimate.

    Parameters
    ----------
    angles   : list of float   All line angles from compute_line_angle().
    max_skew : float           Maximum absolute angle accepted (degrees).
                               Default 15° is generous; most scanned docs
                               tilt less than 5°.

    Returns
    -------
    list of float   Subset of angles within [-max_skew, +max_skew].
    """
    filtered = []
    for a in angles:
        if -max_skew <= a <= max_skew:
            filtered.append(a)
    return filtered


def build_angle_histogram(angles: list, bin_width: float = 0.5) -> dict:
    """
    Build a 1-D histogram (frequency table) of angles.

    Each angle is rounded to the nearest multiple of bin_width and placed
    in the corresponding bin.  The histogram is stored as a plain Python
    dict: {bin_centre -> count}.

    Parameters
    ----------
    angles    : list of float   Filtered near-horizontal angles.
    bin_width : float           Width of each histogram bin in degrees.
                                0.5° gives 0.5° angular resolution.

    Returns
    -------
    dict  {float: int}   Mapping from bin centre (rounded angle) to count.

    Core logic
    ----------
    For each angle a:
        bin = round(a / bin_width) * bin_width   <- snap to nearest bin
        histogram[bin] += 1
    This is a hand-coded loop; we do not call np.histogram or any library.
    """
    histogram = {}
    for a in angles:
        # Snap to nearest bin by rounding to multiples of bin_width
        bin_key = round(round(a / bin_width) * bin_width, 6)
        if bin_key in histogram:
            histogram[bin_key] += 1
        else:
            histogram[bin_key] = 1
    return histogram


def find_dominant_angle(histogram: dict) -> float:
    """
    Find the dominant (most frequent) angle from the histogram.

    Strategy
    --------
    1. Find the maximum count across all bins.
    2. Collect every bin that has count >= (max_count * 0.9) — the "top tier".
    3. Compute a weighted mean of the top-tier bin centres, weighted by their
       counts.  This smooths over the effect of choosing between two
       neighbouring bins that together hold the dominant cluster.

    Parameters
    ----------
    histogram : dict  {float: int}   From build_angle_histogram().

    Returns
    -------
    float   Dominant skew angle in degrees.

    Why weighted mean instead of plain argmax
    ------------------------------------------
    If the peak is spread across two adjacent bins (e.g. 1.0° has 20 hits
    and 1.5° has 19 hits), argmax returns 1.0° but the true dominant angle
    is closer to 1.25°.  The weighted mean gives a more accurate estimate
    without requiring any scipy or numpy statistics call.
    """
    if not histogram:
        return 0.0

    # Step 1: find the maximum count (hand-coded max search)
    max_count = 0
    for count in histogram.values():
        if count > max_count:
            max_count = count

    # Step 2: collect the top-tier bins (>= 90% of max count)
    threshold = max_count * 0.9
    top_bins = []
    for angle, count in histogram.items():
        if count >= threshold:
            top_bins.append((angle, count))

    # Step 3: weighted mean of top-tier bins (hand-coded)
    total_weight = 0.0
    weighted_sum = 0.0
    for angle, count in top_bins:
        weighted_sum += angle * count
        total_weight += count

    if total_weight == 0.0:
        return 0.0

    return weighted_sum / total_weight


def estimate_skew_angle(
    lines,
    max_skew: float = 15.0,
    min_angle_threshold: float = 0.3,
) -> float:
    """
    Estimate the document's skew angle from a set of Hough line segments.

    This function orchestrates the full custom angle-estimation pipeline:
        compute_line_angle → filter_near_horizontal → build_angle_histogram
        → find_dominant_angle

    Parameters
    ----------
    lines               : np.ndarray or None
        Output of cv2.HoughLinesP — shape (N, 1, 4) where each row is
        [x1, y1, x2, y2], or None if no lines were detected.
    max_skew            : float
        Maximum skew angle considered (degrees).  Lines beyond this are
        ignored as they represent content, not tilt.
    min_angle_threshold : float
        If the estimated dominant angle is smaller than this value (in
        absolute terms), the image is considered already straight and
        0.0 is returned.  This prevents unnecessary rotation of images
        that are already well-aligned.

    Returns
    -------
    float   Estimated skew angle in degrees.  Positive = counter-clockwise
            tilt that needs a clockwise correction; the sign is preserved so
            the caller can apply -angle to straighten.

    Algorithm steps (all hand-coded):
    ----------------------------------
    1. Loop over all detected line segments; compute each angle.
    2. Filter to keep only near-horizontal angles.
    3. If fewer than 3 near-horizontal lines found → return 0.0 (not enough
       evidence to estimate skew reliably).
    4. Build histogram and find the dominant bin.
    5. Apply minimum-angle guard: if |dominant| < min_angle_threshold → 0.0.
    """
    # No lines detected
    if lines is None or len(lines) == 0:
        return 0.0

    # Step 1: compute angle for every detected line segment.
    # OpenCV <5 returns shape (N, 1, 4); OpenCV >=5 returns shape (N, 4).
    # Flatten each row to a 1-D sequence of 4 values to handle both formats.
    all_angles = []
    for line in lines:
        row = line.flatten()          # works for both (1,4) and (4,) rows
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        angle = compute_line_angle(x1, y1, x2, y2)
        all_angles.append(angle)

    # Step 2: keep only near-horizontal angles
    near_horizontal = filter_near_horizontal(all_angles, max_skew=max_skew)

    # Step 3: require at least 3 qualifying lines for a reliable estimate
    if len(near_horizontal) < 3:
        return 0.0

    # Step 4: build histogram and find dominant angle
    histogram = build_angle_histogram(near_horizontal, bin_width=0.5)
    dominant = find_dominant_angle(histogram)

    # Step 5: ignore sub-threshold angles (image is already straight enough)
    if abs(dominant) < min_angle_threshold:
        return 0.0

    return dominant


# ---------------------------------------------------------------------------
# Rotation helper (uses OpenCV for high-quality affine transform)
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a BGR image by the given angle around its centre.

    Uses OpenCV's getRotationMatrix2D + warpAffine (the library-based half
    of the 50/50 requirement).  The canvas size is kept identical to the
    input so the pipeline contract (same H×W×3 shape) is preserved.

    Parameters
    ----------
    image : np.ndarray   BGR image, shape (H, W, 3), dtype uint8.
    angle : float        Rotation angle in degrees.  Positive values rotate
                         counter-clockwise in OpenCV convention.

    Returns
    -------
    np.ndarray   Rotated BGR image.  Shape: (H, W, 3).  Dtype: uint8.
                 Pixels outside the original image bounds are filled with
                 white (255, 255, 255) to match the typical white background
                 of scanned documents.

    Why INTER_CUBIC and BORDER_CONSTANT with white fill:
    ----------------------------------------------------
    INTER_CUBIC provides smoother interpolation than INTER_LINEAR for small
    rotation angles, reducing staircase artefacts on text strokes.
    BORDER_CONSTANT with a white fill avoids black corners (which can confuse
    the subsequent border_removal step into thinking there is content there).
    """
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)

    # Build 2×3 affine rotation matrix (OpenCV library call)
    rotation_matrix = cv2.getRotationMatrix2D(centre, angle, scale=1.0)

    # Apply the rotation (OpenCV library call)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),   # white fill for clean background
    )
    return rotated


# ---------------------------------------------------------------------------
# Main public function - this is what the pipeline calls
# ---------------------------------------------------------------------------

def deskew_image(raw_img: np.ndarray, binary_img: np.ndarray) -> tuple:
    """
    Detect skew angle and rotate the image to straighten the document.

    Parameters
    ----------
    raw_img : np.ndarray
        Original BGR image loaded by cv2.imread(). Shape: (H, W, 3).
    binary_img : np.ndarray
        Cleaned binary image from apply_morphology(). Shape: (H, W).
        Values are 0 (background) or 255 (foreground/text).

    Returns
    -------
    tuple[np.ndarray, float]
        (rotated_image, angle)
        rotated_image: Straightened BGR image. Shape: (H, W, 3). Dtype: uint8.
        angle: The skew angle in degrees that was detected and corrected. float.

    Processing pipeline
    -------------------
    Step 1 — Input validation (guard clauses)
        Verify that both images have the expected shapes and dtypes so that
        the subsequent operations do not silently produce wrong results.

    Step 2 — Canny edge detection  [LIBRARY: cv2.Canny]
        Applied to the cleaned binary image (not the raw colour image).
        The binary image, after morphological cleaning, already has clear
        foreground/background boundaries; Canny sharpens these into a thin
        1-pixel-wide edge map suitable for Hough line detection.

    Step 3 — Hough line detection  [LIBRARY: cv2.HoughLinesP]
        Probabilistic Hough Transform extracts explicit line segments from
        the edge map.  Parameters are tuned for typical document images:
        - rho=1 (1-pixel resolution in distance)
        - theta=pi/180 (1° angular resolution)
        - threshold=80 (minimum votes for a line to be reported)
        - minLineLength=max(50, min(H,W)//10): ignore very short fragments
        - maxLineGap=10: allow small breaks within a line

    Step 4 — Custom skew-angle estimation  [CUSTOM: estimate_skew_angle]
        Calls our hand-coded pipeline:
            compute_line_angle → filter_near_horizontal →
            build_angle_histogram → find_dominant_angle
        Returns the dominant skew angle (or 0.0 if no reliable angle found).

    Step 5 — Image rotation  [LIBRARY: rotate_image → warpAffine]
        Rotates the raw BGR image by -angle (negated to correct the skew).
        The binary image is NOT rotated — the pipeline only needs the
        corrected colour image downstream.
    """
    # ---- Step 1: Input validation ------------------------------------------

    # Handle None inputs gracefully
    if raw_img is None:
        raise ValueError("deskew_image: raw_img is None")
    if binary_img is None:
        raise ValueError("deskew_image: binary_img is None")

    # Validate raw_img: must be (H, W, 3) uint8 colour image
    if raw_img.ndim != 3 or raw_img.shape[2] != 3:
        raise ValueError(
            f"deskew_image: raw_img must be a 3-channel BGR image (H, W, 3), "
            f"got shape {raw_img.shape}"
        )
    if raw_img.dtype != np.uint8:
        raise ValueError(
            f"deskew_image: raw_img dtype must be uint8, got {raw_img.dtype}"
        )

    # Validate binary_img: must be (H, W) uint8 with only {0, 255}
    if binary_img.ndim != 2:
        raise ValueError(
            f"deskew_image: binary_img must be a 2-D image (H, W), "
            f"got shape {binary_img.shape}"
        )
    if binary_img.dtype != np.uint8:
        raise ValueError(
            f"deskew_image: binary_img dtype must be uint8, got {binary_img.dtype}"
        )

    # If the binary image is all-black (blank), skip processing
    if np.count_nonzero(binary_img) == 0:
        return raw_img.copy(), 0.0

    # ---- Step 2: Canny edge detection  [LIBRARY — cv2.Canny] ---------------
    #
    # We apply Canny directly on the cleaned binary image.  Since the binary
    # image has only 0/255 pixel values, Canny will produce clean, thin edges
    # along the boundary of every foreground region (text stroke edges).
    # Thresholds 50/150 are standard for binary inputs.
    #
    edges = cv2.Canny(binary_img, threshold1=50, threshold2=150, apertureSize=3)

    # ---- Step 3: Hough line detection  [LIBRARY — cv2.HoughLinesP] ---------
    #
    # Probabilistic Hough Transform: returns explicit (x1,y1,x2,y2) line
    # segments rather than infinite lines.  This is more robust for documents
    # where text lines are finite in length.
    #
    h, w = binary_img.shape
    min_line_length = max(50, min(h, w) // 10)   # adapt to image size
    max_line_gap = 10

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    # ---- Step 4: Custom skew-angle estimation  [CUSTOM] --------------------
    #
    # Calls our hand-coded pipeline:
    #   compute_line_angle()         : angle from (x1,y1,x2,y2) via atan2
    #   filter_near_horizontal()     : keep only lines within ±15° of horizontal
    #   build_angle_histogram()      : count lines per 0.5° bin
    #   find_dominant_angle()        : weighted mean of top bins
    #
    skew_angle = estimate_skew_angle(lines, max_skew=15.0, min_angle_threshold=0.3)

    # ---- Step 5: Image rotation  [LIBRARY — cv2.warpAffine via rotate_image]
    #
    # Rotate the raw BGR image by -skew_angle to cancel the detected tilt.
    # The negation converts our measured skew angle (how much the document
    # is tilted) into the correction angle (how much we must rotate back).
    #
    if skew_angle == 0.0:
        # No rotation needed; return a copy to maintain consistent behaviour
        rotated_img = raw_img.copy()
    else:
        rotated_img = rotate_image(raw_img, angle=-skew_angle)

    # Ensure the output dtype is uint8 (rotation should preserve it, but guard)
    if rotated_img.dtype != np.uint8:
        rotated_img = rotated_img.astype(np.uint8)

    return rotated_img, float(skew_angle)
