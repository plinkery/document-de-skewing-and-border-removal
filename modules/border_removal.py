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

50/50 Rule Breakdown:
---------------------
LIBRARY-BASED (OpenCV) — ~50%:
    1. cv2.findContours()            : Extract external topological contours from edge/binary map.
    2. cv2.convexHull()              : Enclose outer document points into a convex polygon.
    3. cv2.approxPolyDP()            : Douglas-Peucker polygon curve simplification.
    4. cv2.getPerspectiveTransform() : Compute 3x3 perspective transformation matrix.
    5. cv2.warpPerspective()         : Geometric quadrilateral perspective warping.
    (Fallback: cv2.minAreaRect + cv2.boxPoints)

CUSTOM / HAND-CODED (NumPy) — ~50%:
    1. custom_order_points()         : Geometric sorting of 4 arbitrary quad vertices into
                                       [top-left, top-right, bottom-right, bottom-left] order
                                       using coordinate sums (x+y) and differences (y-x).
    2. custom_euclidean_distance()   : Pure Pythagorean formula calculation for distance between points.
    3. custom_compute_quad_dimensions(): Calculate target output width and height from corner coordinates.
    4. custom_four_point_transform() : Orchestrates coordinate normalization, dimension derivation,
                                       and perspective mapping.
    5. custom_find_document_quad()   : Multi-tier contour filtering, area/geometry heuristics,
                                       and bounding fallbacks.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Module-level named constants
# ---------------------------------------------------------------------------
# Minimum area ratio of document relative to original image canvas
MIN_QUAD_AREA_RATIO = 0.15

# Safety fallback: minimum dimension in either axis
TINY_IMAGE_MIN_DIM = 10


# ---------------------------------------------------------------------------
# Custom (hand-coded) Mathematical Geometry Helpers  <- 50/50 proof: no cv2
# ---------------------------------------------------------------------------

def custom_euclidean_distance(ptA: np.ndarray, ptB: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 2D points from first principles:
        d = sqrt((xA - xB)^2 + (yA - yB)^2)
    """
    dx = float(ptA[0] - ptB[0])
    dy = float(ptA[1] - ptB[1])
    return float(np.sqrt(dx * dx + dy * dy))


def custom_order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four 2D corner coordinates into a consistent clockwise order:
        [top-left, top-right, bottom-right, bottom-left]

    Algorithm:
    ----------
    - The top-left point has the smallest (x + y) sum.
    - The bottom-right point has the largest (x + y) sum.
    - The top-right point has the smallest difference (y - x), i.e. x is large, y is small.
    - The bottom-left point has the largest difference (y - x), i.e. x is small, y is large.

    Parameters
    ----------
    pts : np.ndarray of shape (4, 2)
        Unordered (x, y) coordinates of the 4 document corners.

    Returns
    -------
    np.ndarray of shape (4, 2), dtype float32
        Ordered points [tl, tr, br, bl].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: s = x + y
    s = pts[:, 0] + pts[:, 1]
    rect[0] = pts[np.argmin(s)]  # Top-left has minimum sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has maximum sum

    # Difference of coordinates: diff = y - x
    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]  # Top-right has minimum diff (large x, small y)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has maximum diff (small x, large y)

    return rect


def custom_compute_quad_dimensions(rect: np.ndarray) -> tuple[int, int]:
    """
    Compute the width and height of the straightened document from ordered corners
    using custom Euclidean distance calculations.

    Parameters
    ----------
    rect : np.ndarray of shape (4, 2)
        Ordered corners: [top-left, top-right, bottom-right, bottom-left].

    Returns
    -------
    tuple[int, int]
        (max_width, max_height) for the output cropped image.
    """
    tl, tr, br, bl = rect[0], rect[1], rect[2], rect[3]

    # Width: max of bottom edge and top edge distance
    widthA = custom_euclidean_distance(br, bl)
    widthB = custom_euclidean_distance(tr, tl)
    maxWidth = max(int(round(widthA)), int(round(widthB)))

    # Height: max of right edge and left edge distance
    heightA = custom_euclidean_distance(tr, br)
    heightB = custom_euclidean_distance(tl, bl)
    maxHeight = max(int(round(heightA)), int(round(heightB)))

    return max(1, maxWidth), max(1, maxHeight)


# ---------------------------------------------------------------------------
# Perspective Warping Pipeline (Hybrid 50/50)
# ---------------------------------------------------------------------------

def custom_four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Straighten and crop an arbitrary 4-sided document quadrilateral into an
    upright rectangular bird's-eye view.

    Parameters
    ----------
    image : np.ndarray
        Original or deskewed BGR image (H, W, 3).
    pts : np.ndarray of shape (4, 2)
        Detected 4 document corner coordinates.

    Returns
    -------
    np.ndarray
        Cropped and perspective-corrected BGR image.
    """
    # 1. Custom math: Order the corners
    rect = custom_order_points(pts)

    # 2. Custom math: Determine target dimensions
    maxWidth, maxHeight = custom_compute_quad_dimensions(rect)

    # 3. Destination coordinates for bird's-eye rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # 4. Library (OpenCV): Compute matrix and warp perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

    return warped


# ---------------------------------------------------------------------------
# Document Quadrilateral Detection
# ---------------------------------------------------------------------------

def custom_find_document_quad(img: np.ndarray) -> np.ndarray:
    """
    Detect the 4 corners of the document in the image.

    Strategy:
    ---------
    1. Downscale image for fast, robust contour analysis.
    2. Convert to grayscale, apply Gaussian blur, and compute Canny edges.
    3. Strengthen page perimeter using morphological dilation and closing.
    4. Find external contours, filter by area, and compute convex hull.
    5. Simplify polygon using Douglas-Peucker (approxPolyDP) to extract 4 corners.
    6. Robust fallbacks: minAreaRect if polygon has >4 vertices, or full-frame
       corners if no valid contour is found.
    """
    H, W = img.shape[:2]
    if H < TINY_IMAGE_MIN_DIM or W < TINY_IMAGE_MIN_DIM:
        return np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)

    # Downscale for processing (height = 500 for optimal noise/detail balance)
    target_h = 500.0
    scale = target_h / float(H)
    target_w = max(1, int(round(W * scale)))
    resized = cv2.resize(img, (target_w, int(target_h)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges with Canny
    edged = cv2.Canny(blurred, 50, 150)

    # Bridge broken boundary lines with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find external contours (OpenCV)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = target_w * int(target_h)
    quad_pts = None

    if contours:
        # Keep contours that cover at least 15% of image area
        large_contours = [c for c in contours if cv2.contourArea(c) > (MIN_QUAD_AREA_RATIO * image_area)]

        if not large_contours:
            # Fallback to the largest available contour
            large_contours = [max(contours, key=cv2.contourArea)]

        # Combine contour points to form convex hull
        all_pts = np.vstack(large_contours).reshape(-1, 2)
        hull = cv2.convexHull(all_pts)

        # Approximate hull with polygon (Douglas-Peucker)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) == 4:
            quad_pts = approx.reshape(4, 2).astype(np.float32)
        elif len(approx) != 4:
            # Try a slightly more relaxed approximation factor
            approx2 = cv2.approxPolyDP(hull, 0.04 * peri, True)
            if len(approx2) == 4:
                quad_pts = approx2.reshape(4, 2).astype(np.float32)
            else:
                # Minimum area bounding box guarantee (always exactly 4 corners)
                min_rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(min_rect)
                quad_pts = box.astype(np.float32)

    # Safeguard: if quad detection failed or area is unreasonable, default to full image
    if quad_pts is None or len(quad_pts) != 4:
        quad_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, int(target_h) - 1],
            [0, int(target_h) - 1]
        ], dtype=np.float32)

    # Map corner coordinates from scaled space back to original full-resolution space
    inv_scale = float(H) / target_h
    quad_orig = quad_pts * inv_scale

    return quad_orig


# ---------------------------------------------------------------------------
# Main Public Pipeline Function (The Contract)
# ---------------------------------------------------------------------------

def remove_borders(img: np.ndarray) -> np.ndarray:
    """
    Detect document boundary and remove external background/borders.

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
    # Guard clauses for valid inputs
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("[border_removal] input image must be a valid NumPy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"[border_removal] expected a 3-channel colour image (H, W, 3), got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"[border_removal] expected dtype uint8, got {img.dtype}")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"[border_removal] expected a non-empty image, got shape {img.shape}")

    H, W = img.shape[:2]

    # Detect document 4-corner quad
    quad = custom_find_document_quad(img)

    # Check if detected quad is effectively the whole image (within 3% margin)
    ordered = custom_order_points(quad)
    w_crop, h_crop = custom_compute_quad_dimensions(ordered)

    if abs(w_crop - W) < (0.03 * W) and abs(h_crop - H) < (0.03 * H):
        # Already borderless or full page
        return img.copy()

    # Apply perspective crop
    cropped = custom_four_point_transform(img, quad)

    # Final safety check: ensure non-empty valid 3-channel image
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return img

    return np.ascontiguousarray(cropped)
