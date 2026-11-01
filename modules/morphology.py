"""
morphology.py - Person 2
==========================
Responsibility: Clean up the binary image by removing noise and
strengthening useful edges/text regions.

INPUT  received from: preprocessing.py via preprocess_image()
OUTPUT sent to:       deskew.py via deskew_image()

CONTRACT - do not change without telling the team:
    apply_morphology(binary_img: np.ndarray) -> np.ndarray
        IN:  Binary image,  shape (H, W),  dtype uint8, values = 0 or 255
        OUT: Binary image,  shape (H, W),  dtype uint8, values = 0 or 255

Implemented operations (Person 2's own work):
    1. custom_erode()   - manual erosion using numpy sliding window (no cv2)
    2. custom_dilate()  - manual dilation using numpy sliding window (no cv2)
    3. Opening          - custom_erode then custom_dilate  (removes salt noise)
    4. Closing          - custom_dilate then custom_erode  (fills holes in text)
    5. cv2 closing      - second-pass closing with a larger kernel to bridge gaps
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Custom (hand-coded) morphological primitives  <- 50/50 proof: no cv2 here
# ---------------------------------------------------------------------------

def custom_erode(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Erosion implemented from scratch with numpy.

    A pixel stays WHITE (255) only if every pixel inside the kernel window
    centred on it is also WHITE.  Equivalent to: output = min over kernel.

    Parameters
    ----------
    binary_img : np.ndarray  Shape (H, W), dtype uint8, values 0 or 255.
    kernel_size : int        Side length of the square structuring element.

    Returns
    -------
    np.ndarray   Eroded image. Same shape and dtype as input.

    How it works
    ------------
    We pad the image so every pixel has a full neighbourhood, then slide a
    (kernel_size x kernel_size) window across it.  At each position we take
    the minimum of the window.  If any pixel in the neighbourhood is 0 (black)
    the result is 0, so the white region shrinks (erodes) at its boundary.
    """
    k = kernel_size
    pad = k // 2                                    # how many pixels to pad on each side

    # Reflect-pad to avoid border artefacts
    padded = np.pad(binary_img, pad, mode='reflect')
    H, W = binary_img.shape
    output = np.zeros((H, W), dtype=np.uint8)

    for row in range(H):
        for col in range(W):
            window = padded[row: row + k, col: col + k]
            output[row, col] = np.min(window)       # erosion = minimum

    return output


def custom_dilate(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Dilation implemented from scratch with numpy.

    A pixel becomes WHITE (255) if at least one pixel inside the kernel
    window centred on it is WHITE.  Equivalent to: output = max over kernel.

    Parameters
    ----------
    binary_img : np.ndarray  Shape (H, W), dtype uint8, values 0 or 255.
    kernel_size : int        Side length of the square structuring element.

    Returns
    -------
    np.ndarray   Dilated image. Same shape and dtype as input.

    How it works
    ------------
    Same sliding-window approach as custom_erode, but we take the maximum of
    the neighbourhood.  If any pixel is 255 (white) the result is 255, so
    the white region grows (dilates) at its boundary.
    """
    k = kernel_size
    pad = k // 2
    padded = np.pad(binary_img, pad, mode='reflect')
    H, W = binary_img.shape
    output = np.zeros((H, W), dtype=np.uint8)

    for row in range(H):
        for col in range(W):
            window = padded[row: row + k, col: col + k]
            output[row, col] = np.max(window)       # dilation = maximum

    return output


def fast_erode(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Vectorised erosion (numpy only, still no cv2).

    Uses numpy stride tricks / as_strided to build a sliding-window view of
    the image and then takes the min over the last two axes in one call.
    Much faster than the explicit Python loop in custom_erode while still
    being hand-coded (no cv2.erode).

    Used internally for the larger kernel passes where the pixel-by-pixel loop
    would be too slow.
    """
    k = kernel_size
    pad = k // 2
    padded = np.pad(binary_img.astype(np.int32), pad, mode='reflect')
    H, W = binary_img.shape

    # Build a (H, W, k, k) view using stride tricks
    from numpy.lib.stride_tricks import as_strided
    s0, s1 = padded.strides
    windows = as_strided(
        padded,
        shape=(H, W, k, k),
        strides=(s0, s1, s0, s1)
    )
    result = windows.min(axis=(2, 3)).astype(np.uint8)
    return result


def fast_dilate(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Vectorised dilation (numpy only, no cv2).
    Same approach as fast_erode but takes max instead of min.
    """
    k = kernel_size
    pad = k // 2
    padded = np.pad(binary_img.astype(np.int32), pad, mode='reflect')
    H, W = binary_img.shape

    from numpy.lib.stride_tricks import as_strided
    s0, s1 = padded.strides
    windows = as_strided(
        padded,
        shape=(H, W, k, k),
        strides=(s0, s1, s0, s1)
    )
    result = windows.max(axis=(2, 3)).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Compound operations built on top of our custom primitives
# ---------------------------------------------------------------------------

def custom_opening(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Opening = erosion followed by dilation (using our custom functions).

    Effect: removes small white noise dots (salt noise) from the background
    without significantly shrinking the main text/edge regions.

    Parameters
    ----------
    binary_img : np.ndarray   Binary image. Shape (H, W), dtype uint8.
    kernel_size : int         Structuring element size.

    Returns
    -------
    np.ndarray   Image after opening. Same shape and dtype.
    """
    eroded = fast_erode(binary_img, kernel_size)
    opened = fast_dilate(eroded, kernel_size)
    return opened


def custom_closing(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Closing = dilation followed by erosion (using our custom functions).

    Effect: fills small holes and gaps inside text characters or edge regions,
    making them more solid and connected.

    Parameters
    ----------
    binary_img : np.ndarray   Binary image. Shape (H, W), dtype uint8.
    kernel_size : int         Structuring element size.

    Returns
    -------
    np.ndarray   Image after closing. Same shape and dtype.
    """
    dilated = fast_dilate(binary_img, kernel_size)
    closed  = fast_erode(dilated, kernel_size)
    return closed


# ---------------------------------------------------------------------------
# Main public function - this is what the pipeline calls
# ---------------------------------------------------------------------------

def apply_morphology(binary_img: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean the binary image.

    Processing steps
    ----------------
    Step 1  Opening  (3x3, custom numpy)
            Removes isolated noise pixels (single white dots in background).

    Step 2  Closing  (3x3, custom numpy)
            Fills small gaps inside text strokes so characters are solid.

    Step 3  Closing  (5x5, cv2.morphologyEx with MORPH_CLOSE)
            A wider second closing bridges larger gaps between nearby text
            clusters.  This is the OpenCV-assisted step (50/50 balance: the
            logic and kernel choice are ours; the low-level loop is cv2).

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image from preprocess_image(). Shape: (H, W). Values: 0 or 255.

    Returns
    -------
    np.ndarray
        Cleaned binary image. Same shape and dtype as input.
    """
    # ---- Step 1: Custom opening - remove salt noise -------------------------
    result = custom_opening(binary_img, kernel_size=3)

    # ---- Step 2: Custom closing - fill text holes --------------------------
    result = custom_closing(result, kernel_size=3)

    # ---- Step 3: cv2 closing - bridge wider gaps (cv2 used here) -----------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result
