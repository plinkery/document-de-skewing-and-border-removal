"""
preprocessing.py - Person 1
============================
Responsibility: Convert the raw input image into a clean binary image.

INPUT  received from: main.py (raw cv2.imread result)
OUTPUT sent to:       morphology.py via apply_morphology()

CONTRACT - do not change without telling the team:
    preprocess_image(img: np.ndarray) -> np.ndarray
        IN:  BGR colour image,  shape (H, W, 3),  dtype uint8
        OUT: Binary image,      shape (H, W),     dtype uint8, values = 0 or 255

50/50 Rule Breakdown:
---------------------
Custom / Scratch (Person 1's own work):
    1. custom_compute_histogram() - Computes 256-bin intensity distribution from scratch
    2. custom_contrast_stretch()  - Linear min-max dynamic range stretching from scratch
    3. custom_otsu_threshold()    - Otsu's between-class variance maximization from scratch
    4. custom_apply_threshold()   - Binary threshold mapping from scratch into {0, 255}

Library / OpenCV (Allowed 50%):
    1. cv2.cvtColor()             - BGR to single-channel Grayscale conversion
    2. cv2.GaussianBlur()         - 5x5 smoothing to suppress high-frequency sensor noise
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Custom (hand-coded) Preprocessing Primitives <- 50/50 proof: no cv2 here
# ---------------------------------------------------------------------------

def custom_compute_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Compute a 256-bin intensity histogram from scratch using NumPy.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of shape (H, W) with dtype uint8.

    Returns
    -------
    np.ndarray
        1D array of length 256, where hist[i] is the number of pixels with intensity i.
    """
    hist = np.zeros(256, dtype=np.int64)
    # Flatten image and count pixel frequencies
    flat = gray.ravel()
    for val in flat:
        hist[val] += 1
    return hist


def custom_contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """
    Linearly stretch pixel intensities to span the full dynamic range [0, 255].

    This enhances document contrast before thresholding, particularly useful
    for photographs with uneven lighting or washed-out backgrounds.

    Formula
    -------
    stretched = ((pixel - min_val) / (max_val - min_val)) * 255

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of shape (H, W), dtype uint8.

    Returns
    -------
    np.ndarray
        Contrast-stretched grayscale image of shape (H, W), dtype uint8.
    """
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))

    # Guard against flat / zero-variance images
    if max_val == min_val:
        return gray.copy()

    stretched = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def custom_otsu_threshold(gray: np.ndarray) -> int:
    """
    Compute Otsu's optimal bimodal threshold from first principles.

    Otsu's method searches exhaustively for the threshold t* that maximizes
    the between-class variance (sigma_b^2), which mathematically minimizes
    the intra-class variance of the two pixel populations:
        - Class 0 (dark / text):        intensities < t
        - Class 1 (bright / background): intensities >= t

    Mathematical Formulation
    ------------------------
    1. Probability distribution:
       p(i) = hist[i] / total_pixels,  for i = 0, ..., 255

    2. Cumulative weights (probabilities):
       w0(t) = sum_{i=0}^{t-1} p(i)      (weight of Class 0)
       w1(t) = sum_{i=t}^{255} p(i) = 1 - w0(t)  (weight of Class 1)

    3. Class means:
       mu0(t) = (1 / w0(t)) * sum_{i=0}^{t-1} (i * p(i))
       mu1(t) = (1 / w1(t)) * sum_{i=t}^{255} (i * p(i))

    4. Between-class variance:
       sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2

    5. Optimal threshold:
       t* = argmax_{t in [1, 254]} sigma_b^2(t)

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of shape (H, W), dtype uint8.

    Returns
    -------
    int
        Optimal integer threshold in range [0, 255].
    """
    total_pixels = gray.size
    if total_pixels == 0:
        return 128

    hist = custom_compute_histogram(gray)
    prob = hist.astype(np.float64) / total_pixels

    # Precompute global mean
    intensity_indices = np.arange(256, dtype=np.float64)
    total_mean = np.sum(intensity_indices * prob)

    w0 = 0.0
    sum0 = 0.0
    max_variance = -1.0
    best_threshold = 128

    # Iterate over all candidate thresholds t in 1..254
    for t in range(1, 255):
        w0 += prob[t - 1]
        if w0 <= 0.0:
            continue
        w1 = 1.0 - w0
        if w1 <= 0.0:
            break

        sum0 += (t - 1) * prob[t - 1]
        mu0 = sum0 / w0
        mu1 = (total_mean - sum0) / w1

        # Between-class variance: sigma_b^2 = w0 * w1 * (mu0 - mu1)^2
        variance = w0 * w1 * ((mu0 - mu1) ** 2)

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return int(best_threshold)


def custom_apply_threshold(gray: np.ndarray, threshold: int) -> np.ndarray:
    """
    Apply binary thresholding to convert grayscale intensities into {0, 255}.

    Convention:
        - Bright background (paper) > threshold  -> 255 (white)
        - Dark content (ink / text) <= threshold -> 0   (black)

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of shape (H, W), dtype uint8.
    threshold : int
        Intensity cutoff value in [0, 255].

    Returns
    -------
    np.ndarray
        Binary image of shape (H, W), dtype uint8 with values in {0, 255}.
    """
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray > threshold] = 255
    return binary


# ---------------------------------------------------------------------------
# Main Public Pipeline Function (The Contract)
# ---------------------------------------------------------------------------

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR colour image to a clean binary (black-and-white) image.

    Pipeline Steps:
    ---------------
    Step 1 (OpenCV): Grayscale Conversion
        Convert BGR colour image to a single-channel grayscale image using
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).

    Step 2 (OpenCV): Gaussian Noise Smoothing
        Apply a subtle 5x5 Gaussian blur (cv2.GaussianBlur) to suppress
        high-frequency camera sensor noise and paper grain without blurring
        text edges.

    Step 3 (Custom NumPy): Dynamic Range Contrast Stretching
        Normalize intensity range to [0, 255] using custom_contrast_stretch().

    Step 4 (Custom NumPy): Otsu's Threshold Selection
        Analyze the 256-bin intensity histogram and compute the optimal
        separation threshold t* via custom_otsu_threshold().

    Step 5 (Custom NumPy): Binary Threshold Mapping
        Binarize the image into strictly {0, 255} using custom_apply_threshold().

    Parameters
    ----------
    img : np.ndarray
        Raw BGR image loaded by cv2.imread(). Shape: (H, W, 3), dtype: uint8.

    Returns
    -------
    np.ndarray
        Clean binary image. Shape: (H, W), dtype: uint8, values strictly in {0, 255}.
    """
    if img is None:
        raise ValueError("Input image is None. Please provide a valid loaded image array.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected BGR 3-channel image (H, W, 3), got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {img.dtype}")

    # ---- Step 1: Grayscale Conversion (OpenCV) ------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Step 2: Gaussian Blur (OpenCV) -------------------------------------
    # Smooth slight camera noise so the histogram reflects document modes
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---- Step 3: Contrast Stretching (Custom NumPy) -------------------------
    stretched = custom_contrast_stretch(blurred)

    # ---- Step 4: Otsu Optimal Threshold Selection (Custom Hand-Coded) -------
    optimal_t = custom_otsu_threshold(stretched)

    # ---- Step 5: Binary Thresholding (Custom NumPy) -------------------------
    binary = custom_apply_threshold(stretched, optimal_t)

    return binary
