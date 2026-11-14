# Algorithm Notes — Custom Implementations
<!--
  This file proves the 50/50 rule to the grader.
  Each person documents every algorithm they wrote themselves (not just OpenCV calls).
  Fill in your section when your module is complete.
-->

Use this file to document every part of the code that is YOUR OWN work,
not just a call to a built-in function.  The grader uses this as proof that
approximately 50% of the implementation is hand-coded.

---

## Person 1 — Custom Threshold Selection & Preprocessing  (preprocessing.py)

**Algorithm name:** Hand-Coded Otsu's Bimodal Variance Maximization & Linear Contrast Normalization
**Implemented by:** Person 1
**File and functions:** `modules/preprocessing.py → custom_compute_histogram(), custom_contrast_stretch(), custom_otsu_threshold(), custom_apply_threshold()`

**What they do:**

- `custom_compute_histogram`: Generates a 256-bin intensity histogram directly using NumPy array traversal, counting exact frequency distributions without calling `cv2.calcHist`.
- `custom_contrast_stretch`: Linearly expands compressed dynamic ranges across $[0, 255]$ from first principles, ensuring low-contrast or poorly illuminated scans have distinct background/foreground peaks.
- `custom_otsu_threshold`: Implements Nobuyuki Otsu's discriminant analysis from first principles without calling `cv2.threshold(..., cv2.THRESH_OTSU)`. It exhaustively evaluates all candidate thresholds $t \in [1, 254]$ to find the value that maximizes the statistical between-class variance.
- `custom_apply_threshold`: Vectorized mapping of grayscale intensity values into strictly binary $\{0, 255\}$ representations without external thresholding wrappers.

**Inputs and outputs:**

- Input: Raw BGR colour image `(H, W, 3)`, `dtype=uint8`
- Output: Binary image `(H, W)`, `dtype=uint8`, values strictly in $\{0, 255\}$

**Mathematical formulation & core logic:**

1. **Histogram & Probability Distribution**:
   For total pixels $N = H \times W$ and pixel intensity $i \in \{0, 1, \dots, 255\}$:
   $$p(i) = \frac{\text{hist}[i]}{N}$$

2. **Cumulative Class Probabilities (Weights)**:
   For any candidate threshold $t$:
   $$\omega_0(t) = \sum_{i=0}^{t-1} p(i) \quad (\text{Background / Text Class } C_0)$$
   $$\omega_1(t) = \sum_{i=t}^{255} p(i) = 1 - \omega_0(t) \quad (\text{Foreground / Paper Class } C_1)$$

3. **Class Means**:
   $$\mu_0(t) = \frac{1}{\omega_0(t)} \sum_{i=0}^{t-1} i \cdot p(i)$$
   $$\mu_1(t) = \frac{1}{\omega_1(t)} \sum_{i=t}^{255} i \cdot p(i) = \frac{\mu_T - \sum_{i=0}^{t-1} i \cdot p(i)}{\omega_1(t)}$$
   where $\mu_T = \sum_{i=0}^{255} i \cdot p(i)$ is the total mean intensity of the image.

4. **Between-Class Variance ($\sigma_B^2$) Maximization**:
   The total variance is decomposed into within-class variance ($\sigma_W^2$) and between-class variance ($\sigma_B^2$):
   $$\sigma_T^2 = \sigma_W^2(t) + \sigma_B^2(t)$$
   Because $\sigma_T^2$ is constant for a given image, minimizing intra-class variance $\sigma_W^2(t)$ is mathematically equivalent to maximizing between-class variance:
   $$\sigma_B^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot (\mu_0(t) - \mu_1(t))^2$$

5. **Optimal Threshold Selection**:
   $$t^* = \arg\max_{1 \le t \le 254} \sigma_B^2(t)$$

**Why this approach:**
Documents naturally exhibit a bimodal intensity distribution (dark ink versus bright paper). Global fixed thresholds (such as 128) fail on real-world mobile captures with varying ambient lighting, paper discoloration, and camera exposure. Hand-coding Otsu's method guarantees an mathematically optimal cut-point adapted specifically to each individual image's histogram, without requiring arbitrary magic numbers or reliance on OpenCV's internal thresholding implementation.

**50/50 Pipeline Applied in `preprocess_image()`:**

| Step | Operation | Implementation | Purpose | Code Used |
|------|-----------|----------------|---------|-----------|
| 1 | Grayscale conversion | OpenCV (`cv2.cvtColor`) | Convert 3-channel BGR to single-channel luminance | OpenCV |
| 2 | Sensor noise reduction | OpenCV (`cv2.GaussianBlur`) | 5×5 smoothing to reduce scanner sensor noise | OpenCV |
| 3 | Contrast normalization | `custom_contrast_stretch()` | Rescale intensity range to $[0, 255]$ | **Own Code** |
| 4 | Histogram extraction | `custom_compute_histogram()` | Tabulate 256-bin pixel counts | **Own Code** |
| 5 | Skew/Threshold estimation | `custom_otsu_threshold()` | Compute optimal bimodal threshold $t^*$ | **Own Code** |
| 6 | Binary quantification | `custom_apply_threshold()` | Threshold image into strictly $\{0, 255\}$ | **Own Code** |

**How it compares to OpenCV:**
`cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` performs Otsu's calculation inside compiled C++ code. Our implementation achieves mathematically equivalent threshold values ($\pm 1$ intensity level due to integer rounding) entirely in transparent Python and NumPy, fully satisfying the 50/50 custom algorithm requirement.

---

## Person 2 — Custom Erosion, Dilation, Opening, Closing  (morphology.py)

**Algorithm name:** Sliding-window morphological primitives
**Implemented by:** Person 2
**File and functions:** `modules/morphology.py → custom_erode(), custom_dilate(),
fast_erode(), fast_dilate(), custom_opening(), custom_closing()`

**What they do:**

`custom_erode` and `custom_dilate` implement morphological erosion and dilation
from first principles using NumPy, without calling `cv2.erode` or `cv2.dilate`.
`fast_erode` and `fast_dilate` achieve the same result with a vectorised
NumPy stride-trick approach that avoids Python loops.
`custom_opening` and `custom_closing` compose these primitives into the standard
opening (erode→dilate) and closing (dilate→erode) operations.

**Inputs and outputs:**

- Input:  Binary image `(H, W)`, `dtype=uint8`, values `{0, 255}`
- Output: Binary image `(H, W)`, `dtype=uint8`, values `{0, 255}`
- Parameter: `kernel_size` (int) — side length of the square structuring element

**Core logic — erosion:**

```
pad image by (kernel_size // 2) on all sides using reflect mode
for each pixel (row, col):
    extract window = padded[row : row+k, col : col+k]
    output[row, col] = min(window)        ← if any neighbour is 0, result is 0
```

**Core logic — dilation:**

```
pad image by (kernel_size // 2) on all sides using reflect mode
for each pixel (row, col):
    extract window = padded[row : row+k, col : col+k]
    output[row, col] = max(window)        ← if any neighbour is 255, result is 255
```

**Core logic — vectorised version (fast_erode / fast_dilate):**

```
use numpy.lib.stride_tricks.as_strided to build a (H, W, k, k) view
  of the padded image without copying data
take .min(axis=(2,3)) or .max(axis=(2,3)) in one numpy call
→ same result as the loop, ~100× faster for large images
```

**Why this approach:**
Erosion and dilation are defined mathematically as min/max over a neighbourhood.
The sliding-window implementation makes this definition directly visible in code,
which makes it easy to explain and verify.  The vectorised version uses
`numpy.lib.stride_tricks.as_strided` — a standard NumPy technique — so it
does not rely on any OpenCV internals.

**How it compares to OpenCV:**
`cv2.erode(img, kernel)` and `cv2.dilate(img, kernel)` call optimised C++ code
internally.  Our implementation produces identical results but is written
entirely in Python + NumPy, making the algorithm fully transparent.

**pipeline applied in `apply_morphology()`:**

| Step | Operation | Kernel | Purpose | Code used |
|------|-----------|--------|---------|-----------|
| 1 | Opening | 3×3 | Remove isolated noise pixels (salt noise) | `custom_opening()` (own code) |
| 2 | Closing | 3×3 | Fill small gaps inside text strokes | `custom_closing()` (own code) |
| 3 | Closing | 5×5 | Bridge wider gaps between text clusters | `cv2.morphologyEx` (OpenCV) |

---

## Person 3 — Custom Skew Angle Estimation  (deskew.py)

*(Fill in when implemented)*

**Algorithm name:**
**Implemented by:** Person 3
**File and function:** `modules/deskew.py → deskew_image()`

**What it does:**

**Inputs and outputs:**

**Core logic / pseudocode:**
```
run Canny edge detection on the binary image
run HoughLinesP to get a list of line segments
for each line segment:
    compute angle = atan2(y2-y1, x2-x1) in degrees
filter angles to keep only near-horizontal lines (e.g. -45° to +45°)
build a histogram of the filtered angles
dominant angle = bin with highest count (or weighted mean of top bin)
rotate raw_img by -dominant_angle using warpAffine
return (rotated_image, dominant_angle)
```

**Why this approach:**

**OpenCV alternative:**

---

## Person 4 — Custom Border Detection  (border_removal.py)

**Algorithm name:** Row/Column Projection Scanning
**Implemented by:** Person 4
**File and functions:** `modules/border_removal.py → remove_borders(), detect_document_boundaries(), compute_horizontal_projection(), compute_vertical_projection(), scan_boundary_inward()`

**What it does:**

`remove_borders()` accepts a non-empty BGR `uint8` image and returns a
contiguous BGR crop with the same dtype. It converts the image to grayscale
only for analysis, detects the four page limits, and slices the original color
image so the output retains its source pixels.

**Detection logic:**

1. Non-empty images smaller than 10 pixels in either dimension, and images
  with less than 15 levels of grayscale contrast, are treated as unreliable
  and kept unchanged.
2. The median intensity of each outer 2% edge strip is compared with percentiles
   of the central 20%–80% region. The 75th percentile estimates light paper;
   the 25th percentile and median help identify dark paper. A broad 10%–90%
   interior median helps avoid mistaking natural white margins around dark
   illustrations for an external light border.
3. If an edge differs from the estimated substrate by at least 30 intensity
   levels, the image is blurred with a 5×5 Gaussian kernel and Otsu-thresholded
   with the polarity selected for light or dark document content. Dark-page
   detection additionally requires dark-substrate evidence and applies the
   borderless-white-page safeguard.
4. The binary mask is reduced to row and column densities: each projection
   value is the fraction of white mask pixels in that row or column. The scan
   threshold is the greater of 0.04 and 8% of the strongest projection. Each
   edge scan requires at least three consecutive qualifying rows or columns
   (or 0.2% of the corresponding dimension, whichever is larger).
5. A 0.5% safety margin is added to detected crop edges. Invalid or
   over-aggressive bounds, leaving less than 20% of either source dimension,
   are rejected in favor of the original bounds.

**Core logic / pseudocode:**
```
validate non-empty BGR uint8 input
gray = convert BGR image to grayscale
measure four edge medians, central percentiles, and broad interior median
if light-page evidence and a dark edge transition:
  mask = Otsu threshold with light content as foreground
elif dark-page evidence, a light edge transition, and dark-page safeguards:
  mask = inverse Otsu threshold with dark content as foreground
else:
  return original bounds

for horizontal and vertical projections:
  compute foreground fraction for every row or column
  scan inward from both ends for a consecutive run over the adaptive threshold
expand detected bounds by safety padding
if bounds are invalid or retain less than 20% of either dimension:
  return original bounds
return original BGR image sliced to the accepted bounds
```

**Why this approach:**
Independent edge medians allow one-, two-, three-, or four-sided borders to be
considered separately. Projection densities and consecutive-hit scans are
custom NumPy/Python logic; OpenCV is used for grayscale conversion, Gaussian
blur, and Otsu thresholding. Cropping the original image avoids modifying
document colors during analysis.

**Fallbacks and limitations:**
Uniform/low-contrast images, ambiguous substrate classification, weak
projection evidence, and implausible crop bounds safely return the full image.
These safeguards favor avoiding destructive crops, so genuine borders can be
left in place (false negatives). Global Otsu thresholding and one-dimensional
projections cannot reliably distinguish a scanner border from page content
touching an edge, strong gradients/shadows, or some large illustrations. The
fixed intensity thresholds may also be sensitive to lighting and paper tone.
The function expects an already deskewed image; skew correction is owned by
Person 3 and is not implemented here.
