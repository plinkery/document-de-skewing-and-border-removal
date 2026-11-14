# Document De-Skewing and Border Removal Pipeline

An end-to-end computer vision pipeline implemented in Python with OpenCV and NumPy. The system ingests smartphone photographs or scans of tilted documents on cluttered backgrounds, automatically estimates and corrects skew, and crops out external background borders using contour detection and perspective transformation. It also features an interactive boundary refinement tool for complex documents.

---

## Team Roles & Modular Architecture

Each module is balanced with **~50% OpenCV library functions** and **~50% hand-coded NumPy/mathematical algorithms**:

| Person | Module | File | Core Responsibilities & 50/50 Balance |
| :--- | :--- | :--- | :--- |
| **Person 1** | Preprocessing & Thresholding | `modules/preprocessing.py` | • **Library (OpenCV)**: `cv2.cvtColor`, `cv2.GaussianBlur`<br>• **Hand-coded**: 256-bin histogram computation, dynamic range contrast stretching, Otsu's optimal threshold selection from first principles |
| **Person 2** | Morphological Operations & Interactive UI | `modules/morphology.py`<br>`modules/interactive_crop.py` | • **Library (OpenCV/Matplotlib)**: Closing with `cv2.morphologyEx`, Matplotlib GUI event polling<br>• **Hand-coded**: Sliding-window erosion & dilation (`custom_erode`, `custom_dilate`), nearest-vertex Euclidean distance tracker for draggable handles |
| **Person 3** | Hough Transform & De-skewing | `modules/deskew.py` | • **Library (OpenCV)**: `cv2.Canny`, `cv2.HoughLinesP`, `cv2.warpAffine`<br>• **Hand-coded**: Trigonometric line angle computation (`atan2`), angle modulo normalization, dominant angle voting histogram |
| **Person 4** | Boundary Isolation & Integration | `modules/border_removal.py`<br>`main.py` | • **Library (OpenCV)**: `cv2.findContours`, `cv2.convexHull`, `cv2.approxPolyDP`, `cv2.warpPerspective`<br>• **Hand-coded**: Clockwise quadrilateral corner ordering (`custom_order_points`), Pythagorean distance and dimension derivation, end-to-end pipeline integration |

---

## Setup Instructions

```bash
git clone https://github.com/plinkery/document-de-skewing-and-border-removal.git
cd ipcv-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Mass Batch Integration on the Entire Dataset (Saves to `data/outputs/comparisons/`)
To test all images in your dataset automatically and generate 4-panel comparison strips:
```bash
python3 tests/test_pipeline.py --count 100 --debug
```
- **Output comparison folder**: `data/outputs/comparisons/compare_<image_name>.png`
- Each comparison image contains a clean **4-part transition panel**:
  `[1. Grayscale] | [2. Binary (P1)] | [3. Cleaned (P2)] | [4. Deskewed + Border Removed (P3+P4)]`
- Also prints a summary table in the terminal displaying skew angles, output resolutions, and pass/fail status.

### 2. Single Image Test with Individual Stage Outputs (Saves to `data/outputs/debug/`)
To run on a specific image and inspect full-resolution files of each individual stage:
```bash
python3 main.py data/samples/original/11_1.jpg --debug
```
- **Replaces previous outputs in `data/outputs/debug/`** to keep the folder uncluttered:
  - `0_original.jpg` — Raw input image
  - `0_grayscale.jpg` — Single-channel grayscale image
  - `1_binary.jpg` — Preprocessed binary image (P1 Otsu)
  - `2_cleaned_binary.jpg` — Morphologically cleaned image (P2)
  - `3_deskewed.jpg` — Rotated/straightened image (P3)
  - `4_final_auto.jpg` — Automatically cropped document (P4)
  - `pipeline_comparison.png` — 4-panel comparison strip
  - `data/outputs/result.jpg` — Final output image

### 3. Interactive Mode (CamScanner-Style User Refinement)
To launch the interactive GUI where you can manually drag the 4 corner handles to adjust the crop:
```bash
python3 main.py data/samples/original/11_1.jpg --interactive --debug
```
*(Shorthand: `python3 main.py data/samples/original/11_1.jpg -i --debug`)*
- A window opens displaying the deskewed image with 4 draggable corner handles.
- **Controls**:
  - **Drag handles** with left mouse click to adjust corners.
  - Press **`ENTER`** or **`SPACE`** to confirm and apply the perspective crop.
  - Press **`R`** to reset corners to the automatic prediction.
  - Press **`ESC`** or **`Q`** to cancel manual editing and keep the automatic result.
- In interactive debug mode, a **5-panel comparison strip** is generated in `data/outputs/debug/pipeline_comparison.png`:
  `[1. Grayscale] | [2. Binary] | [3. Cleaned] | [4. Deskewed + Border Removed] | [5. Custom Crop]`

---

## Unit & Integration Tests

Run individual module unit tests:
```bash
python3 tests/test_preprocessing.py
python3 tests/test_morphology.py
python3 tests/test_deskew.py
python3 tests/test_border_removal.py
```
Run end-to-end integration test:
```bash
python3 tests/test_pipeline.py
```
