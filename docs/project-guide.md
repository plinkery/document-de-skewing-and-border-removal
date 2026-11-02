# Project Guide — Document De-skewing & Border Removal
<!-- AI AGENT CONTEXT: This file is the single source of truth for every team
     member AND for AI coding assistants.  Read it completely before writing
     any code.  Every design decision is explained here. -->

## 0. One-Paragraph Summary

This project takes a photograph or scan of a document that is tilted and has
unwanted borders around it, and produces a straightened, neatly cropped
output image.  It is implemented as a Python + OpenCV pipeline split across
**four independent modules** — one per team member.  Each module is a single
Python file with one public function.  The modules communicate only through
NumPy arrays with agreed-upon shapes and dtypes (called "the contract").

---

## 1. Repository Layout — What Every Folder Is For

```
ipcv-project/
│
├── modules/                  ← THE CORE: one file per person, never cross-edit
│   ├── __init__.py           ← empty; makes 'modules' a Python package
│   ├── preprocessing.py      ← Person 1: colour image  ──►  binary image
│   ├── morphology.py         ← Person 2: noisy binary  ──►  clean binary
│   ├── deskew.py             ← Person 3: detect angle, rotate colour image
│   └── border_removal.py     ← Person 4: detect document boundary, crop
│
├── tests/                    ← One test file per module. Run before merging.
│   ├── test_preprocessing.py ← Person 1 runs this
│   ├── test_morphology.py    ← Person 2 runs this
│   ├── test_deskew.py        ← Person 3 runs this
│   ├── test_border_removal.py← Person 4 runs this
│   └── test_pipeline.py      ← Full end-to-end: everyone runs this last
│
├── data/
│   ├── samples/              ← 2-3 real test images committed to the repo
│   │                            EVERYONE tests on the SAME images.
│   └── outputs/              ← GITIGNORED. Pipeline results go here.
│
├── docs/
│   ├── project-guide.md      ← THIS FILE: context for humans and AI agents
│   └── algorithm-notes.md    ← Custom algorithm documentation (50/50 proof)
│
├── utils/
│   └── __init__.py           ← Shared helpers (currently empty)
│
├── main.py                   ← Integration pipeline (Person 4 owns this)
├── requirements.txt          ← pip install -r requirements.txt
└── .gitignore                ← Keeps repo clean (no cache/output files)
```

### Why this structure?
- **One file per person** means Git merges almost never conflict.  Each person
  only touches their own file and (occasionally) their own test file.
- **The `data/samples/` images are committed** so everyone tests on identical
  input and results are comparable.
- **`data/outputs/` is gitignored** because generated image files are large
  and not source code.

---

## 2. The Four Modules — Inputs, Outputs, and Contracts

The pipeline is a chain.  The output of each step is the input to the next.
**These function signatures are frozen** — do not change them without a
team discussion, because changing them will break every other person's code.

```
raw_img (BGR colour, H×W×3, uint8)
    │
    ▼  modules/preprocessing.py
binary_img (greyscale binary, H×W, uint8, values={0,255})
    │
    ▼  modules/morphology.py
cleaned_binary_img (same type as above)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
(cleaned_binary_img)             (raw_img is also passed here)
    │
    ▼  modules/deskew.py   receives BOTH raw_img AND cleaned_binary_img
(deskewed_img, angle)   ← returns a TUPLE: colour image + float angle
    │
    ▼  modules/border_removal.py
final_img (BGR colour, h×w×3, uint8 where h≤H, w≤W)
```

### Exact function signatures

| Module | Function | Parameters | Return value |
|--------|----------|-----------|--------------|
| preprocessing.py | `preprocess_image` | `img: np.ndarray` (H,W,3 uint8) | `np.ndarray` (H,W) uint8 {0,255} |
| morphology.py | `apply_morphology` | `binary_img: np.ndarray` (H,W) uint8 {0,255} | `np.ndarray` (H,W) uint8 {0,255} |
| deskew.py | `deskew_image` | `raw_img, binary_img` | `tuple(np.ndarray, float)` — **(image, angle)** |
| border_removal.py | `remove_borders` | `img: np.ndarray` (H,W,3 uint8) | `np.ndarray` (h,w,3) uint8 |

> **CRITICAL for Person 3:** `deskew_image` must return `(image, angle)` — image
> first, angle second.  The wrong order will crash the pipeline silently.

---

## 3. What Each Person Must Implement

### Person 1 — preprocessing.py

**Receives:** Raw BGR colour image from `cv2.imread()`.

**Must return:** Binary image — only two values: 0 (black) and 255 (white).
Text/dark content should be 0; bright background should be 255, or vice versa
(either convention is fine as long as you are consistent).

**Topics to implement:**
- Grayscale conversion (`cv2.cvtColor`)
- Histogram analysis to choose a threshold (custom algorithm — write your own
  loop over the histogram, do not simply call `cv2.threshold` with Otsu)
- Apply the threshold to produce the binary image

**50/50 requirement:** The threshold-selection logic must be hand-coded.
The grayscale conversion and final thresholding call can use OpenCV.

---

### Person 2 — morphology.py  (**COMPLETED by Person 2**)

**Receives:** Binary image from Person 1.

**Must return:** Cleaned binary image — still only 0 and 255.

**What is implemented:**
- `custom_erode()` — manual erosion via numpy sliding window (no cv2)
- `custom_dilate()` — manual dilation via numpy sliding window (no cv2)
- `fast_erode()` / `fast_dilate()` — vectorised numpy versions using stride tricks
- `custom_opening()` — opening = erode then dilate (removes noise dots)
- `custom_closing()` — closing = dilate then erode (fills text holes)
- `apply_morphology()` — three-step pipeline: opening → closing → cv2 closing

**50/50 balance:** Steps 1 and 2 are pure numpy. Step 3 uses `cv2.morphologyEx`
for the wide-kernel pass (the kernel choice and rationale are our own work).

---

### Person 3 — deskew.py

**Receives:** BOTH the original raw BGR image AND the cleaned binary image.
- Use the binary image to detect lines (it is cleaner).
- Apply the rotation to the raw BGR image (to preserve colour in the output).

**Must return:** A tuple `(rotated_BGR_image, angle_float)`.

**Topics to implement:**
- Canny edge detection on the binary image
- Hough Transform (`cv2.HoughLinesP`) to detect line segments
- **Custom angle-filtering algorithm:** From all the detected line angles, pick
  the dominant skew angle (e.g., build a histogram of angles and find the peak,
  or use a median — this logic must be hand-coded, not a library call)
- Rotate the raw BGR image using `cv2.getRotationMatrix2D` + `cv2.warpAffine`

**50/50 requirement:** The angle-filtering/skew-estimation logic must be
hand-coded.  Hough and warpAffine can use OpenCV.

---

### Person 4 — border_removal.py + main.py

**Receives:** Deskewed BGR colour image from Person 3.

**Must return:** Cropped BGR image containing only the document area.

**Topics to implement:**
- Convert to grayscale internally for analysis
- **Custom row/column projection scanning:** For each row, sum the pixel
  intensities.  A row that is mostly white (high sum) is background.  Find
  where the sums transition from high (border) to low (document content).
  Do this for top, bottom, left, right edges.  This logic must be hand-coded.
- Crop with numpy slicing: `img[top:bottom, left:right]`

**Also owns:** `main.py` — the integration file.  Person 2 has set it up with
a complete working skeleton including contract-validation checks and a `--debug`
flag.  Person 4 adds integration testing and verifies the end-to-end pipeline.

**50/50 requirement:** The boundary-detection scanning must be hand-coded.
NumPy sum operations and the final crop slice are fine to use.

---

## 4. How to Run the Pipeline

### Install dependencies (once per machine)

```bash
# Option A — system packages (WSL / Ubuntu)
sudo apt install python3-opencv python3-numpy python3-matplotlib

# Option B — pip into a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the pipeline

```bash
# Basic run (uses data/samples/sample1.jpg by default)
python3 main.py

# Run on a specific image
python3 main.py data/samples/your_image.jpg

# Debug mode — saves intermediate images after each step
python3 main.py data/samples/your_image.jpg --debug
# Debug images are saved to: data/outputs/debug/
#   0_original.jpg, 1_binary.jpg, 2_cleaned_binary.jpg, 3_deskewed.jpg, 4_final.jpg
```

### Run your module's tests

```bash
python3 tests/test_preprocessing.py       # Person 1
python3 tests/test_morphology.py          # Person 2 (synthetic tests only)
python3 tests/test_morphology.py --real   # Person 2 (also tests on real images)
python3 tests/test_morphology.py --real --debug   # saves visual comparison images
python3 tests/test_deskew.py              # Person 3
python3 tests/test_border_removal.py      # Person 4
python3 tests/test_pipeline.py            # Full pipeline (everyone, last step)
```

---

## 5. Git Workflow Step-by-Step

### One-time setup (each person does this once)

```bash
git clone https://github.com/YOUR_USERNAME/ipcv-project.git
cd ipcv-project

# Install dependencies (see above)

# Verify the skeleton runs
python3 main.py data/samples/sample1.jpg
# You should see: "[1/4] Preprocessing...", "[2/4]...", etc. and a saved result.
```

### Create your feature branch (each person does this once after cloning)

```bash
git checkout -b feature/preprocessing     # Person 1
git checkout -b feature/morphology        # Person 2
git checkout -b feature/deskew            # Person 3
git checkout -b feature/border-removal    # Person 4
```

### Daily coding loop

```bash
git checkout feature/YOUR-BRANCH         # always work on your own branch
git pull origin main                     # get any updates from teammates

# ... write code in YOUR module file only (modules/your_file.py) ...

python3 tests/test_YOUR_MODULE.py        # run your test — must pass
git add modules/YOUR_FILE.py             # stage ONLY your file
git commit -m "Brief description of what you did"
git push origin feature/YOUR-BRANCH     # upload to GitHub
```

### Merging into main (when your module is complete)

```bash
# 1. Make sure your tests pass locally
python3 tests/test_YOUR_MODULE.py

# 2. Push your branch to GitHub
git push origin feature/YOUR-BRANCH

# 3. Go to github.com → your repo → click "Compare & pull request"
# 4. Write a short description of what you implemented
# 5. Tag one teammate to review it
# 6. Once approved, click "Merge pull request"

# 7. All teammates pull the updated main:
git checkout main
git pull origin main
```

---

## 6. How Testing Works Across Team Members

**Question: "Person 1 is done but Person 3 is not — can Person 2 still test?"**

Yes, because each test file generates its own synthetic binary inputs without
needing any other module.  Here is how independence is achieved:

```
test_morphology.py generates:
    fake = np.zeros((480, 640), dtype=np.uint8)
    fake[100:200, 100:300] = 255
    # This is a valid binary image — exactly what Person 1 produces.
    # Person 2 does not need Person 1's code to test.
```

For real-image testing (`--real` flag), `test_morphology.py` converts the
sample images to binary internally using OpenCV's Otsu threshold (a temporary
stand-in).  Once Person 1 merges, that stand-in can be swapped with a direct
call to `preprocess_image()`.

**The merge order matters only for final integration testing:**

```
Recommended merge order:
  1. Person 1 merges preprocessing.py
  2. Person 2 merges morphology.py
  3. Person 3 merges deskew.py
  4. Person 4 merges border_removal.py and runs test_pipeline.py
```

Everyone can code and test independently in parallel.  Only the final
integration test needs all modules merged.

---

## 7. The 50/50 Rule — What Counts as "Own Work"

The grader will check that approximately half the implementation is hand-coded
(not just OpenCV calls).  Document every custom algorithm in
`docs/algorithm-notes.md`.

| Module | Custom (own) code | OpenCV used |
|--------|------------------|-------------|
| preprocessing.py | Histogram analysis + threshold-selection loop | `cv2.cvtColor`, `cv2.threshold` |
| morphology.py | `custom_erode`, `custom_dilate`, `fast_erode`, `fast_dilate`, `custom_opening`, `custom_closing` | `cv2.morphologyEx` (one call, step 3 only) |
| deskew.py | Angle histogram + peak-finding + skew estimation | `cv2.Canny`, `cv2.HoughLinesP`, `cv2.warpAffine` |
| border_removal.py | Row/column projection scanning + boundary logic | `np.sum` (NumPy, not OpenCV) |

Fill in your section of `docs/algorithm-notes.md` when your module is done.

---

## 8. Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Returning a grayscale (not binary) image from preprocessing or morphology | Pipeline crash in `_validate_binary()` with a clear error | Apply `cv2.threshold` after any operation that might introduce intermediate values |
| Returning `(angle, image)` instead of `(image, angle)` from `deskew_image` | The pipeline receives an angle as an image — silent crash | Read the contract comment in `deskew.py` carefully |
| Editing someone else's module file | Git merge conflict that breaks the pipeline | Never touch another person's file |
| Committing to `main` directly | Overwrites everyone's work | Always work on your feature branch |
| Not pulling `main` before starting work | Your branch diverges; future merge gets complicated | Run `git pull origin main` at the start of every session |
| Pushing large output images | Slows down the repo for everyone | `data/outputs/` is gitignored — never `git add` from there |

---

## 9. Quick Reference Card for AI Agents

If you are an AI coding assistant helping one team member, read this section.

**Your task scope:**  Edit ONLY the file assigned to your team member.
Do not change `main.py`, other `modules/*.py` files, or test files for
other people.

**Contract you must not break:**

```python
# morphology.py
apply_morphology(binary_img: np.ndarray) -> np.ndarray
#   IN:  shape (H, W), dtype uint8, values in {0, 255}
#   OUT: shape (H, W), dtype uint8, values in {0, 255}

# deskew.py  — return ORDER matters
deskew_image(raw_img, binary_img) -> tuple[np.ndarray, float]
#   OUT: (image, angle) — image first!
```

**How to test your work without other modules:**

```python
import numpy as np
# Generate a fake binary image — valid input to morphology or deskew
fake_binary = np.zeros((480, 640), dtype=np.uint8)
fake_binary[100:200, 50:600] = 255   # simulate a text row

# Generate a fake colour image — valid input to border_removal or deskew
import cv2
fake_colour = cv2.imread('data/samples/sample1.jpg')
# or: fake_colour = np.random.randint(0, 255, (480,640,3), dtype=np.uint8)
```

**Run tests:**

```bash
python3 tests/test_morphology.py --real --debug
python3 main.py data/samples/sample1.jpg --debug
```

**Do not install packages not in `requirements.txt`.**
The project uses only `opencv-python`, `numpy`, and `matplotlib`.
