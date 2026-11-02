# Output Management Guide
<!-- AI AGENT: Read this before generating or saving any image files. -->

This document explains where to put test output images, how to view them,
when to delete them, and what must never be pushed to GitHub.

---

## The Golden Rule

> **`data/outputs/` is gitignored. Anything you save there stays on your
> machine only. It will never go to GitHub — no matter how many files you
> create there.**

You can create, view, and delete as many test images as you want in
`data/outputs/`. Git cannot see them.

---

## Folder Layout for Outputs

```
data/
├── samples/          ← INPUT images. Committed. Everyone shares these.
│   ├── doc1.jpg
│   └── doc2.jpg
│
└── outputs/          ← OUTPUT images. GITIGNORED. Local only.
    ├── result.jpg                    ← final pipeline output
    ├── debug/                        ← --debug flag output (all 4 steps)
    │   ├── 0_original.jpg
    │   ├── 1_binary.jpg
    │   ├── 2_cleaned_binary.jpg
    │   ├── 3_deskewed.jpg
    │   └── 4_final.jpg
    ├── morphology_debug/             ← Person 2's test outputs
    ├── preprocessing_debug/          ← Person 1's test outputs
    ├── deskew_debug/                 ← Person 3's test outputs
    └── border_debug/                 ← Person 4's test outputs
```

These folders are created automatically when you run the test commands.
You never need to create them manually.

---

## How Person 2 (Morphology) Tests on Real Images

Person 1 has not finished yet. You do not need to wait. Here is how to
test your morphology module on the original images right now:

### Step 1 — Run your tests

```bash
cd /home/pink/ipcv-project

# Quick test (synthetic images only, no real images, ~2 seconds):
python3 tests/test_morphology.py

# Full test on a FEW real images (pick 2-3, not all of them):
python3 tests/test_morphology.py --real --debug
```

The `--real --debug` mode:
- Reads images from `data/samples/`
- Converts each to binary internally (simulating Person 1's output)
- Runs your `apply_morphology()` on it
- Saves a **side-by-side comparison** (before | after) to `data/outputs/morphology_debug/`

### Step 2 — View the results

```bash
# Open a comparison image to visually inspect your morphology output:
eog data/outputs/morphology_debug/morphology_doc1.jpg    # on Ubuntu
# or:
display data/outputs/morphology_debug/morphology_doc1.jpg  # ImageMagick
# or just open the file in your file manager
```

The left half of the image = binary input (what Person 1 will send you).  
The right half = your cleaned binary output.

### Step 3 — You only need to check 2-3 images

You do not need to run on all images. Pick 2-3 representative ones:
- One clearly skewed document
- One with heavy noise
- One that is mostly clean

Once those look correct, you are done.

### Step 4 — Delete the outputs when done

Your outputs are temporary. Delete them after inspecting:

```bash
# Delete all morphology debug outputs (safe — they are gitignored anyway):
rm -rf data/outputs/morphology_debug/

# Or delete all outputs at once:
rm -rf data/outputs/
```

Git will not notice. The folder is ignored.

### Step 5 — After Person 1 merges

Once Person 1's `preprocessing.py` is merged into `main`, update your test:

Open `tests/test_morphology.py` and find these two lines:
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Replace them with:
```python
from modules.preprocessing import preprocess_image
bin_img = preprocess_image(img)
```

Then re-run `python3 tests/test_morphology.py --real --debug` to see your
morphology applied to Person 1's actual binary images.

---

## What Each Person Should Do With Their Outputs

### Person 1 — preprocessing.py

Test command:
```bash
python3 tests/test_preprocessing.py --real --debug
# Outputs saved to: data/outputs/preprocessing_debug/
```

What to check: Does the binary image cleanly separate document text from
the background? Text should be solid, background should be uniform.

Delete when done:
```bash
rm -rf data/outputs/preprocessing_debug/
```

---

### Person 2 — morphology.py (see full details above)

Test command:
```bash
python3 tests/test_morphology.py --real --debug
# Outputs saved to: data/outputs/morphology_debug/
```

What to check: Compared to the binary input (left half), does the output
(right half) have less noise and more solid text regions?

---

### Person 3 — deskew.py

Test command:
```bash
python3 tests/test_deskew.py --real --debug
# Outputs saved to: data/outputs/deskew_debug/
```

What to check: Is the text in the output image horizontal? Compare the
printed angle to the actual visible tilt.

---

### Person 4 — border_removal.py

Test command:
```bash
python3 tests/test_border_removal.py --real --debug
# Outputs saved to: data/outputs/border_debug/
```

What to check: Is the white/grey border cropped away? Is any document
content accidentally cut off?

---

## Running the Full Pipeline

Once all four modules are merged into `main`:

```bash
# Basic run:
python3 main.py data/samples/doc1.jpg

# Debug run — saves one image per step:
python3 main.py data/samples/doc1.jpg --debug
# Saves: data/outputs/debug/0_original.jpg through 4_final.jpg
```

---

## What Is Safe to Commit — Quick Reference

| File / folder | Commit? | Reason |
|---------------|---------|--------|
| `modules/YOUR_file.py` | ✅ YES | Your work |
| `tests/test_YOUR_module.py` | ✅ YES | Your tests |
| `docs/*.md` | ✅ YES | Documentation |
| `data/samples/*.jpg` | ✅ YES | Shared test inputs (already committed) |
| `data/outputs/` (any file) | ❌ NO | Generated, gitignored |
| `data/outputs/debug/` | ❌ NO | Generated, gitignored |
| `modules/<other_person>.py` | ❌ NO | Not your file |
| `main.py` | ❌ NO (unless you are Person 4) | Person 4 owns this |
| `.venv/` or `env/` | ❌ NO | Personal virtual environment |

---

## Verification — Confirm Nothing Bad Will Be Committed

Before you `git push`, always run this to see exactly what Git thinks
has changed:

```bash
git status
```

The output will list files under two sections:

```
Changes to be committed:    ← these WILL go to GitHub
Changes not staged:         ← these won't (yet)
Untracked files:            ← these won't (unless you git add them)
```

You should only ever see YOUR module file under "Changes to be committed".
If you see anything else — especially any `.jpg`, `.png`, or someone else's
`.py` file — do NOT push. Run `git restore --staged <filename>` to un-stage it.

```bash
# Example: you accidentally staged a result image
git restore --staged data/outputs/result.jpg

# Example: check that the outputs folder is truly invisible to Git
git status data/outputs/
# Should print: nothing to commit (or show nothing at all)
```

