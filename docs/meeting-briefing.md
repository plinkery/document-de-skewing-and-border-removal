# Team Meeting Briefing — Document De-skewing Project
> Complete picture for today's meeting. Share this with your teammates.
<!-- AI AGENT: Read docs/project-guide.md first for technical details.
     This file is the human-friendly overview of the whole system. -->

---

## The Big Picture — How It All Works Together

```
Your photo/scan (e.g. a tilted, bordered document)
        │
        ▼
┌─────────────────────┐
│   preprocessing.py  │  Person 1
│                     │  Colour image → Black/White binary image
│  "Find the pixels"  │  (Text = black, background = white, or vice versa)
└─────────┬───────────┘
          │  binary image (H×W, only 0s and 255s)
          ▼
┌─────────────────────┐
│    morphology.py    │  Person 2  ← COMPLETE
│                     │  Noisy binary → Clean binary
│  "Fix the noise"    │  Removes specks, fills holes in text
└─────────┬───────────┘
          │  cleaned binary image (same shape, still only 0s and 255s)
          │
   ┌──────┴──────────────────────────────────┐
   │ also receives the original colour image │
   ▼                                         │
┌─────────────────────┐                      │
│      deskew.py      │  Person 3     ◄──────┘
│                     │  Detects the tilt angle, rotates image straight
│  "Straighten it"    │
└─────────┬───────────┘
          │  (straightened colour image, angle in degrees)
          ▼
┌─────────────────────┐
│  border_removal.py  │  Person 4
│                     │  Finds the document edges, crops out surrounding border
│  "Crop the border"  │
└─────────┬───────────┘
          │  final clean image
          ▼
    data/outputs/result.jpg
```

**main.py** orchestrates all four steps. It calls each function in order,
passes outputs to the next step, and validates that everyone returned the
right type of data.

---

## What "The Contract" Means

Before anyone wrote a single line of code, the team agreed on the **exact
input and output of every function**.  This is called the contract.

```python
# Person 1
preprocess_image(img)              → binary_img
#  img:        colour image, shape (H, W, 3), values 0-255
#  binary_img: binary image, shape (H, W),    values ONLY 0 or 255

# Person 2  (complete)
apply_morphology(binary_img)       → cleaned_binary_img
#  Same types in and out

# Person 3  ← CRITICAL: return ORDER is (image, angle) not (angle, image)
deskew_image(raw_img, binary_img)  → (deskewed_img, angle)
#  deskewed_img: colour image (H, W, 3)
#  angle:        a float (e.g. 3.7)

# Person 4
remove_borders(img)                → cropped_img
#  colour image in → smaller colour image out
```

**Why this matters:** Because each person codes independently, the only
thing that connects their code is the function name and what it
accepts/returns.  If anyone changes the return type or parameter order
without telling the team, the whole pipeline breaks — even if their own
module works perfectly.

---

## Question 1: "Can I test my module without the others being done?"

**Yes — every person can test independently.** Here is how:

Each test file generates its own **synthetic input** that simulates the
previous module's output, without actually calling that module.

```python
# Example: Person 2's test creates a fake binary image
fake_binary = np.zeros((480, 640), dtype=np.uint8)
fake_binary[100:200, 50:600] = 255   # simulates a line of text
result = apply_morphology(fake_binary)   # tests morphology in isolation
```

For testing on **real sample images**, each test file also converts the
image internally (using a temporary Otsu threshold) before passing it to
the module.  Once the previous person merges their code, you swap in their
real function.  See `docs/output-management.md` for the exact commands.

---

## Question 2: "Will merging actually make everything work together?"

**Yes — if everyone follows the contract.** Here is why:

When Person 1 pushes their branch and it is merged into `main`, their
`preprocess_image()` function replaces the placeholder.  When you pull `main`,
you now have Person 1's real code.  Your `apply_morphology()` already expects
a binary image — exactly what Person 1 returns.  **Nothing else needs to change.**

The only way merging can break things is:
1. Someone changed a function name → `ImportError` (easy to fix, just rename)
2. Someone returned the wrong type → `main.py` gives a human-readable error
3. Person 3 returns `(angle, image)` instead of `(image, angle)` — silent crash

**main.py has built-in validation** that catches type errors immediately:

```
ValueError: [deskew] expected a 3-channel colour image (H, W, 3), got shape ()
Make sure you return the BGR colour image, not the binary one.
```

**Recommended merge order:**
```
1. Person 1 merges  →  basic grayscale pipeline works
2. Person 2 merges  →  noise removal works  (READY TO MERGE)
3. Person 3 merges  →  skew correction works
4. Person 4 merges  →  border removal works + runs full test
```

---

## Question 3: "What exactly should each person do after cloning?"

### Everyone (one time):
```bash
git clone https://github.com/YOUR-REPO-URL.git
cd ipcv-project
sudo apt install python3-opencv python3-numpy python3-matplotlib
python3 main.py data/samples/sample1.jpg   # verify the skeleton runs
```

### Person 1 (preprocessing):
```bash
git checkout -b feature/preprocessing
# Open: modules/preprocessing.py
# Delete the placeholder, write your grayscale + custom threshold code
# Run: python3 tests/test_preprocessing.py --real --debug
# Commit and push when tests pass
```

### Person 2 (morphology) — complete, ready to merge:
```bash
git checkout -b feature/morphology
# modules/morphology.py is ALREADY COMPLETE
# Run: python3 tests/test_morphology.py --real --debug
# Commit and push
```

### Person 3 (deskew):
```bash
git checkout -b feature/deskew
# Open: modules/deskew.py
# Implement Canny → HoughLinesP → custom angle filter → warpAffine
# CRITICAL: return (image, angle) — image first, angle second
# Run: python3 tests/test_deskew.py --real --debug
```

### Person 4 (border removal + integration):
```bash
git checkout -b feature/border-removal
# Open: modules/border_removal.py
# Implement row/column projection scanning → crop
# main.py skeleton is already done — do not rewrite it, just fill in remove_borders()
# After everyone merges: run python3 tests/test_pipeline.py
```

---

## Git Dos and Don'ts

| ✅ DO | ❌ DON'T |
|-------|---------|
| Work only on your own branch | Commit directly to `main` |
| `git pull origin main` at the start of every session | Work for 3 days without pulling |
| `git add modules/YOUR_file.py` — stage only your file | `git add .` — stages everything |
| Run your test before pushing | Push code that fails its own test |
| Create a Pull Request → teammate reviews → merge | Merge your own PR without a review |
| Write a meaningful commit message | `"update"` or `"fix"` |

For full details including conflict resolution: see `docs/git-guide.md`

---

## Debug Mode — Visual Inspection

Run with `--debug` to see what each module actually does:

```bash
python3 main.py data/samples/your_image.jpg --debug
```

Saves to `data/outputs/debug/`:
- `0_original.jpg` — raw input
- `1_binary.jpg` — after Person 1
- `2_cleaned_binary.jpg` — after Person 2
- `3_deskewed.jpg` — after Person 3
- `4_final.jpg` — after Person 4

This is how you demonstrate your module's effect for the assessment.

---

## For the 50/50 Assessment

Keep `docs/algorithm-notes.md` updated. The grader reads this.

| Person | What to document |
|--------|-----------------|
| 1 | Custom threshold-selection loop (histogram math) |
| 2 | `custom_erode`, `custom_dilate`, stride-trick vectorisation **(already written)** |
| 3 | Angle-histogram peak-finding logic |
| 4 | Row/column projection scanning algorithm |

Rule of thumb: If you wrote a loop, a formula, or a decision rule yourself
(not just called `cv2.something()`), document it in `algorithm-notes.md`.

---

## Documentation Index

All team documentation lives in `docs/`. Read in this order:

| File | What it covers |
|------|---------------|
| `docs/project-guide.md` | Full technical reference — contracts, pipeline, AI agent context |
| `docs/meeting-briefing.md` | This file — human-friendly overview |
| `docs/output-management.md` | How to manage test outputs, what to never push |
| `docs/git-guide.md` | Git from scratch — commits, conflicts, PAT setup |
| `docs/algorithm-notes.md` | Custom algorithm documentation (50/50 proof) |

