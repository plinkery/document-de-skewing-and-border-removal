# Project Guide

## What Each Folder Is For

### modules/
The core of the project. Each file belongs to exactly one person.
Nobody ever edits another person's file.

    preprocessing.py   -> Person 1: colour image -> binary image
    morphology.py      -> Person 2: noisy binary  -> clean binary
    deskew.py          -> Person 3: detect angle, rotate image
    border_removal.py  -> Person 4: detect document boundary, crop
    __init__.py        -> empty file, required by Python

### data/
    samples/  -> 2-3 test images committed to the repo.
                 Everyone tests on the SAME images so results are comparable.
    outputs/  -> where the pipeline saves results. GITIGNORED.
                 Generated files are not source code.

### tests/
One test file per module. Run your test file before merging.
This catches bugs before they break the main branch for everyone else.

    test_preprocessing.py  -> Person 1 runs this
    test_morphology.py     -> Person 2 runs this
    test_deskew.py         -> Person 3 runs this
    test_border_removal.py -> Person 4 runs this
    test_pipeline.py       -> Person 4 runs this for full integration

### docs/
    project-guide.md    -> this file, explains the structure
    algorithm-notes.md  -> each person documents their custom algorithms here
                           Required for the 50/50 assessment proof

### utils/
Optional shared helpers. Currently empty.
Only add files here if multiple people need the same utility function.

### main.py
The integration file that calls all four modules in sequence.
Person 4 is the only one who edits this file.
All other team members should not touch main.py.

### requirements.txt
All Python packages the project needs, listed in one file.
After cloning, each person runs:
    pip install -r requirements.txt
This ensures everyone has the same packages.

### .gitignore
Tells Git which files to never track. Without this, everyone's
repo would be cluttered with Python cache files, editor settings,
and large generated output images.

---

## The Agreed Function Contracts

These are fixed. Do not change name, parameter order, or return type
without a team discussion.

    preprocess_image(img)              -> binary_img
    apply_morphology(binary_img)       -> cleaned_binary_img
    deskew_image(raw_img, binary_img)  -> (deskewed_img, angle)
    remove_borders(img)                -> cropped_img

The return type of deskew_image is a TUPLE: (image, angle) in that order.

---

## Git Workflow for Beginners

### One-time setup (each person does this once)

    git clone https://github.com/YOUR_USERNAME/ipcv-project.git
    cd ipcv-project
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 main.py data/samples/sample1.jpg   <- verify it runs

### Create your branch (each person does this once)

    git checkout -b feature/preprocessing      <- Person 1
    git checkout -b feature/morphology         <- Person 2
    git checkout -b feature/deskew             <- Person 3
    git checkout -b feature/border-removal     <- Person 4

### Daily work loop

    git checkout feature/YOUR-BRANCH           <- make sure you are on YOUR branch
    git pull origin main                       <- get latest changes first
    # ... write code in YOUR module file only ...
    python3 tests/test_YOUR_MODULE.py          <- run your test
    git add modules/YOUR_FILE.py               <- stage only your file
    git commit -m "Describe what you did"      <- save a snapshot
    git push origin feature/YOUR-BRANCH        <- upload to GitHub

### Merging into main

    1. Push your branch (see above)
    2. Go to GitHub.com, open the repo
    3. Click 'Compare & pull request' on your branch
    4. Write a short description of what you implemented
    5. Ask one teammate to review it
    6. Once approved, click 'Merge pull request'
    7. All teammates run: git checkout main && git pull origin main
