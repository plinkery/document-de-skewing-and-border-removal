# Document De-Skewing and Border Removal

A Python/OpenCV pipeline that takes a tilted document image and automatically
straightens it and removes unwanted borders.

## Team Roles

| Person | Module | File |
|--------|--------|------|
| Person 1 | Preprocessing and Thresholding | modules/preprocessing.py |
| Person 2 | Morphological Operations | modules/morphology.py |
| Person 3 | Hough Transform and De-skewing | modules/deskew.py |
| Person 4 | Border Removal and Integration | modules/border_removal.py |

## Setup Instructions

    git clone https://github.com/YOUR_USERNAME/ipcv-project.git
    cd ipcv-project
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Running the Pipeline

    python3 main.py data/samples/sample1.jpg

Result is saved to data/outputs/result.jpg

## Project Structure

See docs/project-guide.md for a full explanation of every folder and file.
See docs/algorithm-notes.md for the 50/50 custom algorithm documentation.
