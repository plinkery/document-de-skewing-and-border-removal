#!/usr/bin/env bash
# commit_all.sh — Run this once to commit all the new documentation and code
# Usage: bash commit_all.sh

set -e
cd /home/pink/ipcv-project

echo "=== Staging all changes ==="
git add .gitignore
git add modules/morphology.py
git add main.py
git add tests/test_morphology.py
git add docs/project-guide.md
git add docs/algorithm-notes.md
git add docs/output-management.md
git add docs/git-guide.md
git add docs/meeting-briefing.md
git add docs/README.md

echo ""
echo "=== Staged files ==="
git status

echo ""
echo "=== Committing ==="
git commit -m "Complete morphology module, upgrade main.py, add full team docs

- morphology.py: implement custom_erode, custom_dilate, fast_erode,
  fast_dilate, custom_opening, custom_closing (all pure NumPy)
- main.py: add --debug flag, per-step image saving, contract validation
- test_morphology.py: add --real and --debug flags, real-image testing
- .gitignore: tighten image rules, add all output subfolders
- docs/output-management.md: what to save, delete, never push
- docs/git-guide.md: complete Git beginner guide with conflict resolution
- docs/meeting-briefing.md: human-friendly pipeline overview
- docs/algorithm-notes.md: Person 2 section complete
- docs/README.md: documentation index"

echo ""
echo "=== Pushing to feature/morphology ==="
git push origin feature/morphology

echo ""
echo "=== Done! ==="
echo "Next: Go to GitHub → Compare & pull request → merge into main"

