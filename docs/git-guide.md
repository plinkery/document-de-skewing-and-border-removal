# Git & GitHub Guide for Complete Beginners
<!-- AI AGENT: Read this file before running any git command for your team member.
     Follow these steps exactly. Do not skip the verification steps. -->

This guide covers everything a first-time Git user needs to know:
cloning, branching, committing, pushing, avoiding conflicts, and fixing
them when they happen.

---

## Part 1 — Core Concepts (Read This First)

### What is Git?
Git is a tool that tracks every change you make to your code files.
Think of it as a save-history for your project. You can go back to any
previous save at any time.

### What is GitHub?
GitHub is a website that hosts a copy of your project online so the whole
team can share it. Your computer has a **local** copy. GitHub has the
**remote** copy. You keep them in sync with `git push` and `git pull`.

### What is a branch?
A branch is your own private workspace inside the project.  
- `main` = the shared, stable version everyone agrees on  
- `feature/morphology` = Person 2's private workspace  

You do all your work on your branch. You never touch `main` directly.
When your work is ready, you merge your branch into `main` through a
Pull Request on GitHub.

```
main        ──────────────────────────────────────► (stable)
                \                        ▲
feature/morph    ──── your commits ──────┘  (Pull Request merge)
```

### What is a conflict?
A conflict happens when two people change the **same line** of the **same
file** at the same time, and Git cannot automatically decide which version
to keep. The solution is always: talk to each other and manually pick the
right version.

> **This project is designed to avoid conflicts** — each person edits only
> their own file. As long as you follow this rule, conflicts are nearly
> impossible.

---

## Part 2 — One-Time Setup (Do This Once After Cloning)

```bash
# 1. Clone the repository (get a copy on your machine)
git clone https://github.com/OWNER_USERNAME/ipcv-project.git
cd ipcv-project

# 2. Install the required Python packages
sudo apt install python3-opencv python3-numpy python3-matplotlib
# --- OR using pip ---
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Verify the skeleton runs
python3 main.py data/samples/sample1.jpg
# You should see: [1/4] Preprocessing... [2/4]... etc.

# 4. Create YOUR branch (replace morphology with your own name)
git checkout -b feature/morphology       # Person 2
git checkout -b feature/preprocessing    # Person 1
git checkout -b feature/deskew           # Person 3
git checkout -b feature/border-removal   # Person 4

# 5. Confirm you are on your branch
git branch
# The branch with * is your current branch. It should be YOUR branch, not main.
```

---

## Part 3 — The Daily Workflow (Every Time You Code)

Follow this exact sequence every coding session.

### Step 1 — Start of session: get the latest changes from teammates

```bash
# Make sure you are on YOUR branch
git checkout feature/YOUR-BRANCH-NAME

# Download any updates from GitHub (teammates may have merged things)
git pull origin main

# If there is nothing new, you'll see: "Already up to date."
# If there are updates, Git applies them automatically.
```

### Step 2 — Write your code

Open your module file (`modules/your_file.py`) and make changes.  
**Do not open or edit anyone else's file.**

### Step 3 — Check what you changed

```bash
git status
```

This shows:
- **Modified files** — files you changed (should only be yours)
- **Untracked files** — new files Git has not seen before

### Step 4 — Stage only your file

```bash
# Stage ONLY your module file (replace morphology with your file name)
git add modules/morphology.py

# If you also updated your test file:
git add tests/test_morphology.py

# DO NOT run: git add .   (this stages everything, including other people's files)
```

### Step 5 — Verify what is staged

```bash
git status
```

Under "Changes to be committed" you should see **only your file(s)**.  
If anything else appears (a `.jpg`, someone else's `.py`, `main.py`), un-stage it:

```bash
git restore --staged data/outputs/result.jpg   # example: un-stage an image
```

### Step 6 — Commit (save a snapshot)

```bash
git commit -m "Your description here"
```

Write a meaningful message — you will thank yourself later:

```bash
# Good messages:
git commit -m "Implement custom_erode with numpy sliding window"
git commit -m "Add vectorised fast_dilate using stride tricks"
git commit -m "Fix closing kernel size from 3 to 5 for better gap filling"

# Bad messages (avoid):
git commit -m "update"
git commit -m "fix"
git commit -m "asdf"
```

### Step 7 — Push to GitHub

```bash
git push origin feature/YOUR-BRANCH-NAME
```

First time pushing a new branch, Git may say:
```
fatal: The current branch feature/morphology has no upstream branch.
```
Fix it by running exactly what Git suggests:
```bash
git push --set-upstream origin feature/YOUR-BRANCH-NAME
```
After that first time, plain `git push` works.

---

## Part 4 — Creating a Pull Request (Merging Into Main)

Do this when your module is complete and all your tests pass.

```bash
# 1. Make sure all your tests pass first
python3 tests/test_YOUR_MODULE.py

# 2. Push your final changes
git push origin feature/YOUR-BRANCH-NAME
```

Then on **github.com**:

1. Open the repository page
2. You will see a yellow banner: **"Compare & pull request"** — click it
3. Write a title: `"Add morphology module — Person 2"`
4. Write a short description of what you implemented
5. Tag one teammate to review: click **Reviewers** on the right → add their username
6. Click **"Create pull request"**

Your teammate reviews it. If they say it looks good, they (or you) click
**"Merge pull request"**.

### After a teammate's PR is merged into main

Every other teammate must pull the updates:

```bash
git checkout feature/YOUR-OWN-BRANCH    # stay on your own branch
git pull origin main                    # pull the merged changes into your branch
```

---

## Part 5 — Avoiding Conflicts (Prevention)

Follow these three rules and you will almost never get a conflict.

### Rule 1: Only edit your own file
Each person has exactly one module file. Never open someone else's file.

```
Person 1 → modules/preprocessing.py   only
Person 2 → modules/morphology.py      only
Person 3 → modules/deskew.py          only
Person 4 → modules/border_removal.py  only (and main.py)
```

### Rule 2: Always pull before you start coding

```bash
git pull origin main    # do this at the start of EVERY session
```

This ensures your branch has the latest merged code before you add to it.
The longer you go without pulling, the more chances for things to diverge.

### Rule 3: Push small, frequent commits

Instead of coding for 5 hours and pushing one giant change, push after
every meaningful step. Smaller changes are easier to review and merge.

---

## Part 6 — Fixing a Conflict (If One Happens)

> Note: If everyone follows Rule 1 (only edit your own file), this
> section may never apply to you. But read it anyway so you are prepared.

### What a conflict looks like

When you run `git pull origin main` and a conflict occurs, Git will say:
```
CONFLICT (content): Merge conflict in modules/some_file.py
Automatic merge failed; fix conflicts and then commit the result.
```

Git will also mark the conflicting lines inside the file:

```python
<<<<<<< HEAD
# This is YOUR version of the line
result = custom_opening(binary_img, kernel_size=3)
=======
# This is the version from main (teammate's merged code)
result = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
>>>>>>> main
```

### How to fix it

**Step 1 — Open the file** and find all the `<<<<<<<` markers.

**Step 2 — Decide** which version is correct, or write a combined version.

**Step 3 — Delete the conflict markers** and keep only the code you want:

```python
# After fixing: just the code, no markers
result = custom_opening(binary_img, kernel_size=3)
```

**Step 4 — Stage and commit the fixed file:**

```bash
git add modules/your_file.py
git commit -m "Resolve merge conflict in morphology"
git push origin feature/YOUR-BRANCH-NAME
```

### What if I am completely confused?

Option A — Abort and start fresh from main:
```bash
git merge --abort         # cancel the conflicting merge
git pull origin main      # try again after talking to your teammate
```

Option B — Look at the current state of your file before your changes:
```bash
git diff HEAD modules/your_file.py    # shows what you changed vs last commit
```

Option C — Ask your teammate directly. Conflicts in this project are almost
always caused by two people editing the same file by mistake.

---

## Part 7 — Useful Commands Reference

```bash
git status                            # see what files changed
git branch                            # see all branches (* = current)
git log --oneline -10                 # see last 10 commits
git diff modules/morphology.py        # see changes in your file
git restore modules/morphology.py     # UNDO all unsaved changes to your file
git restore --staged modules/morph.py # un-stage a file (before commit)
git stash                             # temporarily save unfinished work
git stash pop                         # bring back stashed work
git checkout main                     # switch to main branch (read-only look)
git checkout feature/YOUR-BRANCH      # switch back to your branch
```

---

## Part 8 — Emergency Situations

### "I committed to main by accident"

```bash
# Undo the last commit, keep your changes as unsaved work
git reset --soft HEAD~1

# Now create your branch properly and commit there
git checkout -b feature/YOUR-BRANCH
git add modules/your_file.py
git commit -m "Your message"
git push origin feature/YOUR-BRANCH
```

### "I pushed something wrong"

Contact your team. Do NOT try to force-push or delete history on a shared
repository without everyone's agreement — it can overwrite teammates' work.

### "I deleted my file by accident"

```bash
# Restore it from the last commit
git restore modules/your_file.py
```

### "git push is asking for a username and password"

GitHub no longer accepts passwords for HTTPS pushes. Use a
**Personal Access Token (PAT)**:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token"
3. Name it, set an expiry, tick the **repo** checkbox
4. Copy the token
5. Use it as your **password** when git asks (username = your GitHub username)

To avoid being asked every time:
```bash
git config --global credential.helper store
# After one successful push, Git remembers your token permanently.
```

---

## Part 9 — Quick Checklist Before Every Push

```
[ ] I am on my own branch (git branch shows * feature/MY-BRANCH)
[ ] I only modified my own module file
[ ] My tests pass (python3 tests/test_MY_MODULE.py)
[ ] git status shows only my file(s) staged
[ ] No .jpg, .png, or output files are staged
[ ] My commit message is meaningful
```

If all checkboxes pass → `git push origin feature/YOUR-BRANCH`

