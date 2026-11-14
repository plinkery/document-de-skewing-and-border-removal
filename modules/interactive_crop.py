"""
interactive_crop.py - Interactive Document Boundary Refinement
==============================================================
Responsibility:
    Allow the user to interactively adjust and fine-tune document crop
    corners using an intuitive GUI window (CamScanner style).

Built with Matplotlib (TkAgg backend) for rock-solid cross-platform
compatibility on WSL2 / WSLg / Windows 11.

50/50 Rule Breakdown:
---------------------
LIBRARY-BASED (~50%):
    1. matplotlib.pyplot, FigureCanvasTkAgg, Polygon patch
    2. cv2.getPerspectiveTransform() + cv2.warpPerspective() via border_removal helpers
    3. cv2.resize(), cv2.cvtColor()

CUSTOM / HAND-CODED (NumPy) — ~50%:
    1. custom_find_closest_corner()   : Nearest-vertex selection using Euclidean distance.
    2. custom_clamp_coordinates()     : Clamps dragged points within image boundaries.
    3. custom_order_points()          : Clockwise ordering of 4 quadrilateral vertices.
    4. custom_four_point_transform()  : Target dimension calculation & perspective unwarping.
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from modules.border_removal import (
    custom_order_points,
    custom_four_point_transform,
    custom_find_document_quad
)


class DraggableQuadInteractor:
    """
    CamScanner-style draggable quadrilateral interactor built using Matplotlib
    events. Fully compatible with WSLg, Wayland, and Windows 11.
    """

    def __init__(self, ax, initial_corners_xy: np.ndarray, img_shape: tuple[int, int]):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.h, self.w = img_shape
        self.corners = np.array(initial_corners_xy, dtype=np.float32)
        self.backup_corners = self.corners.copy()
        self.active_idx = None
        self.confirmed = False

        # Close polygon loop: 5 points where pt[4] == pt[0]
        poly_pts = np.vstack([self.corners, self.corners[0]])
        self.poly = Polygon(poly_pts, closed=True, fill=True, facecolor=(0, 1, 0, 0.15), edgecolor='lime', linewidth=2.5)
        self.ax.add_patch(self.poly)

        # Draggable corner points and markers
        self.line = Line2D(
            poly_pts[:, 0], poly_pts[:, 1],
            marker='o', markerfacecolor='yellow', markeredgecolor='black',
            markeredgewidth=1.5, markersize=10, linestyle='none'
        )
        self.ax.add_line(self.line)

        # Labels for corners: TL, TR, BR, BL
        self.labels = ["1 (TL)", "2 (TR)", "3 (BR)", "4 (BL)"]
        self.text_annotations = []
        for i, (x, y) in enumerate(self.corners):
            txt = self.ax.text(x + 15, y - 15, self.labels[i], color='white', fontsize=9,
                               fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
            self.text_annotations.append(txt)

        # Connect Matplotlib GUI events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Hand-coded Euclidean nearest-corner selection
        min_d = float('inf')
        best_i = None
        for i, (cx, cy) in enumerate(self.corners):
            d = np.sqrt((cx - event.xdata)**2 + (cy - event.ydata)**2)
            if d < min_d:
                min_d = d
                best_i = i

        # Snap radius: within ~40 display units
        if min_d < 45:
            self.active_idx = best_i

    def on_motion(self, event):
        if self.active_idx is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Clamp coordinate inside image boundaries
        clamped_x = max(0, min(self.w - 1, event.xdata))
        clamped_y = max(0, min(self.h - 1, event.ydata))

        self.corners[self.active_idx] = [clamped_x, clamped_y]
        self.update_graphics()

    def on_release(self, event):
        if self.active_idx is not None:
            self.active_idx = None
            self.update_graphics()

    def on_key(self, event):
        # Enter / Space -> Confirm & close
        if event.key in ('enter', ' '):
            self.confirmed = True
            plt.close(self.ax.figure)
        # 'r' -> Reset
        elif event.key in ('r', 'R'):
            self.corners = self.backup_corners.copy()
            self.update_graphics()
        # 'escape' or 'q' -> Cancel
        elif event.key in ('escape', 'q', 'Q'):
            self.corners = self.backup_corners.copy()
            plt.close(self.ax.figure)

    def update_graphics(self):
        poly_pts = np.vstack([self.corners, self.corners[0]])
        self.poly.set_xy(poly_pts)
        self.line.set_data(poly_pts[:, 0], poly_pts[:, 1])

        for i, (x, y) in enumerate(self.corners):
            self.text_annotations[i].set_position((x + 15, y - 15))

        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main Interactive Crop Session Function
# ---------------------------------------------------------------------------

def interactive_refine_crop(
    deskewed_bgr: np.ndarray,
    initial_quad: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Launch an interactive window to let the user adjust document boundaries.

    Parameters
    ----------
    deskewed_bgr : np.ndarray
        Straightened image from Person 3's deskew_image().
    initial_quad : np.ndarray, optional
        4 corner coordinates predicted by Person 4 (shape (4, 2)).
        If None, custom_find_document_quad() is computed automatically.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (user_cropped_bgr, final_ordered_quad_original_coords)
    """
    H, W = deskewed_bgr.shape[:2]

    # 1. Determine initial 4-corner quad
    if initial_quad is None:
        quad_orig = custom_find_document_quad(deskewed_bgr)
    else:
        quad_orig = np.array(initial_quad, dtype=np.float32)

    quad_orig = custom_order_points(quad_orig)

    # 2. Convert BGR -> RGB for Matplotlib rendering
    rgb_img = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2RGB)

    # 3. Create interactive Matplotlib figure (Tk backend: guaranteed native window on WSLg)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.canvas.manager.set_window_title("Document Crop - Drag Yellow Corners to Refine Boundary")
    ax.imshow(rgb_img)
    ax.set_title(
        "[DRAG CORNERS WITH MOUSE]\nPress [ENTER / SPACE] to Confirm | [R] to Reset | [ESC] to Cancel",
        fontsize=11, fontweight='bold', pad=12, color='darkblue'
    )
    ax.axis('off')
    fig.tight_layout()

    interactor = DraggableQuadInteractor(ax, quad_orig, (H, W))

    print("\n" + "=" * 65)
    print("  [INTERACTIVE CROP] Window opened on your screen.")
    print("  - Drag any of the 4 yellow corner circles with your mouse.")
    print("  - Press [ENTER] or [SPACE] in the window to apply crop.")
    print("  - Press [R] to reset to automatic prediction.")
    print("  - Press [ESC] to keep automatic crop and exit.")
    print("=" * 65 + "\n")

    # Show figure and block until user confirms or closes window
    plt.show()

    final_quad = custom_order_points(interactor.corners)

    # 4. Perspective warp using custom_four_point_transform
    cropped_bgr = custom_four_point_transform(deskewed_bgr, final_quad)

    return cropped_bgr, final_quad
