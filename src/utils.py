import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, square
from skimage.segmentation import find_boundaries
from pathlib import Path
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter, MultipleLocator


def read_image(path: str) -> np.ndarray:
    """Read image in grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    return img


def binarize_image(img: np.ndarray, thresh_offset: int = -20) -> np.ndarray:
    """Binarize image using Otsu threshold with optional offset.

    Bars are brighter (gray) than background (black), so threshold accordingly.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply offset to make threshold slightly stricter if desired
    th_val = max(0, _ + thresh_offset)
    _, bin_img = cv2.threshold(blur, th_val, 255, cv2.THRESH_BINARY)
    return bin_img


def detect_rows(bin_img: np.ndarray, row_count: int = 3) -> List[Tuple[int, int]]:
    """Detect horizontal rows of bars.

    Returns list of (start_y, end_y) for each detected row sorted top to bottom.
    Uses horizontal projection of binary image.
    """
    proj = bin_img.mean(axis=1)  # average across x
    thresh = proj.max() * 0.3
    inside = proj > thresh
    rows = []
    in_run = False
    start = 0
    for i, val in enumerate(inside):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            in_run = False
            rows.append((start, i))
    if in_run:
        rows.append((start, len(inside) - 1))

    # Merge small gaps
    merged = []
    for s, e in rows:
        if not merged:
            merged.append([s, e])
        else:
            prev_s, prev_e = merged[-1]
            if s - prev_e < 5:  # small gap
                merged[-1][1] = e
            else:
                merged.append([s, e])
    rows = [(s, e) for s, e in merged if e - s > 20]

    # Sort and limit to expected count
    rows.sort(key=lambda x: x[0])
    return rows[:row_count]


def get_middle_row_region(bin_img: np.ndarray) -> Tuple[int, int]:
    """Return (start_y, end_y) of the middle row between horizontal L1 bars."""
    rows = detect_rows(bin_img)
    if len(rows) < 3:
        # Fallback: choose largest row region
        rows.sort(key=lambda x: x[1] - x[0], reverse=True)
        return rows[0]
    # Middle one
    return rows[1]


def find_bars(bin_crop: np.ndarray, min_width: int = 5) -> List[Tuple[int, int, int, int]]:
    """Locate individual vertical bars in cropped binary image.

    Returns list of bounding boxes (min_row, min_col, max_row, max_col) sorted left-to-right.
    """
    # Remove small objects/noise
    cleaned = remove_small_objects(bin_crop > 0, min_size=100)
    # Close small gaps within bars
    closed = closing(cleaned, square(3))

    lbl = label(closed)
    props = regionprops(lbl)
    bars = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        width = maxc - minc
        height = maxr - minr
        if width < min_width:
            continue
        # Want tall narrow features
        if height > width * 2:
            bars.append((minr, minc, maxr, maxc))
    # Sort left to right
    bars.sort(key=lambda b: b[1])
    return bars


def measure_widths(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Measure bar width along its height using the binary mask.

    Parameters
    ----------
    mask : np.ndarray (bool or 0/255)
        Binary image with bar pixels == True/1.
    bbox : tuple
        (min_row, min_col, max_row, max_col) bounding the bar.

    Returns
    -------
    ys_rel : np.ndarray
        Height coordinate **relative to the top of the bar** (0 at top).
    widths : np.ndarray
        Measured width in pixels for each scan-line.
    """
    minr, minc, maxr, maxc = bbox
    bar_mask = mask[minr:maxr, minc:maxc] > 0
    ys_rel = []
    widths = []
    for y_local, row in enumerate(bar_mask):
        xs = np.where(row)[0]
        if xs.size < 2:
            continue
        width = xs.max() - xs.min() + 1
        ys_rel.append(y_local)
        widths.append(width)
    return np.array(ys_rel), np.array(widths)


def detect_scale_bar(img: np.ndarray, nm_value: int = 100) -> Tuple[float, Tuple[int, int, int, int] | None]:
    """Detect the horizontal scale bar (e.g., "100nm" line) at the bottom of an SEM image.

    Parameters
    ----------
    img : np.ndarray (grayscale 0-255)
    nm_value : int
        The physical length represented by the scale bar (nanometres).

    Returns
    -------
    nm_per_px : float
        Conversion factor from pixels to nanometres. `np.nan` if detection fails.
    bbox : (y, x, h, w) or None
        Bounding box of detected scale bar in original-image coordinates (top-left y, top-left x, height, width).
    """
    h, w = img.shape
    # Focus on bottom 30% of image where the scale bar & legend typically reside
    roi_y0 = int(0.7 * h)
    roi = img[roi_y0:]

    # Threshold to keep bright (white) pixels
    _, bin_roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove small noise (text specs)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(bin_roi, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    best_w = 0
    best_bbox = None  # (y, x, h, w) in global coords
    for i in range(1, num):
        x, y, comp_w, comp_h, area = stats[i]
        # Filter: reasonably thin & wide candidates
        if comp_h > 30 or comp_w < 20:
            continue
        aspect = comp_w / max(comp_h, 1)
        if aspect < 4:  # want elongated horizontal object
            continue
        # slightly favour objects with small height (thinner)
        score = comp_w / (comp_h + 1)
        if score > best_w:
            best_w = comp_w
            best_bbox = (roi_y0 + y, x, comp_h, comp_w)

    if best_w > 0:
        nm_per_px = nm_value / best_w
        return nm_per_px, best_bbox

    # ---------- Fallback: Hough line detection ----------
    edges = cv2.Canny(roi, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=5)
    if lines is not None:
        # Choose the longest nearly-horizontal line
        longest_len = 0
        best_line = None
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # check horizontal orientation
            if abs(y2 - y1) > 5:
                continue
            length = abs(x2 - x1)
            if length > longest_len:
                longest_len = length
                best_line = (x1, y1, x2, y2)
        if best_line is not None and longest_len > 20:
            x1, y1, x2, y2 = best_line
            nm_per_px = nm_value / longest_len
            bbox = (roi_y0 + min(y1, y2), min(x1, x2), abs(y2 - y1) + 1, longest_len)
            return nm_per_px, bbox

    # fallback: failure
    return float('nan'), None


def plot_results(img: np.ndarray, bins_mask: np.ndarray, bars: List[Tuple[int, int, int, int]],
                 bar_measurements: List[Tuple[np.ndarray, np.ndarray]],
                 out_path: Path,
                 units: str = 'px',
                 scale_bar_bbox: Tuple[int, int, int, int] | None = None):
    """Plot image with bar edges highlighted and width-vs-height curves.

    The left pane shows the SEM image with **edge traces** (not bounding boxes)
    of each bar coloured uniquely. The right pane plots width (x-axis) versus
    height (y-axis) for every bar, using a modulo 100 approach to show all bars
    on the same scale without horizontal shifts.
    """
    cmap = plt.get_cmap('rainbow')
    n_bars = len(bars)
    if n_bars == 0:
        return
    colors = [cmap(i / n_bars) for i in range(n_bars)]

    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1]})

    # ---------- IMAGE WITH EDGES ----------
    ax_img.imshow(img, cmap='gray')
    ax_img.set_axis_off()

    for idx, (bbox, color, meas) in enumerate(zip(bars, colors, bar_measurements)):
        minr, minc, maxr, maxc = bbox
        # Extract bar mask & find boundaries
        bar_mask = bins_mask[minr:maxr, minc:maxc] > 0
        edges = find_boundaries(bar_mask, mode='outer')
        ys, xs = np.where(edges)
        if ys.size == 0:
            continue
        ax_img.plot(xs + minc, ys + minr, '.', color=color, markersize=0.8)
        # label near the top
        ax_img.text(minc, max(minr - 10, 0), str(idx), color=color, fontsize=8, weight='bold')

    # Optional: draw scale bar bbox
    if scale_bar_bbox is not None:
        y_sb, x_sb, h_sb, w_sb = scale_bar_bbox
        rect_sb = plt.Rectangle((x_sb, y_sb), w_sb, h_sb, edgecolor='white', facecolor='none', linewidth=1.0, linestyle='--')
        ax_img.add_patch(rect_sb)
        ax_img.text(x_sb, y_sb - 8, f'{units}', color='white', fontsize=6)

    # ---------- WIDTH vs HEIGHT PLOT (with offset, labels modulo 100) ----------
    # Determine a horizontal shift for each successive bar so curves are separated
    max_width = max((w.max() if w.size else 0 for _, w in bar_measurements), default=10)
    # Add 30 % padding so neighbouring bars don't collide
    x_shift_unit = max(100, max_width * 1.3)

    for idx, ((ys_rel, widths), color) in enumerate(zip(bar_measurements, colors)):
        if widths.size == 0:
            continue
        x_vals = widths + idx * x_shift_unit
        ax_plot.plot(x_vals, ys_rel, '-', color=color, lw=1)
        ax_plot.scatter(x_vals, ys_rel, color=color, s=5)

        # Add text annotation with the mean actual width (nm or px) near curve top
        if len(ys_rel):
            top_y = ys_rel.min()
            avg_w = np.mean(widths)
            ax_plot.text(x_vals[0], top_y - 20, f"{avg_w:.1f}", color=color, fontsize=8)

    # Make x-axis labels repeat 0-100 by formatting tick labels modulo 100
    ax_plot.set_xlabel(f'Bar width ({units}) [modulo 100]')
    ax_plot.set_ylabel(f'Height within bar ({units})')
    ax_plot.set_title('Bar width vs height (each colour = bar)')
    ax_plot.grid(True, alpha=0.3)

    # If there are N bars we expect x range up to N*x_shift_unit + 100
    ax_plot.set_xlim(0, n_bars * x_shift_unit)

    # Tick locator every 20 units but labels shown modulo 100
    ax_plot.xaxis.set_major_locator(MultipleLocator(20))
    # Alternate between 0 and 100 for multiples of 100
    def alt_format(val, pos):
        if val % 100 != 0:  # Not a multiple of 100, no label
            return ""
        # Check if it's an even or odd multiple of 100
        if (val // 100) % 2 == 0:
            return "0"
        else:
            return "100"
    ax_plot.xaxis.set_major_formatter(FuncFormatter(alt_format))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig) 