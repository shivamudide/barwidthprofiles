import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import sys, os

sys.path.append(os.path.dirname(__file__))

from utils import (
    read_image,
    binarize_image,
    get_middle_row_region,
    find_bars,
    measure_widths,
    plot_results,
    detect_scale_bar,
)


def analyze_image(img_path: Path, out_dir: Path, debug: bool = False, smooth: int = 1):
    img = read_image(str(img_path))

    # Binarize image
    bin_img = binarize_image(img)

    # Detect if this is unzoomed (multiple rows) or zoomed (single row)
    rows = [get_middle_row_region(bin_img)]  # default assume one row: middle
    # Heuristic: if detected rows count >=3 we treat as unzoomed, else zoomed
    # This is handled inside get_middle_row_region. For analysis we'll just use middle row.
    y0, y1 = rows[0]
    crop_bin = bin_img[y0:y1]

    # Find bars in cropped binary image
    bars_bboxes = find_bars(crop_bin)
    # Offset bbox y coords by y0
    bars_bboxes = [(b[0] + y0, b[1], b[2] + y0, b[3]) for b in bars_bboxes]

    # Measure widths per bar (in pixels)
    bar_measurements_px: List[Tuple[np.ndarray, np.ndarray]] = []
    for bbox in bars_bboxes:
        ys_rel, widths = measure_widths(bin_img > 0, bbox)
        bar_measurements_px.append((ys_rel, widths))

    # Optional smoothing
    if smooth > 1:
        def smooth_arr(arr: np.ndarray, k: int):
            kernel = np.ones(k) / k
            return np.convolve(arr, kernel, mode='same')

        bar_measurements_px = [
            (ys, smooth_arr(w, smooth)) for ys, w in bar_measurements_px
        ]

    # --------------------------------------------------
    # Detect scale bar to convert pixels -> nm
    # --------------------------------------------------
    nm_per_px, sb_bbox = detect_scale_bar(img)
    if debug:
        if np.isfinite(nm_per_px):
            print(f"Detected scale bar: 1px = {nm_per_px:.3f} nm")
        else:
            print("Scale bar not detected – keeping pixel units.")

    if np.isfinite(nm_per_px):
        units = 'nm'
        bar_measurements = [(ys * nm_per_px, w * nm_per_px) for ys, w in bar_measurements_px]
    else:
        units = 'px'
        bar_measurements = bar_measurements_px

    # Plot results
    out_path = out_dir / f"{img_path.stem}_analysis.png"
    plot_results(img, bin_img, bars_bboxes, bar_measurements, out_path, units=units, scale_bar_bbox=sb_bbox)

    if debug:
        print(f"Processed {img_path.name}: {len(bars_bboxes)} bars found. Output saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CAT grating bar width from SEM images.")
    parser.add_argument("image", type=str, help="Path to input SEM image (zoomed or unzoomed).")
    parser.add_argument("--out", type=str, default="outputs", help="Directory to save outputs.")
    parser.add_argument("--smooth", type=int, default=1, help="Window size for moving-average smoothing of width profiles (>=1). Use 1 for no smoothing.")
    parser.add_argument("--debug", action="store_true", help="Print debug messages.")
    args = parser.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_image(img_path, out_dir, debug=args.debug, smooth=args.smooth) 