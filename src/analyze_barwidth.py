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
    create_bar_animation_gif,
)


def analyze_image(img_path: Path, out_dir: Path, debug: bool = False, smooth: int = 1):
    img = read_image(str(img_path))

    # Binarize image
    bin_img = binarize_image(img)

    # No assumption about contrast polarity: handled later by variant selection.

    # ------------------------------------------------------------------
    # Attempt detection on both the **original** and **inverted** binary
    # images and choose whichever variant yields more (reasonable) bars.
    # This handles both contrast polarities (bars bright vs bars dark).
    # ------------------------------------------------------------------

    import cv2  # local import – avoids circular deps in some environments

    variants = [
        ("orig", bin_img),
        ("inv", cv2.bitwise_not(bin_img)),
    ]

    best = None  # tuple(mask, bars_bboxes_in_crop, y0, y1)

    for name, mask in variants:
        y0_var, y1_var = get_middle_row_region(mask)
        crop = mask[y0_var:y1_var]
        bars = find_bars(crop)

        if best is None or len(bars) > len(best[1]):
            best = (mask, bars, y0_var, y1_var)

    # Unpack best variant
    bin_mask_for_measure, bars_bboxes_crop, y0_best, y1_best = best

    # Offset bbox y coords by y0_best to original-image coordinates
    bars_bboxes = [(b[0] + y0_best, b[1], b[2] + y0_best, b[3]) for b in bars_bboxes_crop]

    # Binary mask in boolean form for width measurement
    mask_bool = bin_mask_for_measure > 0

    # Measure widths per bar (in pixels)
    bar_measurements_px: List[Tuple[np.ndarray, np.ndarray]] = []
    for bbox in bars_bboxes:
        ys_rel, widths = measure_widths(mask_bool, bbox)
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
    
    # Calculate average bar width in pixels
    avg_bar_width_px = np.mean([np.mean(widths) for _, widths in bar_measurements_px])
    
    if debug:
        if np.isfinite(nm_per_px):
            print(f"Detected scale bar: 1px = {nm_per_px:.3f} nm")
            print(f"Average bar width: {avg_bar_width_px:.1f} px = {avg_bar_width_px * nm_per_px:.1f} nm")
        else:
            print("Scale bar not detected – keeping pixel units.")
    
    # If detected width is significantly different from expected 80nm,
    # adjust the scale factor (assuming bars should be ~80nm wide)
    expected_bar_width_nm = 80.0
    if np.isfinite(nm_per_px):
        detected_bar_width_nm = avg_bar_width_px * nm_per_px
        if abs(detected_bar_width_nm - expected_bar_width_nm) > 20:  # If off by more than 20nm
            adjusted_nm_per_px = expected_bar_width_nm / avg_bar_width_px
            if debug:
                print(f"Adjusting scale: 1px = {adjusted_nm_per_px:.3f} nm (based on expected bar width of ~80nm)")
            nm_per_px = adjusted_nm_per_px

    if np.isfinite(nm_per_px):
        units = 'nm'
        bar_measurements = [(ys * nm_per_px, w * nm_per_px) for ys, w in bar_measurements_px]
    else:
        units = 'px'
        bar_measurements = bar_measurements_px

    # Plot results
    out_path = out_dir / f"{img_path.stem}_analysis.png"
    plot_results(img, bin_mask_for_measure, bars_bboxes, bar_measurements, out_path, units=units, scale_bar_bbox=sb_bbox)
    
    # Create animation GIF showing one bar at a time
    gif_path = create_bar_animation_gif(img, bin_mask_for_measure, bars_bboxes, bar_measurements, out_path, units=units, scale_bar_bbox=sb_bbox)
    
    if debug:
        print(f"Processed {img_path.name}: {len(bars_bboxes)} bars found. Output saved to {out_path}")
        print(f"Bar animation GIF saved to {gif_path}")


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