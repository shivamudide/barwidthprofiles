# CAT-Grating Bar-Width Analyzer

Automatically extracts vertical CAT-grating bar widths from top-down SEM images (both *zoomed* and *un-zoomed*) and visualises how each bar's width varies with height.

---
## 1. Quick start

### 1.1  Create the environment

```bash
# clone / cd into this repo …
conda env create -f environment.yml        # installs Python + all needed pkgs
conda activate barwidth                    # or whatever you named it
```

### 1.2  Run the script

```bash
python src/analyze_barwidth.py IMAGE_PATH [options]
```

Key options (all optional):

```
--out      outputs/            # directory to save annotated PNGs
--smooth   1                   # moving-average window (px) applied to every
                               #   width-vs-height curve; 1 = no smoothing
--debug                        # verbose progress messages
```

Example – analyse a zoomed SEM micrograph with 7-point smoothing:

```bash
python src/analyze_barwidth.py images/zoomed/PostKOH_Center_Frontside.jpg \
       --out outputs --smooth 7 --debug
```

The output `outputs/PostKOH_Center_Frontside_analysis.png` contains:

1. **Left pane** – original SEM image
   * each bar is traced in colour (edges, not rectangles)
   * bars are numbered, colours match the plot
   * detected scale-bar is highlighted (dashed white box)
2. **Right pane** – width-vs-height curves
   * x-axis = bar width (nm if scale-bar detected, otherwise px)
   * y-axis = height within bar, starting at bar bottom
   * curves are horizontally offset so they don't overlap but still share the same x-scale.

---
## 2. Intuitive algorithm overview

1. **Load → grayscale**  (`cv2.imread`).
2. **Global Otsu threshold + offset** gives a binary image where bar material is white.
3. **Row selection** – for *un-zoomed* images we locate the horizontal "L1" guide bars and keep only the middle CAT-bar row.
4. **Bar detection** – connected-component analysis (with morphological clean-ups) finds tall, narrow white regions; sorted left-to-right.
5. **Edge tracing** – `skimage.segmentation.find_boundaries` provides per-pixel left/right edges.
6. **Width measurement** – for every image row inside a bar we record `(height_index, width_in_pixels)`.
7. **Scale-bar detection** – search the bottom of the image for the thin "100 nm" white line (CC analysis, fallback Canny+Hough).  If found we convert pixels → nanometres.
8. **(optional) Smoothing** – user-selected moving average applied to each width profile.
9. **Plot + save** – colour-coded overlays and curves are rendered with Matplotlib.

---
## 3. Formal pipeline details

| Step | Technique / function | Important parameters |
|------|----------------------|----------------------|
| Binarisation | `cv2.threshold` (Otsu) | `thresh_offset = −20` (makes mask conservative) |
| Row localisation | Horizontal projection, gap-merging | `detect_rows()` in `utils.py` |
| Morphology | `remove_small_objects`, `closing` | `min_size = 100`, struct-el 3×3 |
| CC analysis | `skimage.measure.label`, `regionprops` | Bar accepted if `height > 2×width` |
| Edge extraction | `find_boundaries(mask, mode="outer")` | — |
| Width vs height | Scan-line loop, record min/max x | see `measure_widths()` |
| Smoothing | 1-D moving average | window `k = --smooth` |
| Scale-bar | CC heuristic → Hough fallback | assumes "100 nm" (edit in `detect_scale_bar`) |

---
## 4. Handling bar **yaw (rotation)**

The script assumes bars are roughly vertical; small yaw (≤ a few degrees) simply appears as left/right edges drifting slowly with height, which the width measurement naturally captures.  No explicit de-rotation is applied.  If you encounter severe tilt you can pre-rotate the image or add a rotation-correction step before measurement.

---
## 5. Customisation pointers

* **Parameters** – tweak thresholds and morphology constants in `src/utils.py`.
* **Scale value** – change the physical length of the reference bar (`nm_value` argument in `detect_scale_bar`).
* **Different smoothing** – swap the moving-average in `analyze_barwidth.py` with a Savitzky–Golay filter, spline, etc.

---
Happy analysing!  Open issues or PRs are welcome. 