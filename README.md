# CAT Grating Bar Width Analyzer

This tool automatically extracts the vertical CAT-grating bar widths from SEM images (zoomed or unzoomed) and visualizes the results.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python src/analyze_barwidth.py path/to/image.png --out outputs --smooth 5 --debug
```

* `path/to/image.png` – path to either a **zoomed** or **unzoomed** SEM image.
* `--out` – directory where the annotated figure will be saved (created automatically).
* `--smooth` – moving-average window size applied to every width-vs-height curve. `1` (default) means *no* smoothing.
* `--debug` – print extra information during processing.

The output file `<image>_analysis.png` contains:

1. **Left:** the original SEM image with each detected bar highlighted with a unique colour. The bar index is also printed above it.
2. **Right:** a plot of the bar width (y-axis) versus bar index (x-axis). Each bar's curve appears as a coloured vertical line corresponding to the colour used for the bar in the image.

## How it works (high-level)

1. **Binarisation** – adaptive thresholding (Otsu) converts the image to black/white.
2. **Row selection** – for unzoomed images, the middle bar row (between the two thick L1 horizontal bars) is isolated automatically.
3. **Bar detection** – connected-component analysis locates tall, narrow bright regions (bars).
4. **Width measurement** – for every scan line across the bar height, the pixel width of the bar is measured.
5. **Visualisation** – the image is annotated and bar-width curves are plotted using a consistent colour map.

Feel free to tweak thresholds or morphology parameters inside `src/utils.py` to best match your dataset. 