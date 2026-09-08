"""
Three-stage VLM pipeline runner: localize -> classify -> count bends.

Single-page PDF input per drawing (page index 0).

PDF rendering / resizing / cropping utilities live here since they're
shared plumbing, not a "stage" -- each stage's own model/prompt/VLM-call
logic is fully self-contained in stage1_localize.py, stage2_classify.py,
and stage3_count_bends.py.

Usage:
    python pipeline.py --input path/to/drawing.pdf --outdir outputs
    python pipeline.py --input path/to/folder_of_pdfs --outdir outputs
"""

import argparse
import json
import os
import glob
import traceback

import fitz  # PyMuPDF
import cv2
import numpy as np

from stage1_localize import run_stage1, REGION_TYPES
from stage2_classify import run_stage2
from stage3_count_bends import run_stage3

# --- shared config -------------------------------------------------------

DPI = 300                  # fixed render DPI, for determinism + crop quality
MAX_SIZE = 2048             # longest side of the image sent to Stage 1
CROP_PADDING_FRAC = 0.03    # padding added around each cropped region
OUTPUT_DIR = "outputs"


# --- PDF / image utilities ------------------------------------------------

def render_pdf_page(pdf_path, page_index=0, dpi=DPI):
    """Render a single PDF page to a full-resolution BGR numpy array.

    Returns (image_bgr, width, height) at the fixed render DPI -- this is
    the resolution all bounding boxes get mapped onto for crops.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_index >= len(doc):
            raise ValueError(
                f"{pdf_path} has only {len(doc)} page(s); "
                f"requested page index {page_index}"
            )
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi, alpha=False)

        image = np.frombuffer(pix.samples, dtype=np.uint8)
        image = image.reshape(pix.height, pix.width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, pix.width, pix.height
    finally:
        doc.close()


def resize_for_vlm(image, max_size=MAX_SIZE):
    """Downscale so the longest side is <= max_size (aspect preserved).

    Only used for the image sent to the Stage 1 VLM call -- never for
    cropping. Stage 1 boxes come back normalized to 0-1000, which is
    invariant to this resize, so no scale factor needs to be tracked.
    """
    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    if scale >= 1:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_jpeg(image, quality=95):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode image to JPEG")
    return buffer.tobytes()


def crop_normalized(image, bbox_norm, padding_frac=0.0):
    """Crop a region out of `image` given a bbox normalized to 0-1000.

    Because the image sent to the VLM is only ever a downscaled version of
    `image` (same aspect ratio, never pre-cropped), the same normalized box
    maps directly onto `image`'s own pixel dimensions.
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox_norm

    px0 = x0 / 1000.0 * width
    py0 = y0 / 1000.0 * height
    px1 = x1 / 1000.0 * width
    py1 = y1 / 1000.0 * height

    pad_x = (px1 - px0) * padding_frac
    pad_y = (py1 - py0) * padding_frac
    px0 -= pad_x
    py0 -= pad_y
    px1 += pad_x
    py1 += pad_y

    px0 = max(0, int(round(px0)))
    py0 = max(0, int(round(py0)))
    px1 = min(width, int(round(px1)))
    py1 = min(height, int(round(py1)))

    if px1 <= px0 or py1 <= py0:
        raise ValueError(f"Degenerate crop box after clipping: {bbox_norm}")

    return image[py0:py1, px0:px1]


# --- pipeline --------------------------------------------------------------

def _empty_output(drawing_id):
    return {
        "drawing_id": drawing_id,
        "regions": [],
        "class": None,
        "class_confidence": None,
        "class_evidence": None,
        "num_bends": None,
        "bend_confidence": None,
        "bend_evidence": None,
        "flags": [],
    }


def _write_output(output, drawing_outdir):
    os.makedirs(drawing_outdir, exist_ok=True)
    out_path = os.path.join(drawing_outdir, "result.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


def process_pdf(pdf_path, outdir):
    drawing_id = os.path.splitext(os.path.basename(pdf_path))[0]
    drawing_outdir = os.path.join(outdir, drawing_id)
    crops_dir = os.path.join(drawing_outdir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    output = _empty_output(drawing_id)
    flags = output["flags"]

    # --- Render full-res page (page 0 -- single-page PDF input) ---
    full_image, width, height = render_pdf_page(pdf_path, page_index=0)

    # --- Stage 1: localize regions on a downscaled copy ---
    vlm_input_bytes = encode_jpeg(resize_for_vlm(full_image))

    try:
        regions = run_stage1(vlm_input_bytes)
    except RuntimeError as e:
        flags.append(f"stage1_failed: {e}")
        _write_output(output, drawing_outdir)
        return output

    if not regions:
        flags.append("stage1_no_regions_found")
        _write_output(output, drawing_outdir)
        return output

    # --- Crop each detected region from the FULL-RES image ---
    crop_bytes = {}
    output_regions = []
    for i, region in enumerate(regions):
        try:
            crop_img = crop_normalized(full_image, region["bbox_2d"], CROP_PADDING_FRAC)
        except ValueError as e:
            flags.append(f"crop_failed[{region['type']}#{i}]: {e}")
            continue

        crop_path = os.path.join(crops_dir, f"{region['type']}_{i}.jpg")
        cv2.imwrite(crop_path, crop_img)
        crop_bytes.setdefault(region["type"], []).append(encode_jpeg(crop_img))

        x0, y0, x1, y1 = region["bbox_2d"]
        px_bbox = [
            round(x0 / 1000.0 * width), round(y0 / 1000.0 * height),
            round(x1 / 1000.0 * width), round(y1 / 1000.0 * height),
        ]
        output_regions.append({
            "type": region["type"],
            "bbox": px_bbox,
            "conf": region["conf"],
        })

    output["regions"] = output_regions

    if "title_block" not in crop_bytes:
        flags.append("no_title_block_found")

    # --- Stage 2: classify (title_block + all view crops) ---
    view_types = ["orthographic_view", "isometric_view", "section_view", "flat_pattern"]
    stage2_inputs = []
    for t in ["title_block"] + view_types:
        stage2_inputs.extend(crop_bytes.get(t, []))

    if not stage2_inputs:
        flags.append("stage2_skipped_no_crops")
        _write_output(output, drawing_outdir)
        return output

    try:
        cls_result = run_stage2(stage2_inputs)
    except RuntimeError as e:
        flags.append(f"stage2_failed: {e}")
        _write_output(output, drawing_outdir)
        return output

    output["class"] = cls_result["class"]
    output["class_confidence"] = cls_result["conf"]
    output["class_evidence"] = cls_result["evidence"]

    # --- Stage 3: count bends (sheet parts only) ---
    if cls_result["class"] == "sheet":
        stage3_types = ["flat_pattern", "section_view", "orthographic_view", "isometric_view"]
        stage3_inputs = []
        for t in stage3_types:
            stage3_inputs.extend(crop_bytes.get(t, []))

        if not stage3_inputs:
            flags.append("stage3_skipped_no_crops")
        else:
            try:
                bend_result = run_stage3(stage3_inputs)
                output["num_bends"] = bend_result["num_bends"]
                output["bend_confidence"] = bend_result["conf"]
                output["bend_evidence"] = bend_result["evidence"]
            except RuntimeError as e:
                flags.append(f"stage3_failed: {e}")
    else:
        output["num_bends"] = "n/a"

    _write_output(output, drawing_outdir)
    return output


def main():
    parser = argparse.ArgumentParser(description="VLM manufacturing-drawing pipeline")
    parser.add_argument("--input", required=True,
                         help="Path to a single PDF, or a folder of PDFs (batch mode)")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                         help="Output directory (default: outputs)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        pdf_paths = sorted(glob.glob(os.path.join(args.input, "*.pdf")))
    else:
        pdf_paths = [args.input]

    if not pdf_paths:
        print(f"No PDFs found at {args.input}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path} ...")
        try:
            result = process_pdf(pdf_path, args.outdir)
            print(json.dumps(result, indent=2))
        except Exception:
            print(f"FAILED on {pdf_path}:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
