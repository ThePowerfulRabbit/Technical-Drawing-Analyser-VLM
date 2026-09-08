"""
Stage 1 -- region localization (grounding VLM).

Self-contained: model choice, decoding settings, prompt, and the Ollama
call + JSON parsing all live in this one file. `pipeline.py` just imports
and calls run_stage1().
"""

import json
import ollama

# --- config -----------------------------------------------------------

MODEL = "qwen3-vl:8b"
TEMPERATURE = 0          # deterministic
SEED = 42
NUM_CTX = 8192

REGION_TYPES = [
    "flat_pattern",
    "orthographic_view",
    "isometric_view",
    "section_view",
    "title_block",
]

# --- prompt (v1) --------------------------------------------------------

PROMPT = """
You are analyzing a manufacturing engineering drawing (a single-part
fabrication drawing: sheet metal or tube/profile).

Locate every instance of the following region types:

- flat_pattern       (unfolded / developed pattern, often labeled "Abwicklung")
- orthographic_view  (front/top/side projection views)
- isometric_view     (3D / isometric view)
- section_view       (cross-section view, often hatched)
- title_block        (title block / BOM table, usually bottom-right or bottom edge)

Do NOT report the following as regions:
- dimensions, dimension lines, extension lines
- centerlines
- arrows / leaders
- notes / general-notes text blocks that are not the title block
- the drawing sheet border / frame
- individual holes
- individual annotations or balloons

Return ONLY a JSON object of this exact shape, nothing else:

{
  "regions": [
    {"type": "<one of the types above>", "bbox_2d": [x0, y0, x1, y1], "conf": <0-1 float>}
  ]
}

bbox_2d MUST be normalized to a 0-1000 integer scale, NOT pixel coordinates:
  x0, x1 are horizontal positions where 0 = left edge, 1000 = right edge
  y0, y1 are vertical positions where 0 = top edge, 1000 = bottom edge
  x0 < x1 and y0 < y1

If a region type does not appear on the drawing, omit it. If a type appears
more than once (e.g. two orthographic views), include one entry per instance.
"""


# --- Ollama call + JSON parsing (with one retry on malformed output) ---

def _strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def _call_vlm_json(image_bytes, max_retries=1):
    last_raw, last_error = None, None
    for _ in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT, "images": [image_bytes]}],
                format="json",
                options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "seed": SEED},
            )
        except Exception as e:  # connection error, model not pulled, etc.
            last_error = e
            continue

        raw = response["message"]["content"]
        last_raw = raw
        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Stage 1 ({MODEL}) did not return valid JSON after {max_retries + 1} "
        f"attempt(s). Last error: {last_error}. Last raw output:\n{last_raw}"
    )


# --- public entry point -------------------------------------------------

def run_stage1(image_bytes):
    """image_bytes: JPEG-encoded full-page image (already resized for the VLM).

    Returns a list of {"type", "bbox_2d" (0-1000 normalized), "conf"} dicts,
    filtered to known region types and structurally validated.
    """
    result = _call_vlm_json(image_bytes)
    raw_regions = result.get("regions", [])

    regions = []
    for r in raw_regions:
        rtype = r.get("type")
        bbox = r.get("bbox_2d") or r.get("box_2d") or r.get("bbox")
        conf = r.get("conf", None)

        if rtype not in REGION_TYPES:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue

        x0, y0, x1, y1 = bbox
        if not (x0 < x1 and y0 < y1):
            continue

        regions.append({
            "type": rtype,
            "bbox_2d": [float(x0), float(y0), float(x1), float(y1)],
            "conf": float(conf) if conf is not None else None,
        })

    return regions
