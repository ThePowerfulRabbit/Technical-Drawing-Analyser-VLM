"""
Stage 2 -- sheet vs. tube classification.

Self-contained: model choice, decoding settings, prompt, and the Ollama
call + JSON parsing all live in this one file.
"""

import json
import ollama

# --- config -----------------------------------------------------------

MODEL = "qwen3-vl:8b"
TEMPERATURE = 0
SEED = 42
NUM_CTX = 8192

# --- prompt (v1) --------------------------------------------------------

PROMPT = """
You are classifying a manufacturing drawing as either a SHEET METAL part or
a TUBE/PROFILE part, using the cropped title block and view images provided.

Evidence for SHEET:
- Title block or notes mention "Abwicklung" (developed/flat pattern) or
  "Blechdicke" (sheet thickness)
- Description contains PLATE, BRACKET, or BLECH
- A thin, uniform-thickness edge profile visible in a side/section view
  (constant wall thickness across a flat or bent plate, not a hollow tube)

Evidence for TUBE:
- A constant hollow cross-section combined with a long overall length
- A section view showing wall thickness around a hollow bore
- Description contains TUBE, ROHR, or PROFIL

Weigh the title block text (part description, material spec) most heavily;
use the view geometry to corroborate or break ties.

Return ONLY this JSON object, nothing else:

{
  "class": "sheet" | "tube",
  "conf": <0-1 float>,
  "evidence": "<short phrase citing the specific cue(s) used>"
}
"""


# --- Ollama call + JSON parsing (with one retry on malformed output) ---

def _strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def _call_vlm_json(image_bytes_list, max_retries=1):
    last_raw, last_error = None, None
    for _ in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT, "images": image_bytes_list}],
                format="json",
                options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "seed": SEED},
            )
        except Exception as e:
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
        f"Stage 2 ({MODEL}) did not return valid JSON after {max_retries + 1} "
        f"attempt(s). Last error: {last_error}. Last raw output:\n{last_raw}"
    )


# --- public entry point -------------------------------------------------

def run_stage2(crop_bytes_list):
    """crop_bytes_list: JPEG bytes for the title_block crop plus view crops.
    Returns {"class", "conf", "evidence"}.
    """
    result = _call_vlm_json(crop_bytes_list)

    cls = result.get("class")
    if cls not in ("sheet", "tube"):
        cls = "sheet" if str(cls).lower().startswith("sheet") else "tube"

    return {
        "class": cls,
        "conf": float(result.get("conf", 0.0)),
        "evidence": result.get("evidence", ""),
    }
