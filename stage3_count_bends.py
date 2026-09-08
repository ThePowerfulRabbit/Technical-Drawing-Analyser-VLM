"""
Stage 3 -- bend counting (sheet parts only).

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
You are counting bends on a SHEET METAL part, using the flat pattern,
side/section view, and isometric view images provided.

Definition: a bend is one fold line where the plane of the material changes
direction. If the thinnest folded profile (typically visible in a
side/section view) is made of N straight planar segments, the bend count is
N - 1. Cross-check this against fold lines visible in the flat pattern and
against the isometric view.

Do NOT count as bends:
- chamfers (e.g. "2x45 deg" corner cuts)
- edge fillets or rounded corners / radii
- holes or cutouts
- stepped outlines that are cuts, not folds
- radius callouts like "R2 4x" -- these describe a corner radius, not a bend

Return ONLY this JSON object, nothing else:

{
  "num_bends": <integer >= 0>,
  "conf": <0-1 float>,
  "evidence": "<short phrase, e.g. 'side profile = 3 segments -> 2 folds'>"
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
        f"Stage 3 ({MODEL}) did not return valid JSON after {max_retries + 1} "
        f"attempt(s). Last error: {last_error}. Last raw output:\n{last_raw}"
    )


# --- public entry point -------------------------------------------------

def run_stage3(crop_bytes_list):
    """crop_bytes_list: JPEG bytes for flat_pattern + section/orthographic +
    isometric crops. Returns {"num_bends", "conf", "evidence"}.
    """
    result = _call_vlm_json(crop_bytes_list)

    return {
        "num_bends": int(result.get("num_bends", 0)),
        "conf": float(result.get("conf", 0.0)),
        "evidence": result.get("evidence", ""),
    }
