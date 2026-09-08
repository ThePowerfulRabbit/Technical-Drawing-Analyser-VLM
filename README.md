# VLM Manufacturing-Drawing Pipeline

Three-stage VLM pipeline -- **localize -> classify (sheet/tube) -> count bends**
-- for single-page manufacturing PDF drawings.

## Layout (4 files)

```
stage1_localize.py     # Stage 1, self-contained: model, prompt, decoding
                        # settings, Ollama call, JSON parsing, run_stage1()
stage2_classify.py     # Stage 2, same pattern, run_stage2()
stage3_count_bends.py  # Stage 3, same pattern, run_stage3()
pipeline.py            # PDF render/resize/crop utilities + CLI + batch mode,
                        # imports and chains the three stages
```

Each stage file is fully self-contained -- its own `MODEL`, `PROMPT`,
decoding params (`temperature`, `seed`), and Ollama call/retry logic -- so
you can open, tune, or swap out one stage without touching the others.
The Ollama-call boilerplate is duplicated across the three stage files by
design, to keep each one readable in isolation.

## Pipeline

1. **Localize** (`stage1_localize.py`) -- a grounding VLM finds bounding
   boxes for `flat_pattern`, `orthographic_view`, `isometric_view`,
   `section_view`, `title_block` on the full page. Boxes are requested
   **normalized to 0-1000** (how Qwen-VL-family models are trained to emit
   boxes) and mapped to pixel space in `pipeline.py`.
2. **Classify** (`stage2_classify.py`) -- title block + view crops ->
   `sheet` or `tube`, weighing text cues (Abwicklung/Blechdicke,
   PLATE/BRACKET/BLECH vs. TUBE/ROHR/PROFIL) most heavily, geometry
   (thin uniform profile vs. hollow constant section) to break ties.
3. **Count bends** (`stage3_count_bends.py`, sheet parts only) -- flat
   pattern + section/orthographic + isometric crops -> bend count, defined
   as (straight segments in the thinnest folded profile - 1), explicitly
   excluding chamfers, fillets, holes, and stepped cuts.

## Determinism

- Fixed render DPI (`pipeline.DPI = 300`).
- Fixed decoding per stage: `temperature=0`, `seed=42`.
- Crops are cut from the **full-resolution render**, not the downscaled
  image sent to Stage 1. Because Stage-1 boxes are normalized (0-1000),
  they map directly onto the full-res image regardless of how much it was
  downscaled for the VLM call.

## Usage

Single drawing (page 0 is used -- single-page PDF input):
```bash
python pipeline.py --input Input_pdfs/001_bracket.pdf --outdir outputs
```

Batch mode (folder of single-page PDFs):
```bash
python pipeline.py --input Input_pdfs/ --outdir outputs
```

Each drawing produces:
```
outputs/<drawing_id>/result.json
outputs/<drawing_id>/crops/<region_type>_<i>.jpg   # one crop per detected region
```

## Output schema

```json
{
  "drawing_id": "001_bracket",
  "regions": [{"type": "flat_pattern", "bbox": [x0,y0,x1,y1], "conf": 0.94}],
  "class": "sheet",
  "class_confidence": 0.98,
  "class_evidence": "Abwicklung + Blechdicke 3mm; desc BRACKET",
  "num_bends": 2,
  "bend_confidence": 0.90,
  "bend_evidence": "side profile = 3 segments -> 2 folds",
  "flags": []
}
```
`bbox` values are pixel coordinates in the full-DPI render. `num_bends` is
`"n/a"` for tube parts, matching the reference-sample table.

`flags` records soft failures (`stage1_no_regions_found`,
`no_title_block_found`, `stage2_skipped_no_crops`, `crop_failed[...]`, etc.)
so a problem drawing still produces an inspectable JSON instead of a crash.

## Requirements

```bash
pip install pymupdf opencv-python numpy ollama
```
A local Ollama server with each stage's model pulled (default:
`qwen3-vl:8b` for all three -- edit the `MODEL` constant at the top of the
relevant stage file after running the sec.4 eval and picking a winner).

## Known limitations / not yet done

- Assumes single-page PDF input (page index 0) -- matches current task scope.
- One retry on malformed JSON per VLM call; no retry on connection errors
  beyond that.
- Model-choice rationale, inference-time/hardware analysis, and the sec.4
  eval results (accuracy vs. latency vs. cost) are separate deliverables
  not covered by this code.
