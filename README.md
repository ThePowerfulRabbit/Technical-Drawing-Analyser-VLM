# Technical Drawing Analyser VLM

A VLM-based system for analyzing technical engineering drawings using Qwen3-VL.

The system takes a technical drawing PDF and processes it through three stages:

1. **Localization**
   - Converts the PDF page into an image.
   - Uses Qwen3-VL to detect and localize different regions of the drawing.
   - Detects:
     - Orthographic views
     - Isometric views
     - Section views
     - Flat patterns
     - Title blocks
   - Crops the detected regions with a small additional margin.

2. **Classification**
   - Uses the cropped views from Stage 1.
   - Provides all views to the VLM together so they can be interpreted as different views of the same component.
   - Classifies the component as either:
     - `Sheet`
     - `Tube`
   - Stores the classification result and supporting evidence as JSON.

3. **Bend Counting**
   - Runs only when the component is classified as sheet metal.
   - Uses the cropped views from Stage 1.
   - Estimates the number of bends in the sheet-metal component.
   - Stores the result and supporting evidence as JSON.

## Pipeline

```text
Technical Drawing PDF
        │
        ▼
   Stage 1
   Localization
        │
        ▼
    View Crops
        │
        ▼
   Stage 2
  Classification
        │
        ├── Tube ──────────────► End
        │
        ▼
      Sheet
        │
        ▼
   Stage 3
  Bend Counting
        │
        ▼
     Results
