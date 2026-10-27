# telescoop-ocr

Transcribes handwritten workshop canvases (scanned PDF) into one Markdown file per participant, question by question.

Each scanned page is registered onto the blank template, every answer zone is cropped, handwritten zones are transcribed with [LightOnOCR-3](https://huggingface.co/lightonai), and checkboxes and star ratings are read from ink density.

## Inputs

Put the two PDFs in `data/` (not versioned, they contain personal data):

- `data/Scan_atelier_cotech_elargi.pdf`: the scan, 24 A4 pages rotated 90°, two canvases per page
- `data/canvas cotech elargis.pdf`: the blank template, 2 landscape pages

Odd pages hold Canevas 2 (top) and Consommation (bottom), even pages hold Canevas 3 (top) and Canevas 1 (bottom). Pages 2k−1 and 2k belong to participant k. The blank questionnaire is in `questions.md`.

## Install

Requires [uv](https://docs.astral.sh/uv/). Pick the torch build with an extra:

```bash
uv sync --extra cpu     # CPU
uv sync --extra cu126   # NVIDIA GPU (CUDA 12.6 wheels, still support Volta / V100)
```

Then run the scripts with `uv run --no-sync <script>` so uv keeps the chosen torch build.

## Run

```bash
uv run --no-sync run.py                                          # all 12 participants: LightOnOCR-3-4B on GPU, 1B on CPU
uv run --no-sync run.py --model lightonai/LightOnOCR-3-1B        # pick another model
uv run --no-sync run.py --participants 1 2                       # subset
uv run --no-sync register.py work/pages/*.png                    # registration overlays in work/debug/
```

Outputs go to `output/participant_XX.md`. Words the model was unsure about are marked `[word?]`; proofread those against the scan. OCR results (text and per-word confidence) are cached per model in `work/cache/`, so reruns only transcribe new zones and changing `LOW_CONFIDENCE` in `run.py` only needs a rerun of the cached pipeline.

## Pipeline

| File | Role |
|---|---|
| `split.py` | Renders each page at 300 dpi and cuts it into upright top/bottom halves (`work/pages/`) |
| `zones.yaml` | Every question per canvas in template PDF points: `text` zones (with the printed `label` to strip), `stars` and `checkboxes` |
| `template.py` | Renders the template canvases and converts zones to pixels |
| `register.py` | ORB + RANSAC similarity alignment of a scan half onto the best matching template canvas |
| `marks.py` | Checkboxes and stars from the mean darkness inside each shape; `has_handwriting` skips empty text zones |
| `ocr.py` | LightOnOCR-3 transcription with French-only decoding |
| `run.py` | Full pipeline and Markdown writer |

## Model notes

- `LightOnOCR-3-1B` is labelled `mistral3` in its config; it is loaded with `LightOnOcrForConditionalGeneration` and a key mapping `language_model.model.` → `model.language_model.`. The 0.8B and 4B use `Qwen3_5ForConditionalGeneration` through `AutoModelForImageTextToText`.
- Dtype: float32 on CPU and on pre-Ampere GPUs (V100 has no native bf16; float32 4B needs ~16 GB VRAM), bf16 on Ampere and newer. On a laptop CPU float32 with 4 threads is ~2.7× faster than bf16 (~11 s per zone for the 1B).
- Decoding is masked to French characters. If at any step the model's top choice is a letter from another script, the zone is left empty.
- Markup characters (`<>#|` and backtick) are banned too: on crops with a checkbox or box borders the 4B otherwise switches to `<figure>` descriptions instead of transcribing.
- Model choice, compared on participant 1 on a V100S: 1B 0.9 s/zone, 0.8B 2.5 s/zone, 4B 2.1 s/zone. The 4B reads best (names, "Végéstrate", "déconnexion") but still misreads rare words.
- Word confidence is the lowest probability of the word's tokens (after the French mask). `LOW_CONFIDENCE = 0.55` was set on participant 1 with the 4B: it flags 11 of 14 wrong words and about 7 % of correct ones. Confidently misread phrases ("et des eaux" for "en dessous") are not caught.
- Without `flash-linear-attention` and `causal-conv1d`, the Qwen3.5-based models (0.8B, 4B) use slower reference kernels.

## Calibration

- Text zones start at the printed label because handwriting overlaps it; the label is stripped from the OCR output (fuzzy match), as are labels of neighbouring zones caught at the end.
- `has_handwriting`: at least 400 dark pixels (< 150) outside the dilated template print. On this scan empty zones stay below ~450 and the shortest real answer ("Oui") is 471.
- Marks: empty shapes score ≤ 0.08, marked ones ≥ 0.108, threshold 0.09; scores in 0.05–0.15 are flagged "(à vérifier)". The star rating is the highest marked star.
