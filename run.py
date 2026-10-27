import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
from PIL import Image

from marks import has_handwriting, read_checkboxes, read_stars
from ocr import DEFAULT_MODEL, Transcriber, is_french_text
from register import Registrar
from split import render_halves
from template import load_canvases

PDF = Path("data/Scan_atelier_cotech_elargi.pdf")
PAGES_DIR = Path("work/pages")
CACHE_DIR = Path("work/cache")
OUTPUT_DIR = Path("output")
CANVAS_ORDER = ["canevas1", "canevas2", "canevas3", "consommation"]
LABEL_MATCH = 0.75
UNCERTAIN = " _(à vérifier)_"
EMPTY = "_(vide)_"


def ensure_halves():
    if any(PAGES_DIR.glob("*.png")):
        return
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for page_number, position, half in render_halves(PDF):
        half.save(PAGES_DIR / f"page_{page_number:02d}_{position}.png")


def normalize(text):
    return text.replace("’", "'").lower()


def label_end(text, label):
    target = normalize(label)
    score = lambda end: SequenceMatcher(None, normalize(text[:end]), target).ratio()
    best = max(range(len(target) // 2, min(len(text), len(target) * 2) + 1), key=score, default=None)
    return best if best is not None and score(best) >= LABEL_MATCH else 0


def strip_labels(text, label, other_labels):
    if label:
        text = text[label_end(text, label):]
    for other in other_labels:
        text = text[:len(text) - label_end(text[::-1], other[::-1])]
    return text


def clean(text):
    text = re.sub(r"\s+", " ", text.replace("**", "")).strip(" :*-\n")
    return re.sub(r"^[-–□☐■•*\s]+", "", text)


class Reader:
    def __init__(self, model_id, scale=1):
        self.model_id = model_id
        self.scale = scale
        self.cache_dir = CACHE_DIR / (model_id.replace("/", "__") + (f"@{scale}x" if scale != 1 else ""))
        self.transcriber = None
        self.seconds = []

    def transcribe(self, stem, key, crop):
        path = self.cache_dir / f"{stem}.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        if key not in cache or not is_french_text(cache[key]):
            if self.transcriber is None:
                self.transcriber = Transcriber(self.model_id)
            start = time.time()
            image = Image.fromarray(crop)
            if self.scale != 1:
                image = image.resize((round(image.width * self.scale), round(image.height * self.scale)), Image.LANCZOS)
            cache[key] = self.transcriber.transcribe(image)
            self.seconds.append(time.time() - start)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        return cache[key]


def field_key(field):
    return f"{field['section']}|{field['question']}|{field['bbox']}"


def read_text(reader, stem, warped, template, field, labels):
    x0, y0, x1, y1 = field["bbox"]
    if not has_handwriting(warped, template, field["bbox"]):
        return ""
    raw = reader.transcribe(stem, field_key(field), warped[y0:y1, x0:x1])
    other_labels = labels - {field.get("label")}
    return clean(strip_labels(clean(raw), field.get("label"), other_labels))


def read_canvas(reader, stem, warped, canvas):
    template = canvas["image"]
    labels = {field["label"] for field in canvas["fields"] if "label" in field}
    answers = []
    for field in canvas["fields"]:
        if field["type"] == "text":
            answers.append((field, read_text(reader, stem, warped, template, field, labels)))
        elif field["type"] == "stars":
            answers.append((field, read_stars(warped, template, field["stars"])))
        else:
            answers.append((field, read_checkboxes(warped, template, field["options"])))
    return answers


def is_filled(field, value):
    if field["type"] == "text":
        return bool(value)
    if field["type"] == "stars":
        return value[0] is not None
    return any(checked for _, checked, _ in value)


def format_value(field, value):
    if field["type"] == "text":
        return value or EMPTY
    if field["type"] == "stars":
        rating, uncertain = value
        text = f"{rating}/{len(field['stars'])}" if rating else "_(aucune note)_"
        return text + (UNCERTAIN if uncertain else "")
    return "\n".join(f"- [{'x' if checked else ' '}] {label}{UNCERTAIN if uncertain else ''}" for label, checked, uncertain in value)


def render_canvas(canvas, answers, stem):
    lines = [f"## {canvas['title']}", ""]
    sections = {}
    for field, value in answers:
        sections.setdefault(field["section"], []).append((field, value))
    for section, items in sections.items():
        if section and not any(is_filled(field, value) for field, value in items):
            continue
        level = "###"
        if section:
            lines += [f"### {section}", ""]
            level = "####"
        for field, value in items:
            lines += [f"{level} {field['question']}", "", format_value(field, value), ""]
    lines += [f"Source : `{PAGES_DIR / stem}.png`", ""]
    return lines


def participant_name(answers):
    return next((value for field, value in answers.get("canevas1", []) if field["question"] == "Votre nom"), "")


def write_participant(number, canvases, results):
    name = participant_name({name: answers for name, (answers, _) in results.items()})
    title = f"# Participant {number:02d}" + (f" — {name}" if name else "")
    lines = [title, "", f"Pages {2 * number - 1} et {2 * number} du scan.", ""]
    for canvas_name in CANVAS_ORDER:
        if canvas_name in results:
            answers, stem = results[canvas_name]
            lines += render_canvas(canvases[canvas_name], answers, stem)
    path = OUTPUT_DIR / f"participant_{number:02d}.md"
    path.write_text("\n".join(lines))
    return path


def participant_stems(number):
    return [f"page_{page:02d}_{position}" for page in (2 * number - 1, 2 * number) for position in ("top", "bottom")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--participants", type=int, nargs="*", default=list(range(1, 13)))
    args = parser.parse_args()
    ensure_halves()
    OUTPUT_DIR.mkdir(exist_ok=True)
    canvases = load_canvases()
    registrar = Registrar(canvases)
    reader = Reader(args.model)
    for number in args.participants:
        results = {}
        for stem in participant_stems(number):
            name, warped, _ = registrar.register(cv2.imread(str(PAGES_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
            results[name] = (read_canvas(reader, stem, warped, canvases[name]), stem)
        path = write_participant(number, canvases, results)
        timing = f", {sum(reader.seconds):.0f} s OCR over {len(reader.seconds)} zones so far" if reader.seconds else ""
        print(f"{path}{timing}", flush=True)


if __name__ == "__main__":
    main()
