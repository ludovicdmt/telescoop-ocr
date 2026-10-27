from pathlib import Path

import numpy as np
import pymupdf
import yaml

from split import DPI

SCALE = DPI / 72


def to_pixels(bbox, origin, dy=0):
    ox, oy = origin
    x0, y0, x1, y1 = bbox
    return [round((x0 - ox) * SCALE), round((y0 + dy - oy) * SCALE), round((x1 - ox) * SCALE), round((y1 + dy - oy) * SCALE)]


def convert_field(field, origin, dy=0, section=None):
    converted = {**field, "section": section}
    if "bbox" in field:
        converted["bbox"] = to_pixels(field["bbox"], origin, dy)
    if "stars" in field:
        converted["stars"] = [to_pixels(star, origin, dy) for star in field["stars"]]
    if "options" in field:
        converted["options"] = [{**option, "bbox": to_pixels(option["bbox"], origin, dy)} for option in field["options"]]
    return converted


def expand_fields(canvas, origin):
    repeat = canvas.get("repeat")
    if not repeat:
        return [convert_field(field, origin) for field in canvas["fields"]]
    return [
        convert_field(field, origin, index * repeat["dy"], f"{repeat['prefix']} {index + 1}")
        for index in range(repeat["count"])
        for field in canvas["fields"]
    ]


def render_clip(page, clip):
    pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY, clip=pymupdf.Rect(clip))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def load_canvases(zones_path=Path("zones.yaml")):
    config = yaml.safe_load(Path(zones_path).read_text())
    doc = pymupdf.open(config["template"])
    canvases = {}
    for name, canvas in config["canvases"].items():
        origin = canvas["clip"][:2]
        canvases[name] = {
            "title": canvas["title"],
            "image": render_clip(doc[canvas["page"]], canvas["clip"]),
            "fields": expand_fields(canvas, origin),
        }
    return canvases
