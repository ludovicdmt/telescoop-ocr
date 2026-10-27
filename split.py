import argparse
from pathlib import Path

import pymupdf
from PIL import Image

DPI = 300
OVERLAP = 0.05


def render_halves(pdf_path, dpi=DPI):
    doc = pymupdf.open(pdf_path)
    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        width, height = image.size
        margin = int(height * OVERLAP)
        halves = {
            "top": image.crop((0, 0, width, height // 2 + margin)),
            "bottom": image.crop((0, height // 2 - margin, width, height)),
        }
        for position, half in halves.items():
            yield page_number, position, half.rotate(90, expand=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("work/pages"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for page_number, position, half in render_halves(args.pdf):
        half.save(args.out / f"page_{page_number:02d}_{position}.png")


if __name__ == "__main__":
    main()
