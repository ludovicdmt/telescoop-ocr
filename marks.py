import cv2
import numpy as np

INK_LEVEL = 140
PAPER_LEVEL = 225
INTERIOR_EROSION = 9
THRESHOLD = 0.09
UNCERTAIN = (0.05, 0.15)


def interior_mask(template_crop):
    outline = (template_crop < INK_LEVEL).astype(np.uint8)
    flood = outline.copy() * 255
    height, width = flood.shape
    cv2.floodFill(flood, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 128)
    interior = (flood == 0).astype(np.uint8)
    return cv2.erode(interior, np.ones((INTERIOR_EROSION, INTERIOR_EROSION), np.uint8)).astype(bool)


def darkness(warped, template, bbox, pad=4):
    height, width = template.shape
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, width), min(y1 + pad, height)
    mask = interior_mask(template[y0:y1, x0:x1])
    pixels = warped[y0:y1, x0:x1][mask].astype(np.float32)
    return float(np.clip((PAPER_LEVEL - pixels) / PAPER_LEVEL, 0, 1).mean())


def is_uncertain(score):
    return UNCERTAIN[0] <= score <= UNCERTAIN[1]


def read_checkboxes(warped, template, options):
    results = []
    for option in options:
        score = darkness(warped, template, option["bbox"])
        results.append((option["label"], score >= THRESHOLD, is_uncertain(score)))
    return results


def read_stars(warped, template, stars):
    scores = [darkness(warped, template, star) for star in stars]
    marked = [index for index, score in enumerate(scores) if score >= THRESHOLD]
    rating = max(marked) + 1 if marked else None
    return rating, any(is_uncertain(score) for score in scores)


HANDWRITING_LEVEL = 150
PRINTED_DILATION = 11
HANDWRITING_MIN_PIXELS = 400


def handwriting_pixels(warped, template, bbox):
    x0, y0, x1, y1 = bbox
    printed = (template[y0:y1, x0:x1] < PAPER_LEVEL).astype(np.uint8)
    printed = cv2.dilate(printed, np.ones((PRINTED_DILATION, PRINTED_DILATION), np.uint8)).astype(bool)
    written = ((warped[y0:y1, x0:x1] < HANDWRITING_LEVEL) & ~printed).astype(np.uint8)
    return int(cv2.morphologyEx(written, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).sum())


def has_handwriting(warped, template, bbox):
    return handwriting_pixels(warped, template, bbox) >= HANDWRITING_MIN_PIXELS
