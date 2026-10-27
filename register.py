import argparse
from pathlib import Path

import cv2
import numpy as np

from template import load_canvases

WORK_SCALE = 0.5
MIN_INLIERS = 40

orb = cv2.ORB_create(nfeatures=8000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def detect(image):
    small = cv2.resize(image, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    return orb.detectAndCompute(small, None)


def estimate(scan_features, template_features):
    scan_keypoints, scan_descriptors = scan_features
    template_keypoints, template_descriptors = template_features
    pairs = matcher.knnMatch(scan_descriptors, template_descriptors, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, 0
    source = np.float32([scan_keypoints[m.queryIdx].pt for m in good])
    target = np.float32([template_keypoints[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None, 0
    matrix[:, 2] /= WORK_SCALE
    return matrix, int(inliers.sum())


class Registrar:
    def __init__(self, canvases):
        self.canvases = canvases
        self.features = {name: detect(canvas["image"]) for name, canvas in canvases.items()}

    def register(self, scan):
        scan_features = detect(scan)
        scores = {name: estimate(scan_features, features) for name, features in self.features.items()}
        name = max(scores, key=lambda key: scores[key][1])
        matrix, inliers = scores[name]
        if inliers < MIN_INLIERS:
            raise ValueError(f"no template matched (best {name} with {inliers} inliers)")
        height, width = self.canvases[name]["image"].shape
        warped = cv2.warpAffine(scan, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
        return name, warped, inliers


def field_boxes(field):
    if "bbox" in field:
        yield field["bbox"]
    yield from field.get("stars", [])
    yield from (option["bbox"] for option in field.get("options", []))


def overlay(warped, canvas):
    image = cv2.cvtColor(cv2.addWeighted(warped, 0.6, canvas["image"], 0.4, 0), cv2.COLOR_GRAY2BGR)
    colors = {"text": (0, 160, 0), "stars": (0, 0, 220), "checkboxes": (220, 0, 0)}
    for field in canvas["fields"]:
        for x0, y0, x1, y1 in field_boxes(field):
            cv2.rectangle(image, (x0, y0), (x1, y1), colors[field["type"]], 3)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pages", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("work/debug"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    canvases = load_canvases()
    registrar = Registrar(canvases)
    for path in args.pages:
        scan = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        name, warped, inliers = registrar.register(scan)
        print(f"{path.name}: {name} ({inliers} inliers)")
        cv2.imwrite(str(args.out / f"{path.stem}_{name}_zones.png"), overlay(warped, canvases[name]))


if __name__ == "__main__":
    main()
