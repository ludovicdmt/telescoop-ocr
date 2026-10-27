import argparse
import gc

import cv2
import torch

from marks import has_handwriting
from register import Registrar
from run import PAGES_DIR, Reader, ensure_halves, field_key, participant_stems, read_text
from template import load_canvases

CONFIGS = ["lightonai/LightOnOCR-3-1B", "lightonai/LightOnOCR-3-0.8B", "lightonai/LightOnOCR-3-4B"]


def parse_config(config):
    model_id, _, scale = config.partition("@")
    return model_id, float(scale) if scale else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=CONFIGS, help="model id, optionally suffixed with @scale")
    parser.add_argument("--participant", type=int, default=1)
    args = parser.parse_args()
    ensure_halves()
    canvases = load_canvases()
    registrar = Registrar(canvases)
    zones = []
    for stem in participant_stems(args.participant):
        name, warped, _ = registrar.register(cv2.imread(str(PAGES_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        canvas = canvases[name]
        labels = {field["label"] for field in canvas["fields"] if "label" in field}
        for field in canvas["fields"]:
            if field["type"] == "text" and has_handwriting(warped, canvas["image"], field["bbox"]):
                zones.append((stem, warped, canvas["image"], field, labels))
    results = {}
    for config in args.configs:
        model_id, scale = parse_config(config)
        reader = Reader(model_id, scale)
        results[config] = [read_text(reader, stem, warped, template, field, labels) for stem, warped, template, field, labels in zones]
        seconds = f"{sum(reader.seconds) / len(reader.seconds):.1f} s/zone" if reader.seconds else "cached"
        print(f"{config}: {seconds}", flush=True)
        del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for index, (stem, _, _, field, _) in enumerate(zones):
        print(f"\n## {stem} {field_key(field).rsplit('|', 1)[0]}")
        for config in args.configs:
            print(f"  {config:32} {results[config][index]}")


if __name__ == "__main__":
    main()
