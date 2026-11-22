"""Find horse boxes with the official COCO weights for Ultralytics YOLO26x."""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt"
WEIGHTS_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"


def load_detector(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    from ultralytics import YOLO
    model = YOLO(str(path), task="detect")
    horse = next(index for index, name in model.names.items() if name == "horse")
    return model, horse


@torch.inference_mode()
def detect(image, detector):
    model, horse = detector
    result = model.predict(image, classes=[horse], conf=0.35, imgsz=640,
                           device="cpu", max_det=30, verbose=False)[0]
    boxes = []
    for x1, y1, x2, y2 in result.boxes.xyxyn.cpu().tolist():
        width, height = x2 - x1, y2 - y1
        if min(width, height) >= 0.015:
            boxes.append({"x": x1, "y": y1, "width": width, "height": height})
    return sorted(boxes[:15], key=lambda box: (box["x"] + box["width"] / 2, box["y"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("images", nargs="*", type=Path)
    args = parser.parse_args()
    if not args.images:
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(args.weights), hash_prefix=WEIGHTS_SHA256)
        return
    torch.set_num_threads(4)
    detector = load_detector(args.weights)
    results = []
    for path in args.images:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        results.append(detect(image, detector))
    print(json.dumps(results[0] if len(results) == 1 else results))


if __name__ == "__main__":
    main()
