"""Recrop catalog photos to the largest detected horse and refresh features.

The service compares a query crop (cut by the client around a YOLO box)
against catalog references. Seeded references were full frames, so background
dominated the comparison. This script replaces each reference with the largest
detected horse crop and recomputes the visual fallback features.

Usage:
    .venv/bin/python ml/recrop_catalog.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from detect_horses import detect, load_detector
from model import ROOT
from seed_catalog import extract_features

IMAGE_DIR = ROOT / "data" / "images"
DB_FILE = ROOT / "data" / "horses.json"
WEIGHTS = ROOT / "data" / "ml" / "yolo26x.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N photos")
    args = parser.parse_args()
    horses = json.loads(DB_FILE.read_text(encoding="utf-8"))
    detector = load_detector(WEIGHTS)
    cropped = unchanged = 0
    processed = 0
    for horse in horses:
        for photo in horse["photos"]:
            if args.limit and processed >= args.limit:
                break
            processed += 1
            path = IMAGE_DIR / Path(photo["url"]).name
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            boxes = detect(image, detector)
            if not boxes:
                unchanged += 1
                continue
            best = max(boxes, key=lambda box: box["width"] * box["height"])
            x, y, w, h = (best[key] for key in ("x", "y", "width", "height"))
            padding_x, padding_y = w * 0.08, h * 0.08
            left = max(0, x - padding_x) * image.width
            top = max(0, y - padding_y) * image.height
            right = min(image.width, (x + w + padding_x) * image.width)
            bottom = min(image.height, (y + h + padding_y) * image.height)
            if right - left < 40 or bottom - top < 40:
                unchanged += 1
                continue
            crop = image.crop((int(left), int(top), int(right), int(bottom)))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=IMAGE_DIR) as tmp:
                crop.save(tmp.name, "JPEG", quality=92)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
            photo["features"] = extract_features(crop)
            cropped += 1
    if cropped:
        target = DB_FILE.with_name(f"{DB_FILE.name}.{__import__('uuid').uuid4()}.tmp")
        target.write_text(json.dumps(horses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(DB_FILE)
    print(json.dumps({"cropped": cropped, "unchanged": unchanged, "processed": processed},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
