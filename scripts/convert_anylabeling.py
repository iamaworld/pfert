"""Convert X-AnyLabeling COCO export into the Takhi part-label dataset.

The fork's part trainer (ml/parts.py) expects:
    dataset/manifest.json  {"horses": {name: [{"id", "capturedOn"}, ...]}}
    dataset/photos/<id>.jpg
    dataset/labels/<id>.json {"source":"manual","polygons":{"ears":...,"legs":...,"body":...}}

X-AnyLabeling categories are mapped as:
    ears -> ears, legs -> legs, head -> body, flink -> body
Polygons are normalized to 0..1 and capturedOn is read from EXIF when possible.

Usage:
    .venv/bin/python scripts/convert_anylabeling.py labels.json photos_dir --out dataset_dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

from PIL import Image

PART_MAP = {
    "ear": "ears", "ears": "ears",
    "leg": "legs", "legs": "legs",
    "head": "body", "body": "body", "torso": "body",
    "flank": "body", "flink": "body", "side": "body",
}


def exif_day(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            stamp = exif.get_ifd(34665).get(36867) or exif.get(36867) or exif.get(36868) or exif.get(306)
        return date.fromisoformat(str(stamp)[:10].replace(":", "-")).isoformat()
    except (OSError, ValueError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco", type=Path, help="X-AnyLabeling COCO JSON")
    parser.add_argument("photos", type=Path, help="Folder with the source photos")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    args = parser.parse_args()

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    annotations: dict[int, list] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    images = {item["id"]: item for item in coco["images"]}

    photos_dir = args.out / "photos"
    labels_dir = args.out / "labels"
    photos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    photos = []
    skipped = 0
    for image_id, image in images.items():
        polygons = {"ears": [], "legs": [], "body": []}
        for annotation in annotations.get(image_id, []):
            part = PART_MAP.get(categories.get(annotation["category_id"], ""))
            if not part:
                continue
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation or not isinstance(segmentation[0], list):
                continue
            flat = segmentation[0]
            points = [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
            polygons[part].append([[x / image["width"], y / image["height"]] for x, y in points])
        if not any(polygons.values()):
            skipped += 1
            continue
        source = next(args.photos.rglob(image["file_name"]), None)
        if source is None:
            print(f"Missing photo: {image['file_name']}", flush=True)
            skipped += 1
            continue
        photo_id = hashlib.sha1(image["file_name"].encode()).hexdigest()[:20]
        shutil.copy(source, photos_dir / f"{photo_id}.jpg")
        (labels_dir / f"{photo_id}.json").write_text(
            json.dumps({"source": "manual", "polygons": polygons}), encoding="utf-8")
        photos.append({"id": photo_id, "capturedOn": exif_day(source), "file_name": image["file_name"]})

    manifest = {"horses": {"imported": photos}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    days = {photo["capturedOn"] for photo in photos if photo["capturedOn"]}
    print(json.dumps({"converted": len(photos), "skipped": skipped, "dated_days": len(days)}, ensure_ascii=False, indent=2))
    if len(days) < 3:
        print("Warning: parts trainer needs at least three distinct shooting dates for validation.")


if __name__ == "__main__":
    main()
