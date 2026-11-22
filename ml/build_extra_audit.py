"""Add 300 sourced domestic-horse frames from ten previously unused video scenes."""

from __future__ import annotations

import hashlib
import json
import shutil
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from model import ROOT


SOURCE = ROOT / "datasets" / "horse30" / "labeled-data"
OUT = ROOT / "datasets" / "extra_audit"
SCENES = [
    "BrownHorseinShadow", "BrownHorseintoshadow", "Brownhorselight",
    "Brownhorseoutofshadow", "ChestnutHorseLight", "Chestnuthorseongrass",
    "GreyHorseLightandShadow", "GreyHorseNoShadowBadLight",
    "TwoHorsesinvideobothmoving", "Twohorsesinvideoonemoving",
]
PER_SCENE = 30


def main() -> None:
    (OUT / "negative").mkdir(parents=True, exist_ok=True)
    rows = []
    content_hashes = set()
    for scene in SCENES:
        folder = SOURCE / scene
        annotations = next(folder.glob("CollectedData_*.csv"))
        with Image.open(next(folder.glob("*.png"))) as image:
            width, height = image.size
        eligible = []
        with annotations.open(newline="") as handle:
            for item in list(csv.reader(handle))[3:]:
                points = [(float(item[index]), float(item[index + 1]))
                          for index in range(1, len(item) - 1, 2)
                          if item[index] and item[index + 1]]
                if len(points) < 20:
                    continue
                if (min(x for x, _ in points) < 0.02 * width or
                    max(x for x, _ in points) > 0.98 * width or
                    min(y for _, y in points) < 0.01 * height or
                    max(y for _, y in points) > 0.99 * height):
                    continue
                span_x = max(x for x, _ in points) - min(x for x, _ in points)
                span_y = max(y for _, y in points) - min(y for _, y in points)
                if span_x < 0.2 * width or span_y < 0.2 * height:
                    continue
                source = SOURCE.parent / item[0]
                if source.exists():
                    eligible.append((source, len(points)))
        eligible.sort()
        # Avoid the beginning and end of a pass, where the horse is often
        # entering or leaving the camera frame despite partial keypoints.
        trim = len(eligible) // 5
        eligible = eligible[trim:-trim]
        if len(eligible) < PER_SCENE:
            raise ValueError(f"Not enough labeled horse photos in {scene}")
        for slot in range(PER_SCENE):
            center = (slot + 0.5) * len(eligible) // PER_SCENE
            source, point_count = eligible[int(center)]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in content_hashes:
                raise ValueError(f"Duplicate source image: {source}")
            content_hashes.add(digest)
            with Image.open(source) as image:
                image.verify()
            relative = f"negative/{scene}_{source.stem}.png"
            destination = OUT / relative
            if not destination.exists():
                shutil.copyfile(source, destination)
            rows.append({"file": relative, "label": "not_przewalski", "kind": "domestic_horse",
                         "title": f"Horse-30 {scene} frame {source.stem}",
                         "page": "https://huggingface.co/datasets/mwmathis/Horse-30",
                         "license": "CC BY-NC-SA 4.0", "source_group": f"Horse-30-scene:{scene}",
                         "source_scene": scene, "labeled_keypoints": point_count,
                         "sha256": digest})
    if len(rows) != 300:
        raise ValueError(f"Expected 300 photos, found {len(rows)}")
    selected_files = {row["file"] for row in rows}
    for path in (OUT / "negative").glob("*.png"):
        if str(path.relative_to(OUT)) not in selected_files:
            path.unlink()
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    sheet = Image.new("RGB", (1000, 1140), "white")
    draw = ImageDraw.Draw(sheet)
    selected = [row for scene in SCENES for row in
                [item for item in rows if item["source_scene"] == scene][::14][:3]]
    for index, row in enumerate(selected):
        with Image.open(OUT / row["file"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (195, 160))
        x, y = index % 5 * 200, index // 5 * 190
        sheet.paste(thumb, (x, y))
        draw.text((x + 3, y + 163), row["source_scene"][:24], fill="black")
    sheet.save(OUT / "contact.jpg", quality=85)
    print(f"Saved {len(rows)} domestic-horse photos across {len(SCENES)} source scenes")


if __name__ == "__main__":
    main()
