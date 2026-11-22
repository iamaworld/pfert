"""Build the service catalog from the reserve photo bank, without extracting it.

The archive layout is ``Фотобанк_Хакатон/<Имя (N фото)>/*.jpg``. Folder names
are treated as identity labels only after cross-checking against the passport
filenames in ``Паспорта животных/``; this script keeps the mapping explicit and
records the passport/studbook number in ``notes`` when it is known.

Selection is date-aware: for every identity we prefer photos from different
EXIF days, spread across the whole observation period, so the catalog doubles
as a growth timeline. Duplicate copies (same CRC and size) are collapsed and
conflicting duplicates are skipped.

Usage:
    .venv/bin/python ml/seed_catalog.py [archive] [--limit-per-id 6]
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
DB_FILE = ROOT / "data" / "horses.json"

# Names from the passport filenames in the archive. Femida and Tmin have no
# number in their filename; the stud book is the authoritative source.
PASSPORT_NUMBERS = {
    "Амага": "7454",
    "Азарм": "7458",
    "Витр": "7633",
    "Буда": "7669",
    "Беркут": "7706",
    "Бася": "7838",
    "Стэнли": "8008",
    "Иней": "8288",
    "Руди": "8292",
    "Шторм": "8300",
    "Весна": "8303",
    "Асфея": "8352",
    "Тигр": "8481",
    "Фемида": None,
    "Тмин": None,
}

PHOTO_FOLDER = "Фотобанк_Хакатон"
FOLDER_LABEL = re.compile(r"\s*\(\d+\s*фото\)\s*$", re.IGNORECASE)


def take_timestamp(image: Image.Image) -> str | None:
    """Return the EXIF shooting date as an ISO day, or None."""
    try:
        exif = image.getexif()
        stamp = exif.get_ifd(34665).get(36867) or exif.get(36867) or exif.get(36868) or exif.get(306)
        if not stamp:
            return None
        return date.fromisoformat(str(stamp)[:10].replace(":", "-")).isoformat()
    except (ValueError, TypeError, OSError):
        return None


def image_day(archive: zipfile.ZipFile, member) -> str:
    with archive.open(member) as file, Image.open(file) as image:
        return take_timestamp(image) or "без даты"


def luma(pixel):
    return (0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]) / 255.0


def center_square(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    box = ((width - side) // 2, (height - side) // 2, (width - side) // 2 + side, (height - side) // 2 + side)
    return image.resize(size, Image.Resampling.LANCZOS, box=box)


def extract_features(image: Image.Image) -> dict:
    """Replicate the browser-side features (dHash + histogram + grid)."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    hash_pixels = center_square(rgb, (9, 8)).load()
    bits = 0
    for y in range(8):
        for x in range(8):
            left = luma(hash_pixels[x, y])
            right = luma(hash_pixels[x + 1, y])
            bits = (bits << 1) | (1 if left > right else 0)
    sample = center_square(rgb, (64, 64)).load()
    histogram = [0] * 24
    grid = [0.0] * 64
    for y in range(64):
        for x in range(64):
            pixel = sample[x, y]
            for channel in range(3):
                histogram[channel * 8 + min(7, pixel[channel] // 32)] += 1
            grid[(y // 8) * 8 + (x // 8)] += luma(pixel)
    return {
        "hash": f"{bits:016x}",
        "histogram": [n / 4096 for n in histogram],
        "grid": [n / 64 for n in grid],
    }


def deduplicate_entries(archive: zipfile.ZipFile):
    """Collapse byte-identical copies; drop copies shared by different horses."""
    by_content: dict[tuple[int, int], list[tuple[str, object]]] = defaultdict(list)
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if (len(parts) != 3 or parts[0] != PHOTO_FOLDER
                or member.is_dir() or member.filename.startswith("__MACOSX/")):
            continue
        if member.filename.lower().endswith((".jpg", ".jpeg")):
            horse = FOLDER_LABEL.sub("", parts[1])
            by_content[(member.CRC, member.file_size)].append((horse, member))
    skipped = Counter()
    kept: dict[str, list] = defaultdict(list)
    for copies in by_content.values():
        horses = {horse for horse, _ in copies}
        if len(horses) > 1:
            skipped["conflicting_duplicates"] += len(copies)
            continue
        member = copies[0][1]
        kept[copies[0][0]].append(member)
        skipped["duplicate_copies"] += len(copies) - 1
    return kept, skipped


def select(rows: list[tuple[str, object]], limit: int) -> list[tuple[str, object]]:
    """Pick up to `limit` photos with the widest possible EXIF date spread."""
    by_day: dict[str, list] = defaultdict(list)
    for day, member in rows:
        by_day[day].append(member)
    days = sorted(by_day, key=lambda day: (day == "без даты", day))
    if len(days) <= limit:
        chosen = days
        per_day = max(1, limit // len(days) + 1)
    else:
        indexes = [round(i * (len(days) - 1) / (limit - 1)) for i in range(limit)]
        chosen = [days[i] for i in indexes]
        per_day = 1
    picked = []
    for day in chosen:
        members = sorted(by_day[day], key=lambda member: member.file_size, reverse=True)
        picked.extend((day, member) for member in members[:per_day])
    return picked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?",
                        default=ROOT / "przewalski-horses-2026-10-29 (1).zip")
    parser.add_argument("--limit-per-id", type=int, default=6)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Архив не найден: {args.archive}")

    with zipfile.ZipFile(args.archive) as archive:
        kept, skipped = deduplicate_entries(archive)
        horses = []
        for name in sorted(kept, key=lambda item: PASSPORT_NUMBERS.get(item) or ""):
            dated = []
            for member in kept[name]:
                with archive.open(member) as file, Image.open(file) as image:
                    dated.append((take_timestamp(image) or "без даты", member))
            picked = select(dated, args.limit_per_id)
            photos = []
            for day, member in picked:
                with archive.open(member) as file:
                    raw = file.read()
                with Image.open(archive.open(member)) as image:
                    features = extract_features(image)
                photo_id = uuid.uuid4()
                target = IMAGE_DIR / f"{photo_id}.jpg"
                target.write_bytes(raw)
                number = PASSPORT_NUMBERS.get(name)
                photos.append({
                    "id": str(photo_id),
                    "url": f"/images/{target.name}",
                    "features": features,
                    "createdAt": date.today().isoformat(),
                    "takenAt": day if day != "без даты" else None,
                })
            if not photos:
                continue
            notes = f"Полевой № {number}" if number else "Полевой № не указан в паспорте"
            horses.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "notes": notes,
                "createdAt": date.today().isoformat(),
                "photos": photos,
            })
            span = sorted({photo["takenAt"] for photo in photos if photo["takenAt"]})
            dates = f" {span[0]}…{span[-1]}" if span else ""
            print(f"{name}: {len(photos)} фото{dates}", flush=True)

    if not horses:
        raise SystemExit("В архиве не найдено подходящих фотографий.")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DB_FILE.write_text(json.dumps(horses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"identities": len(horses), "photos": sum(len(h["photos"]) for h in horses),
               "skipped": dict(skipped), "path": str(DB_FILE)}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
