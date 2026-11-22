"""Build a sourced, resumable 300+ photo audit set, never used for fitting."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from build_benchmark import HORSE30, OUT as OLD, get, image_info
from expand_benchmark import OUT as NEW, category_titles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "large_audit"
MANIFEST = OUT / "manifest.json"
GROUPS = [
    ("Equus przewalskii in zoos", "przewalski", "photo", 3),
    ("Zebras of Kenya", "not_przewalski", "zebra", 13),
    ("1 donkey", "not_przewalski", "donkey", 15),
]
REJECTED_TITLES = {
    "File:Animals In Zoo Park - panoramio - ---=XEON=---.jpg",
    "File:Equus przewalskii, Kraków Zoo.JPG",
    "File:International Takhi Group Zoo Salzburg 2014.jpg",
    "File:Lovech Zoo Iz7.jpg",
    "File:Melitopol Ostrich Farm, Przewalski's Horse.JPG",
    "File:Ngựa hoang tại Sở thú.JPG",
    "File:Odessa zoo 1.JPG",
    "File:Park Dzikich Zwierząt w Kadzidłowo 01.jpg",
    "File:ParkDzikichZwierzatwKadzidlowie.02.JPG",
    "File:500px photo (136539515).jpeg",
    "File:Elephants of Kenya 02.jpg",
}


def write(records: list[dict]) -> None:
    MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def existing_titles() -> set[str]:
    titles = set()
    for folder in (OLD, NEW):
        titles.update(row["title"] for row in json.loads((folder / "manifest.json").read_text()))
    return titles


def horse30(records: list[dict]) -> None:
    used = {row["title"] for row in records} | existing_titles() | REJECTED_TITLES
    folders = sorted(HORSE30.glob("Sample*"), key=lambda p: int(p.name[6:]))
    for folder in folders:
        paths = sorted(folder.glob("*.png"))
        positions = [(position + 1) * len(paths) // 14 for position in range(13)]
        positions += [(position + 1) * len(paths) // 4 for position in range(3)]
        for offset in positions:
            source = paths[offset]
            title = f"Horse-30 {folder.name} frame {source.stem}"
            if title in used:
                # Nearby frame still belongs to the same sequence; use only
                # one frame from this slot rather than increasing correlation.
                continue
            relative = f"negative/horse30_{folder.name}_{source.stem}.png"
            destination = OUT / relative
            if not destination.exists():
                shutil.copyfile(source, destination)
            records.append({"file": relative, "label": "not_przewalski", "kind": "domestic_horse",
                            "title": title, "page": "https://huggingface.co/datasets/mwmathis/Horse-30",
                            "license": "CC BY-NC-SA 4.0", "individual_id": folder.name,
                            "source_group": f"Horse-30:{folder.name}"})
            used.add(title)
    write(records)


def commons(records: list[dict]) -> None:
    used = {row["title"] for row in records} | existing_titles() | REJECTED_TITLES
    for category, label, kind, target in GROUPS:
        have = sum(row.get("source_category") == category for row in records)
        if have >= target:
            continue
        titles = [title for title in category_titles(category) if title not in used]
        infos = image_info(titles[:50])
        for title in titles:
            if have >= target:
                break
            info = infos.get(title)
            if not info or info.get("mime") != "image/jpeg":
                continue
            meta = info.get("extmetadata", {})
            license_name = meta.get("LicenseShortName", {}).get("value", "")
            if not re.match(r"^(CC0|CC BY|Public domain)", license_name, re.I):
                continue
            filename = hashlib.sha256(title.encode()).hexdigest()[:12] + ".jpg"
            relative = f"{'positive' if label == 'przewalski' else 'negative'}/{filename}"
            destination = OUT / relative
            try:
                if not destination.exists():
                    destination.write_bytes(get(info.get("thumburl") or info["url"]))
                with Image.open(destination) as image:
                    image.verify()
            except Exception as error:
                destination.unlink(missing_ok=True)
                print(f"Skipped {title}: {error}", flush=True)
                continue
            records.append({"file": relative, "label": label, "kind": kind, "title": title,
                            "page": info["descriptionurl"],
                            "author_html": meta.get("Artist", {}).get("value", ""),
                            "license": license_name,
                            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                            "individual_id": None, "source_category": category,
                            "source_group": title.split(" (")[0]})
            used.add(title)
            have += 1
            write(records)
            print(f"{category}: {have}/{target}, total {len(records)}", flush=True)
            time.sleep(0.4)


def contact(records: list[dict], label: str, name: str) -> None:
    selected = [row for row in records if row["label"] == label and row.get("source_category")]
    if not selected:
        return
    sheet = Image.new("RGB", (1000, ((len(selected) + 4) // 5) * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        with Image.open(OUT / row["file"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (195, 160))
        x, y = i % 5 * 200, i // 5 * 190
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + 164), f"{i + 1}: {row['kind']}", fill="black")
    sheet.save(OUT / name, quality=85)


def main() -> None:
    for folder in (OUT / "positive", OUT / "negative"):
        folder.mkdir(parents=True, exist_ok=True)
    records = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    records = [row for row in records if row["title"] not in REJECTED_TITLES]
    write(records)
    horse30(records)
    commons(records)
    contact(records, "przewalski", "positive_contact.jpg")
    contact(records, "not_przewalski", "commons_negative_contact.jpg")
    print(f"Audit set: {len(records)} files", flush=True)


if __name__ == "__main__":
    main()
