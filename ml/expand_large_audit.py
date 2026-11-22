"""Expand the audit with attributed photos from additional zoo groups."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time

from PIL import Image

from build_benchmark import HORSE30, OUT as OLD, get, image_info
from build_large_audit import OUT, MANIFEST, contact, write
from expand_benchmark import OUT as NEW, category_titles

TARGET_POSITIVE = 100
MAX_PER_CATEGORY = 5
CURATION_EXCLUDE = {"File:Животные на фоне новостроек, обступивших зоопарк со всех сторон.jpg"}
ZOO_CATEGORIES = [
    "Equus przewalskii in Askania-Nova Biosphere Reserve",
    "Equus ferus przewalskii in Diergaarde Blijdorp",
    "Equus przewalskii in Kyiv Zoo",
    "Equus przewalskii in Tierpark Hellabrunn",
    "Equus przewalskii in Cincinnati Zoo",
    "Equus przewalskii in Cologne Zoo",
    "Equus przewalskii in Juraparc",
    "Equus przewalskii in Natuurpark Lelystad",
    "Equus przewalskii in Tiergarten Nürnberg",
    "Equus przewalskii in Planckendael",
    "Equus przewalskii in Prague Zoo",
    "Equus przewalskii in Zoo Salzburg",
    "Equus przewalskii in Toronto Zoo",
    "Equus przewalskii in Warsaw Zoo",
    "Equus przewalskii in Smithsonian National Zoological Park",
    "Equus przewalskii in Werribee Open Range Zoo",
    "Equus przewalskii in Woburn Safari Park",
    "Equus przewalskii in Welsh Mountain Zoo",
    "Equus przewalskii in Tierpark Berlin",
    "Equus przewalskii in Wildpark Bruderhaus",
    "Equus przewalskii in Wilhelma Zoo",
    "Equus przewalskii in Wildpark Pforzheim",
    "Equus przewalskii in Zoo Neuwied",
    "Equus przewalskii in Tiergarten Kleve",
]


def add_domestic(records: list[dict]) -> None:
    used = {row["title"] for row in records} | CURATION_EXCLUDE
    for folder in (OLD, NEW):
        used.update(row["title"] for row in json.loads((folder / "manifest.json").read_text()))
    for individual in sorted(HORSE30.glob("Sample*"), key=lambda p: int(p.name[6:])):
        paths = sorted(individual.glob("*.png"))
        added = sum(row.get("expansion_phase") == "large_600" and row.get("individual_id") == individual.name
                    for row in records)
        if added >= 12:
            continue
        for i in range(1, len(paths)):
            source = paths[(i * 37) % len(paths)]
            title = f"Horse-30 {individual.name} frame {source.stem}"
            if title in used:
                continue
            relative = f"negative/horse30_{individual.name}_{source.stem}.png"
            destination = OUT / relative
            if not destination.exists():
                shutil.copyfile(source, destination)
            records.append({"file": relative, "label": "not_przewalski", "kind": "domestic_horse",
                            "title": title, "page": "https://huggingface.co/datasets/mwmathis/Horse-30",
                            "license": "CC BY-NC-SA 4.0", "individual_id": individual.name,
                            "source_group": f"Horse-30:{individual.name}", "expansion_phase": "large_600"})
            used.add(title)
            added += 1
            if added >= 12:
                break
    write(records)


def add_positive(records: list[dict]) -> None:
    used = {row["title"] for row in records} | CURATION_EXCLUDE
    for folder in (OLD, NEW):
        used.update(row["title"] for row in json.loads((folder / "manifest.json").read_text()))
    for category in ZOO_CATEGORIES:
        if sum(row["label"] == "przewalski" for row in records) >= TARGET_POSITIVE:
            break
        have = sum(row.get("source_category") == category for row in records)
        if have >= MAX_PER_CATEGORY:
            continue
        try:
            titles = [title for title in category_titles(category) if title not in used]
            infos = image_info(titles[:50])
        except Exception as error:
            print(f"Metadata delayed for {category}: {error}", flush=True)
            continue
        for title in titles:
            if have >= MAX_PER_CATEGORY or sum(row["label"] == "przewalski" for row in records) >= TARGET_POSITIVE:
                break
            info = infos.get(title)
            if not info or info.get("mime") != "image/jpeg":
                continue
            meta = info.get("extmetadata", {})
            license_name = meta.get("LicenseShortName", {}).get("value", "")
            if not re.match(r"^(CC0|CC BY|Public domain)", license_name, re.I):
                continue
            filename = hashlib.sha256(title.encode()).hexdigest()[:12] + ".jpg"
            relative = f"positive/{filename}"
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
            records.append({"file": relative, "label": "przewalski", "kind": "photo",
                            "title": title, "page": info["descriptionurl"],
                            "author_html": meta.get("Artist", {}).get("value", ""),
                            "license": license_name,
                            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                            "individual_id": None, "source_category": category,
                            "source_group": category})
            used.add(title)
            have += 1
            write(records)
            print(f"Positive {sum(row['label'] == 'przewalski' for row in records)}/{TARGET_POSITIVE}: {title}", flush=True)
            time.sleep(0.5)


def main() -> None:
    records = json.loads(MANIFEST.read_text())
    records = [row for row in records if row["title"] not in CURATION_EXCLUDE]
    write(records)
    add_domestic(records)
    add_positive(records)
    used_files = {row["file"] for row in records}
    for folder in (OUT / "positive", OUT / "negative"):
        for path in folder.iterdir():
            if path.is_file() and str(path.relative_to(OUT)) not in used_files:
                path.unlink()
    contact(records, "przewalski", "positive_contact.jpg")
    print(f"Large audit: {len(records)} files, {sum(row['label'] == 'przewalski' for row in records)} Przewalski", flush=True)


if __name__ == "__main__":
    main()
