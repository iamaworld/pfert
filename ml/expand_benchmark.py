"""Collect a separate, attributed species test set from Commons and Horse-30."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.parse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from build_benchmark import BASE, HORSE30, OUT as OLD, get, image_info

OUT = OLD.parent / "benchmark_new"
GROUPS = [
    ("Equus przewalskii in Mongolia", "przewalski", "Mongolia", 15),
    ("Equus przewalskii in zoos", "przewalski", "zoo", 15),
]
EXCLUDED_TITLES = {
    "File:Khustain Nuruu National Park (7).JPG",
    "File:Animals In Zoo Park - panoramio - ---=XEON=---.jpg",
    "File:Equus przewalskii, Kraków Zoo.JPG",
    "File:International Takhi Group Zoo Salzburg 2014.jpg",
    "File:Lovech Zoo Iz7.jpg",
    "File:Melitopol Ostrich Farm, Przewalski's Horse.JPG",
}
REVIEW_EXCLUSIONS = {
    "File:Ngựa hoang tại Sở thú.JPG",
    "File:Odessa zoo 1.JPG",
    "File:Park Dzikich Zwierząt w Kadzidłowo 01.jpg",
}


def category_titles(category: str) -> list[str]:
    titles = []
    continuation = None
    while len(titles) < 200:
        params = {"action": "query", "format": "json", "list": "categorymembers",
                  "cmtitle": f"Category:{category}", "cmtype": "file", "cmlimit": "50"}
        if continuation:
            params["cmcontinue"] = continuation
        payload = json.loads(get(BASE + urllib.parse.urlencode(params)))
        titles += [row["title"] for row in payload["query"]["categorymembers"]]
        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            break
    return titles


def collect_commons(category: str, label: str, kind: str, count: int,
                    used: set[str]) -> list[dict]:
    titles = [title for title in category_titles(category) if title not in used]
    infos = image_info(titles)
    records = []
    for title in titles:
        info = infos.get(title)
        if not info or info.get("mime") != "image/jpeg":
            continue
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not re.match(r"^(CC0|CC BY|Public domain)", license_name, re.I):
            continue
        file = hashlib.sha256(title.encode()).hexdigest()[:12] + ".jpg"
        relative = f"{'positive' if label == 'przewalski' else 'negative'}/{file}"
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
                        "individual_id": None})
        used.add(title)
        print(f"{kind} {len(records)}/{count}: {title}", flush=True)
        if len(records) == count:
            break
    return records


def contact_sheet(records: list[dict], name: str) -> None:
    sheet = Image.new("RGB", (1000, ((len(records) + 4) // 5) * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(records):
        with Image.open(OUT / row["file"]) as source:
            thumb = ImageOps.fit(source.convert("RGB"), (195, 160))
        x, y = index % 5 * 200, index // 5 * 190
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + 164), f"{index + 1}: {row['kind']}", fill="black")
    sheet.save(OUT / name, quality=86)


def collect_horse30(old: list[dict]) -> list[dict]:
    used_titles = {row["title"] for row in old}
    records = []
    # These identities did not participate in fitting the species filter.
    for folder in sorted(HORSE30.glob("Sample*"), key=lambda p: int(p.name[6:]))[10:20]:
        paths = [path for path in sorted(folder.glob("*.png"))
                 if f"Horse-30 {folder.name} frame {path.stem}" not in used_titles]
        for source in (paths[0], paths[len(paths) // 2], paths[-1]):
            relative = f"negative/horse30_{folder.name}_{source.stem}_new.png"
            shutil.copyfile(source, OUT / relative)
            records.append({"file": relative, "label": "not_przewalski", "kind": "domestic_horse",
                            "title": f"Horse-30 {folder.name} frame {source.stem}",
                            "page": "https://huggingface.co/datasets/mwmathis/Horse-30",
                            "license": "CC BY-NC-SA 4.0", "individual_id": folder.name})
    return records


def main() -> None:
    for folder in (OUT / "positive", OUT / "negative"):
        folder.mkdir(parents=True, exist_ok=True)
    old = json.loads((OLD / "manifest.json").read_text())
    used = {row["title"] for row in old if row["page"].startswith("https://commons.wikimedia.org/")} | EXCLUDED_TITLES
    records = []
    for category, label, kind, count in GROUPS:
        found = collect_commons(category, label, kind, count, used)
        if len(found) != count:
            raise RuntimeError(f"Only {len(found)}/{count} valid photos in {category}")
        records += found
    records += collect_horse30(old)
    records = [row for row in records if row["title"] not in REVIEW_EXCLUSIONS]
    if len(records) != 57:
        raise RuntimeError(f"Expected 57 reviewed photos, found {len(records)}")
    (OUT / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet([row for row in records if row["label"] == "przewalski"], "positive_contact.jpg")
    contact_sheet([row for row in records if row["label"] == "not_przewalski"], "negative_contact.jpg")
    print("New benchmark: 27 positive, 30 negative", flush=True)


if __name__ == "__main__":
    main()
