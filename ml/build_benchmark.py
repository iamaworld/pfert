"""Build a sourced species-level test set: Przewalski horses vs domestic horses."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "benchmark"
EXISTING = ROOT / "datasets" / "przewalski_unlabeled"
HORSE30 = ROOT / "datasets" / "horse30" / "labeled-data"
BASE = "https://commons.wikimedia.org/w/api.php?"
USER_AGENT = "TakhiFieldbook/1.0 (research benchmark; source details in DATASETS.md)"

POSITIVE_TITLES = [
    "File:Equus przewalskii Parc du Thot 02 2006-07-18.jpg",
    "File:Equuus ferus przewalskii.jpg",
    "File:In den Rödern bei Babenhausen, Przewalskipferd Portrait 2014.JPG",
    "File:Kone Prevalskaho Na Divcich Hradech Praha 5 D.jpg",
    "File:MaleMongolianHorses.jpg",
    "File:Przewalski's wild horse (2917708174).jpg",
    "File:Przewalski's wild horse (2917714434).jpg",
    "File:Przewalskipferd baumstamm beknabbernd 1.jpg",
    "File:SalgueroDeJuarrosPrzewalskiNIeve2016-02-16.JPG",
    "File:03 2-NP Hustain (74).JPG",
    "File:Cheval de Przewalski Mongolia.jpg",
    "File:Group of Takhi at Hustai.jpg",
    "File:Herd of Babeta.jpg",
    "File:Khustain Nuruu National Park (13).JPG",
    "File:Khustain Nuruu National Park (15).JPG",
    "File:Mongolian Wild Horse (8367798393).jpg",
    "File:Mongolian Wild Horse (8368865828).jpg",
    "File:Takhi Hustai.jpg",
    "File:Takhi – Wild horse.jpg",
    "File:Takhi.jpg",
    "File:Przewalskis horse 02.jpg",
    "File:2016-09 zoo sauvage de Saint-Félicien - Equus przewalskii.jpg",
    "File:Caballo de przewalski en Santillana.JPG",
    "File:Cheval de Przewalski DSCF5552.jpg",
    "File:Equus przewalskii in Djeyran ecological centre frontal view.jpg",
    "File:Equus przewalskii in Djeyran ecological centre.jpg",
    "File:Equus przewalskii Tallinn.JPG",
    "File:Przewalski's Horse at The Wilds.jpg",
    "File:Przewalski's Wild Horse in Taipei Zoo.jpg",
    "File:Przewalski's colt running.jpg",
    "File:Portrait of Przewalski's Horse.jpg",
    "File:Cavalls de Przewalski (03-11-2022).JPG",
    "File:Лошадь Пржевальского в полесье (cropped).jpg",
]

NEGATIVE_TITLES = [
    ("File:Medium size zebra.jpg", "zebra"),
    ("File:Masai Mara Zebra.jpg", "zebra"),
    ("File:Lake Nakuru National Park Zebras 03.jpg", "zebra"),
    ("File:Ngorongoro zebra in a defensive position.jpg", "zebra"),
    ("File:Eskişehir Hayvanat Bahçesi.jpg", "zebra"),
    ("File:A donkey standing.jpg", "donkey"),
    ("File:Donkey near Close Lea - geograph.org.uk - 5753898.jpg", "donkey"),
    ("File:A Donkey at Unilorin zoo.jpg", "donkey"),
    ("File:Horse statue of the deserted shrine - panoramio.jpg", "horse_statue"),
]


def get(url: str) -> bytes:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=45) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == 4 or isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503):
                raise
            time.sleep(2 ** attempt + 1)
    raise RuntimeError("unreachable")


def image_info(titles: list[str]) -> dict[str, dict]:
    result = {}
    for start in range(0, len(titles), 25):
        params = {"action": "query", "format": "json", "titles": "|".join(titles[start:start + 25]),
                  "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": "960"}
        payload = json.loads(get(BASE + urllib.parse.urlencode(params)))
        for page in payload["query"]["pages"].values():
            if page.get("imageinfo"):
                result[page["title"]] = page["imageinfo"][0]
    return result


def positive_manifest() -> list[dict]:
    target = OUT / "positive"
    target.mkdir(parents=True, exist_ok=True)
    previous = {entry["title"]: EXISTING / entry["file"]
                for entry in json.loads((EXISTING / "manifest.json").read_text(encoding="utf-8"))}
    infos = image_info(POSITIVE_TITLES)
    records = []
    for title in POSITIVE_TITLES:
        info = infos.get(title)
        if not info or info.get("mime") != "image/jpeg":
            continue
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not re.match(r"^(CC0|CC BY|Public domain)", license_name, re.I):
            continue
        file = hashlib.sha256(title.encode()).hexdigest()[:12] + ".jpg"
        destination = target / file
        try:
            if not destination.exists():
                if title in previous and previous[title].is_file():
                    shutil.copyfile(previous[title], destination)
                else:
                    destination.write_bytes(get(info.get("thumburl") or info["url"]))
            with Image.open(destination) as image:
                image.verify()
        except Exception as error:
            destination.unlink(missing_ok=True)
            print(f"Skipped {title}: {error}", flush=True)
            continue
        records.append({"file": f"positive/{file}", "label": "przewalski", "kind": "photo",
                        "title": title, "page": info["descriptionurl"],
                        "author_html": meta.get("Artist", {}).get("value", ""),
                        "license": license_name,
                        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                        "individual_id": None})
        print(f"Positive {len(records)}: {title}", flush=True)
        if len(records) >= 30:
            break
    return records


def negative_manifest() -> list[dict]:
    target = OUT / "negative"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    sample_dirs = sorted(HORSE30.glob("Sample*"), key=lambda p: int(p.name[6:]))
    if len(sample_dirs) < 20:
        raise RuntimeError("Horse-30 is missing. Download and extract it first.")
    for position in range(20):
        folder = sample_dirs[position]
        paths = sorted(folder.glob("*.png"))
        source = paths[len(paths) // 3]
        file = f"horse30_{folder.name}_{position:02d}.png"
        shutil.copyfile(source, target / file)
        records.append({"file": f"negative/{file}", "label": "not_przewalski",
                        "kind": "domestic_horse", "title": f"Horse-30 {folder.name} frame {source.stem}",
                        "page": "https://huggingface.co/datasets/mwmathis/Horse-30",
                        "license": "CC BY-NC-SA 4.0", "individual_id": folder.name})
    infos = image_info([title for title, _ in NEGATIVE_TITLES])
    for title, kind in NEGATIVE_TITLES:
        info = infos.get(title)
        if not info or info.get("mime") != "image/jpeg":
            print(f"Missing negative: {title}", flush=True)
            continue
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not re.match(r"^(CC0|CC BY|Public domain)", license_name, re.I):
            print(f"Unlicensed negative: {title}", flush=True)
            continue
        file = hashlib.sha256(title.encode()).hexdigest()[:12] + ".jpg"
        destination = target / file
        try:
            if not destination.exists():
                destination.write_bytes(get(info.get("thumburl") or info["url"]))
            with Image.open(destination) as image:
                image.verify()
        except Exception as error:
            destination.unlink(missing_ok=True)
            print(f"Skipped negative {title}: {error}", flush=True)
            continue
        records.append({"file": f"negative/{file}", "label": "not_przewalski",
                        "kind": kind, "title": title, "page": info["descriptionurl"],
                        "author_html": meta.get("Artist", {}).get("value", ""),
                        "license": license_name,
                        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                        "individual_id": None})
        print(f"Negative {len(records)}: {title}", flush=True)
    while len(records) < 30:
        position = len(records) - 20
        folder = sample_dirs[position]
        paths = sorted(folder.glob("*.png"))
        source = paths[len(paths) * 2 // 3]
        file = f"horse30_{folder.name}_extra.png"
        shutil.copyfile(source, target / file)
        records.append({"file": f"negative/{file}", "label": "not_przewalski",
                        "kind": "domestic_horse", "title": f"Horse-30 {folder.name} frame {source.stem}",
                        "page": "https://huggingface.co/datasets/mwmathis/Horse-30",
                        "license": "CC BY-NC-SA 4.0", "individual_id": folder.name})
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = positive_manifest() + negative_manifest()
    if sum(item["label"] == "przewalski" for item in records) != 30:
        raise RuntimeError("Need exactly 30 reviewed Przewalski photos")
    if sum(item["label"] == "not_przewalski" for item in records) != 30:
        raise RuntimeError("Need exactly 30 negative examples")
    (OUT / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    used = {item["file"] for item in records}
    for folder in (OUT / "positive", OUT / "negative"):
        for file in folder.iterdir():
            if file.is_file() and str(file.relative_to(OUT)) not in used:
                file.unlink()
    for label, target_name in (("przewalski", "positive_contact.jpg"),
                               ("not_przewalski", "negative_contact.jpg")):
        selected = [item for item in records if item["label"] == label]
        sheet = Image.new("RGB", (1000, 6 * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(selected):
            with Image.open(OUT / item["file"]) as source:
                thumb = ImageOps.fit(source.convert("RGB"), (195, 160))
            x, y = (index % 5) * 200, (index // 5) * 190
            sheet.paste(thumb, (x, y))
            draw.text((x + 4, y + 164), f"{index + 1}: {item['kind']}", fill="black")
        sheet.save(OUT / target_name, quality=86)
    print(f"Benchmark: {sum(x['label'] == 'przewalski' for x in records)} positive, "
          f"{sum(x['label'] == 'not_przewalski' for x in records)} negative", flush=True)


if __name__ == "__main__":
    main()
