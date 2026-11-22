"""Download a small, attributed, *unlabeled by individual* photo set from Wikimedia Commons."""

from __future__ import annotations

import json
import hashlib
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "przewalski_unlabeled"
TITLES = [
    "File:Equus przewalskii Parc du Thot 02 2006-07-18.jpg",
    "File:Equuus ferus przewalskii.jpg",
    "File:In den Rödern bei Babenhausen, Przewalskipferd Portrait 2014.JPG",
    "File:Kone Prevalskaho Na Divcich Hradech Praha 5 D.jpg",
    "File:Koń.przewalskiego.2025.jpg",
    "File:MaleMongolianHorses.jpg",
    "File:Przewalski's wild horse (2917708174).jpg",
    "File:Przewalski's wild horse (2917714434).jpg",
    "File:Przewalskipferd baumstamm beknabbernd 1.jpg",
    "File:SalgueroDeJuarrosPrzewalskiNIeve2016-02-16.JPG",
]


def request(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "TakhiFieldbook/1.0 (research photo catalogue)"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = {
        "action": "query", "format": "json", "titles": "|".join(TITLES),
        "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": "1200",
    }
    api = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    pages = json.loads(request(api))["query"]["pages"].values()
    manifest = []
    for page in sorted(pages, key=lambda item: item.get("title", "")):
        info = (page.get("imageinfo") or [None])[0]
        if not info or info.get("mime") != "image/jpeg":
            continue
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not re.match(r"^(CC0|CC BY|Public domain)", license_name, re.I):
            continue
        title = page["title"]
        destination = OUTPUT / f"{hashlib.sha256(title.encode()).hexdigest()[:12]}.jpg"
        try:
            if not destination.exists():
                destination.write_bytes(request(info.get("thumburl") or info["url"]))
            with Image.open(destination) as image:
                image.verify()
        except Exception as error:
            destination.unlink(missing_ok=True)
            print(f"Skipped {title}: {error}")
            continue
        manifest.append({
            "file": destination.name,
            "title": title,
            "page": info["descriptionurl"],
            "author_html": meta.get("Artist", {}).get("value", ""),
            "license": license_name,
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "individual_id": None,
        })
        print(f"Saved {destination.name}: {title}")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Downloaded {len(manifest)} photos to {OUTPUT}")


if __name__ == "__main__":
    main()
