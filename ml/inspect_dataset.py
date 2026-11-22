"""Inventory a large horse dataset without unpacking or reading image bytes."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic"}
DATE_PATTERN = re.compile(r"(?<!\d)(?:20\d{2})[-_. ]?(?:0[1-9]|1[0-2])[-_. ]?(?:0[1-9]|[12]\d|3[01])(?!\d)")
METADATA_WORDS = ("passport", "pedigree", "studbook", "horse", "identity", "label",
                  "паспорт", "племен", "особ", "лошад", "метк")


def entries(path: Path):
    if path.name.endswith((".crdownload", ".part", ".download")):
        raise ValueError("Архив ещё скачивается. Запустите проверку после завершения загрузки.")
    if path.is_dir():
        for folder, _, files in os.walk(path):
            for filename in files:
                file = Path(folder) / filename
                relative = file.relative_to(path).as_posix()
                try:
                    yield relative, file.stat().st_size
                except OSError:
                    continue
    elif path.is_file():
        try:
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if not item.is_dir():
                        yield item.filename, item.file_size
        except zipfile.BadZipFile as error:
            raise ValueError("ZIP не открывается: загрузка может быть не завершена или архив повреждён.") from error
    else:
        raise ValueError(f"Путь не найден: {path}")


def inventory(path: Path) -> dict:
    extensions = collections.Counter()
    first_folders = collections.Counter()
    second_folders = collections.Counter()
    metadata = []
    dated_examples = []
    image_examples = []
    total = 0
    images = 0
    unpacked_bytes = 0
    for relative, size in entries(path):
        parts = PurePosixPath(relative).parts
        if not parts or relative.startswith("__MACOSX/"):
            continue
        total += 1
        unpacked_bytes += size
        suffix = PurePosixPath(relative).suffix.lower()
        extensions[suffix or "(без расширения)"] += 1
        if len(parts) > 1:
            first_folders[parts[0]] += 1
        if len(parts) > 2:
            second_folders["/".join(parts[:2])] += 1
        if suffix in IMAGE_EXTENSIONS:
            images += 1
            if len(image_examples) < 12:
                image_examples.append(relative)
        elif (suffix in {".csv", ".xlsx", ".xls", ".json", ".pdf", ".txt"}
              or any(word in relative.lower() for word in METADATA_WORDS)) and len(metadata) < 30:
            metadata.append(relative)
        if DATE_PATTERN.search(relative) and len(dated_examples) < 12:
            dated_examples.append(relative)
    return {"source": str(path.resolve()), "source_type": "directory" if path.is_dir() else "zip",
            "files": total, "images": images, "unpacked_gb": round(unpacked_bytes / 1_000_000_000, 2),
            "extensions": dict(extensions.most_common(15)),
            "top_level_folders": dict(first_folders.most_common(20)),
            "two_level_folders": dict(second_folders.most_common(20)),
            "image_examples": image_examples, "metadata_examples": metadata,
            "date_in_filename_examples": dated_examples,
            "note": "Папки и имена файлов не доказывают идентификатор особи. Метки и даты надо сверить с таблицами датасета."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Путь к завершённому ZIP-архиву или папке")
    parser.add_argument("--output", type=Path, help="Необязательный путь для JSON-отчёта")
    args = parser.parse_args()
    try:
        result = inventory(args.source.expanduser())
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
