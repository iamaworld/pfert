"""Print the EXIF shooting date of an image as an ISO day, best-effort."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from PIL import Image


def main() -> None:
    path = Path(sys.argv[1])
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            stamp = exif.get_ifd(34665).get(36867) or exif.get(36867) or exif.get(36868) or exif.get(306)
        if not stamp:
            return
        print(date.fromisoformat(str(stamp)[:10].replace(":", "-")).isoformat())
    except (OSError, ValueError, TypeError, IndexError):
        return


if __name__ == "__main__":
    main()
