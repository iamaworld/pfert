"""Conservative image-quality check before a decisive species result."""

from PIL import Image, ImageFilter, ImageOps, ImageStat


def poor_quality(image: Image.Image) -> bool:
    gray = ImageOps.exif_transpose(image).convert("L").resize((224, 224))
    brightness = ImageStat.Stat(gray).mean[0]
    edge_contrast = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).stddev[0]
    return brightness < 80 or edge_contrast < 30
