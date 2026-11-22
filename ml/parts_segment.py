"""Run the optional horse part segmentation model (ears/legs/body).

The working model is trained externally (see kaggle/train_parts_dinov3.py) and
exported to TorchScript as ``data/ml/horse_parts.pt``. Without that file the
endpoint reports ``available: false`` and the UI simply hides the feature.

Contract of the TorchScript module:
    forward(rgb_uint8_tensor[N,3,H,W]) -> logits[N,4,H,W]
    channel order: 0 background, 1 ears, 2 legs, 3 body
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "ml" / "horse_parts.pt"
PART_NAMES = ["background", "ears", "legs", "body"]
PART_COLORS = [(0, 0, 0, 0), (220, 60, 60, 150), (60, 90, 220, 150), (60, 180, 90, 150)]
INPUT_SIZE = 448


def overlay_png(image: Image.Image, logits: torch.Tensor) -> str:
    probs = torch.softmax(logits, dim=0)
    labels = probs.argmax(dim=0).cpu().numpy()
    confidence = {PART_NAMES[index]: round(float(probs[index].mean()), 3) for index in range(1, 4)}
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pixels = mask.load()
    scaled = Image.fromarray(labels.astype("uint8")).resize(image.size, Image.Resampling.NEAREST)
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            pixels[x, y] = PART_COLORS[scaled.getpixel((x, y))]
    canvas = ImageOps.exif_transpose(image).convert("RGBA")
    canvas.alpha_composite(mask)
    buffer = io.BytesIO()
    canvas.convert("RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), confidence


def main() -> None:
    weights, image_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not weights.is_file():
        print(json.dumps({"available": False, "reason": "Модель частей тела не установлена."}))
        return
    try:
        model = torch.jit.load(str(weights), map_location="cpu").eval()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(__import__("numpy").array(resized)).permute(2, 0, 1).unsqueeze(0).float()
        with torch.inference_mode():
            logits = model(tensor)[0]
        overlay, confidence = overlay_png(image, logits)
        print(json.dumps({"available": True, "overlayDataUrl": overlay, "confidence": confidence}))
    except Exception as error:
        print(json.dumps({"available": False, "reason": f"Не удалось запустить модель частей тела: {error}"}))


if __name__ == "__main__":
    main()
