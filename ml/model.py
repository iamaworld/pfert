"""Image embeddings for animal re-identification (MegaDescriptor-T-224)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))

import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


MODEL_ID = os.environ.get("TAKHI_MODEL", "hf-hub:BVRA/MegaDescriptor-T-224")
HEAD_PATH = ROOT / "data" / "ml" / "horse_head.pt"
IMAGE_SIZE = 384 if "384" in MODEL_ID else 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def device() -> torch.device:
    # CPU is reliable on machines without Metal; MPS is used when available.
    if torch.backends.mps.is_available() and os.environ.get("TAKHI_CPU") != "1":
        return torch.device("mps")
    return torch.device("cpu")


def load_model() -> tuple[torch.nn.Module, torch.device, torch.nn.Module | None]:
    target = device()
    model = timm.create_model(MODEL_ID, pretrained=True).eval().to(target)
    head = None
    if HEAD_PATH.exists():
        checkpoint = torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
        head = torch.nn.Linear(model.num_features, model.num_features, bias=False)
        head.load_state_dict(checkpoint["state_dict"])
        head = head.eval().to(target)
    return model, target, head


def crop_views(image: Image.Image, query: bool) -> list[Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    boxes = [(0, 0, 1, 1), (0.1, 0.1, 0.9, 0.9)]
    if query:
        boxes += [
            (0, 0, 0.7, 0.7), (0.3, 0, 1, 0.7),
            (0, 0.3, 0.7, 1), (0.3, 0.3, 1, 1),
        ]
    return [image.crop((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))
            for x1, y1, x2, y2 in boxes]


@torch.inference_mode()
def embed_image(path: str | Path, model: torch.nn.Module, target: torch.device,
                head: torch.nn.Module | None = None, query: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        views = crop_views(image, query)
        batch = torch.stack([TRANSFORM(view) for view in views]).to(target)
    output = model(batch)
    if head is not None:
        output = head(output)
    return F.normalize(output.float(), dim=1).cpu()
