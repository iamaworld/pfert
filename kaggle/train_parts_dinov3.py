"""Kaggle: fine-tune DINOv3 part segmentation (ears/legs/body) with a
hand-labeled seed and a pseudo-label self-training loop.

Run this in a Kaggle notebook with GPU (P100/T4 x2 is enough):

    export HF_TOKEN=hf_...          # DINOv3 checkpoints are gated
    pip install transformers torch torchvision pillow
    python train_parts_dinov3.py --images /kaggle/input/takhi-parts/images \
        --labels /kaggle/input/takhi-parts/labels.json \
        --unlabeled /kaggle/input/takhi-unlabeled

The script exports ``horse_parts.pt`` (TorchScript) that the local service
loads via ``ml/parts_segment.py``. Contract of the exported module:
    forward(rgb_uint8_tensor[N,3,H,W]) -> logits[N,4,H,W]
    channel order: 0 background, 1 ears, 2 legs, 3 body
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, DINOv3ViTModel

CLASSES = ["background", "ears", "legs", "body"]
COLORS = [(0, 0, 0), (214, 75, 60), (58, 111, 216), (58, 169, 92)]
SIZE = 448
PATCH = 16
PATCHES = SIZE // PATCH  # 28 for ViT-S/16
# DINOv3 ViT hidden sequence: 1 CLS + 4 register tokens + HxW patches.
REGISTERS = 4
PATCH_OFFSET = 1 + REGISTERS


class PartsDataset(Dataset):
    def __init__(self, images: list[Path], labels: dict, transform, train: bool = True):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.transform(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]
        if self.train:
            mask = torch.zeros(PATCHES, PATCHES, dtype=torch.long)
            for polygon in self.labels.get(path.name, []):
                color = COLORS[CLASSES.index(polygon["className"])]
                points = [(p["x"] * PATCHES, p["y"] * PATCHES) for p in polygon["points"]]
                mask = fill_polygon(mask, points, CLASSES.index(polygon["className"]))
            return pixel_values, mask
        return pixel_values, path.name


def fill_polygon(mask: torch.Tensor, points: list, class_id: int) -> torch.Tensor:
    """Rasterize one polygon into the patch grid (CPU, fine for small grids)."""
    from PIL import Image as PILImage, ImageDraw
    canvas = PILImage.new("L", (PATCHES, PATCHES), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(x, y) for x, y in points], fill=class_id)
    return torch.maximum(mask, torch.tensor(list(canvas.getdata()), dtype=torch.long).view(PATCHES, PATCHES))


class PartsModel(nn.Module):
    def __init__(self, backbone: DINOv3ViTModel, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, len(CLASSES))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.backbone(pixel_values).last_hidden_state[:, PATCH_OFFSET:]
        b, n, _ = hidden.shape
        grid = int(n ** 0.5)
        logits = self.head(hidden).transpose(1, 2).reshape(b, len(CLASSES), grid, grid)
        return F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = correct = 0
    for pixel_values, masks in loader:
        pixel_values, masks = pixel_values.to(device), masks.to(device)
        logits = model(pixel_values)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += masks.numel(); correct += (logits.argmax(1) == masks).sum().item()
    return correct / total


def pseudo_label(model, unlabeled: list[Path], transform, device, threshold=0.9):
    """Assign confident labels to unlabeled photos, then reuse them as data."""
    labels = {}
    model.eval()
    for path in unlabeled:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = transform(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs["pixel_values"].to(device))
        probs = F.softmax(logits[0], dim=0).cpu()
        confident = (probs.max(0).values > threshold)
        # Keep only pixels of the three foreground classes that pass the bar.
        for class_id in (1, 2, 3):
            if (confident & (probs.argmax(0) == class_id)).float().mean() > 0.005:
                labels.setdefault(path.name, []).append({
                    "className": CLASSES[class_id],
                    "points": mask_to_polygon((probs.argmax(0) == class_id)),
                })
    return labels


def mask_to_polygon(mask) -> list:
    """Cheap mask -> rectangle polygon fallback for pseudo-labels."""
    ys, xs = torch.where(mask)
    if not len(xs):
        return []
    x1, x2 = xs.min().item() / PATCHES, xs.max().item() / PATCHES
    y1, y2 = ys.min().item() / PATCHES, ys.max().item() / PATCHES
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder with hand-labeled photos")
    parser.add_argument("--labels", type=Path, required=True, help="labels.json from tools/labeler.html")
    parser.add_argument("--unlabeled", type=Path, default=None, help="Optional folder for self-training")
    parser.add_argument("--backbone", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=3, help="Pseudo-label rounds")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("horse_parts.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.backbone, use_fast=True)
    backbone = DINOv3ViTModel.from_pretrained(args.backbone).eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    images = sorted(args.images.glob("*"))
    images = [path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    labeled = [path for path in images if path.name in labels]
    print(f"Labeled {len(labeled)} photos, {sum(len(v) for v in labels.values())} polygons", flush=True)

    model = PartsModel(backbone, backbone.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=1e-3)
    transform = lambda image, **kw: processor(images=image, return_tensors="pt", **kw)

    for round_index in range(1 + args.rounds):
        dataset = PartsDataset(labeled, labels, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epochs):
            accuracy = train_one_epoch(model, loader, optimizer, device)
            if (epoch + 1) % 5 == 0:
                print(f"Round {round_index} epoch {epoch + 1}: pixel accuracy {accuracy:.3f}", flush=True)
        if args.unlabeled and round_index < args.rounds:
            pool = sorted(p for p in args.unlabeled.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            new_labels = pseudo_label(model, pool, transform, device)
            labels.update(new_labels)
            labeled = sorted({path.name: path for path in labeled + pool if path.name in labels}.values())
            print(f"Round {round_index}: added {len(new_labels)} pseudo-labeled photos", flush=True)

    # Export the full pipeline to TorchScript for the local service.
    example = torch.zeros(1, 3, SIZE, SIZE)
    traced = torch.jit.trace(model.eval().cpu(), example)
    traced.save(str(args.output))
    print(f"Exported {args.output}", flush=True)


if __name__ == "__main__":
    main()
