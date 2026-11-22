"""Train or apply a horse-part mask head over frozen DINOv3 features."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

PARTS = ("body", "legs", "ears")
MODEL = "vit_small_patch16_dinov3.lvd1689m"
SIZE = 448
PREPROCESS = Compose([
    Resize((SIZE, SIZE), antialias=True),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and not os.environ.get("TAKHI_CPU"):
        return torch.device("mps")
    return torch.device("cpu")


def encoder(target, weights=None):
    model = timm.create_model(MODEL, pretrained=weights is None, img_size=SIZE,
                              checkpoint_path=str(weights) if weights else "")
    return model.eval().requires_grad_(False).to(target)


@torch.no_grad()
def features(model, images):
    tokens = model.forward_features(images)[:, model.num_prefix_tokens:]
    side = SIZE // 16
    return tokens.transpose(1, 2).reshape(len(images), model.num_features, side, side)


def image_tensor(path):
    with Image.open(path) as image:
        return PREPROCESS(ImageOps.exif_transpose(image).convert("RGB"))


def mask_tensor(polygons):
    mask = Image.new("L", (SIZE, SIZE))
    draw = ImageDraw.Draw(mask)
    for index, part in enumerate(PARTS, 1):
        for polygon in polygons.get(part, []):
            draw.polygon([(round(x * (SIZE - 1)), round(y * (SIZE - 1))) for x, y in polygon], fill=index)
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def head(channels):
    return nn.Sequential(nn.Conv2d(channels, 96, 1), nn.GELU(),
                         nn.Conv2d(96, len(PARTS) + 1, 3, padding=1))


def logits(model, patches):
    return F.interpolate(model(patches), (SIZE, SIZE), mode="bilinear", align_corners=False)


def data_rows(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for horse, photos in manifest["horses"].items():
        for photo in photos:
            yield horse, photo["capturedOn"], photo["id"]


def model_features(model, target, root, keys):
    for index in range(0, len(keys), 4):
        batch = keys[index:index + 4]
        images = torch.stack([image_tensor(root / "photos" / f"{key}.jpg") for key in batch])
        yield from features(model, images.to(target)).cpu()


def train(args):
    labels = {}
    for horse, day, key in data_rows(args.dataset):
        annotation = args.dataset / "labels" / f"{key}.json"
        if annotation.is_file():
            record = json.loads(annotation.read_text())
            if record.get("source") == "manual" and record.get("polygons"):
                labels[key] = (horse, day, record["polygons"])
    groups = defaultdict(list)
    for key, (_, day, _) in labels.items():
        groups[day].append(key)
    days = sorted(groups)
    if len(labels) < 10 or len(days) < 3:
        raise SystemExit("Need at least 10 manual masks covering at least three shooting dates.")
    validation_days = set(days[-max(1, len(days) // 5):])
    training = [key for day in days if day not in validation_days for key in groups[day]]
    validation = [key for day in days if day in validation_days for key in groups[day]]
    if not all(any(labels[key][2].get(part) for key in training) for part in PARTS):
        raise SystemExit("Training annotations must cover ears, legs and body.")
    torch.manual_seed(19)
    target = device()
    backbone = encoder(target, args.weights)
    keys = training + validation
    vectors = dict(zip(keys, model_features(backbone, target, args.dataset, keys), strict=True))
    masks = {key: mask_tensor(labels[key][2]) for key in keys}
    if args.pseudo:
        pseudo_path = args.dataset / "predictions.json"
        predictions = json.loads(pseudo_path.read_text())
        eligible = {key for _, day, key in data_rows(args.dataset) if day < min(validation_days)}
        pseudo = [key for key, record in predictions.items() if key in eligible and key not in labels
                  and all(record["partConfidence"].get(part, 0) >= args.pseudo_threshold for part in PARTS)]
        pseudo = pseudo[:len(training)]
        vectors.update(zip(pseudo, model_features(backbone, target, args.dataset, pseudo), strict=True))
        masks.update({key: mask_tensor(predictions[key]["polygons"]) for key in pseudo})
    else:
        pseudo = []

    decoder = head(backbone.num_features).to(target)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.002)
    frequencies = torch.bincount(torch.cat([masks[key].flatten() for key in training]), minlength=4)
    weights = frequencies.float().clamp_min(1).rsqrt()
    weights = (weights / weights.mean()).to(target)
    mixed = training + pseudo
    rng = random.Random(19)
    for _ in range(args.epochs):
        rng.shuffle(mixed)
        for key in mixed:
            prediction = logits(decoder, vectors[key].unsqueeze(0).to(target))
            loss = F.cross_entropy(prediction, masks[key].unsqueeze(0).to(target), weight=weights)
            optimizer.zero_grad()
            (loss * (1 if key in labels else 0.35)).backward()
            optimizer.step()

    confusion = torch.zeros(16, dtype=torch.int64)
    decoder.eval()
    with torch.no_grad():
        for key in validation:
            prediction = logits(decoder, vectors[key].unsqueeze(0).to(target)).argmax(1)[0].cpu()
            confusion += torch.bincount((masks[key] * 4 + prediction).flatten(), minlength=16)
    confusion = confusion.reshape(4, 4)
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    iou = {part: round((confusion[i, i] / union[i]).item(), 4) if union[i] else None
           for i, part in enumerate(("background", *PARTS))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": MODEL, "size": SIZE, "state_dict": decoder.cpu().state_dict()}, args.output)
    report = {"trainImages": len(training), "manualValidationImages": len(validation), "pseudoImages": len(pseudo),
              "validationDates": sorted(validation_days), "iou": iou, "device": str(target)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def load_segmenter(checkpoint_path, weights, target):
    backbone = encoder(target, weights)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["backbone"], checkpoint["size"]) != (MODEL, SIZE):
        raise ValueError("Incompatible horse part checkpoint")
    decoder = head(backbone.num_features).to(target)
    decoder.load_state_dict(checkpoint["state_dict"])
    return backbone, decoder.eval()


@torch.inference_mode()
def part_embeddings(path, segmenter, target):
    backbone, decoder = segmenter
    patches = features(backbone, image_tensor(path).unsqueeze(0).to(target))
    # Pool only patches predominantly inside the predicted part.
    probabilities = F.avg_pool2d(logits(decoder, patches).softmax(1), 16, 16)[0]
    labels = probabilities.argmax(0)
    result = {}
    for index, part in enumerate(PARTS, 1):
        keep = (labels == index) & (probabilities[index] >= 0.65)
        if keep.sum().item() < 2:
            continue
        weight = probabilities[index][keep]
        pooled = (patches[0, :, keep] * weight).sum(-1) / weight.sum()
        result[part] = F.normalize(pooled, dim=0).cpu()
    return result


@torch.inference_mode()
def predict(args):
    import cv2

    target = device()
    backbone, decoder = load_segmenter(args.checkpoint, args.weights, target)

    def result(patches):
        distribution = logits(decoder, patches.unsqueeze(0).to(target)).softmax(1)[0].cpu().numpy()
        classes = distribution.argmax(0)
        polygons, confidence = {}, {}
        for index, part in enumerate(PARTS, 1):
            mask = (classes == index).astype("uint8")
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons[part] = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
                if cv2.contourArea(contour) < 12:
                    continue
                points = cv2.approxPolyDP(contour, 2, True).reshape(-1, 2)
                if len(points) >= 3:
                    polygons[part].append([[round(int(x) / (SIZE - 1), 4), round(int(y) / (SIZE - 1), 4)]
                                           for x, y in points])
            if polygons[part]:
                confidence[part] = round(float(distribution[index][mask > 0].mean()), 4)
        return {"source": "model", "polygons": polygons, "partConfidence": confidence}

    if args.image:
        patches = features(backbone, image_tensor(args.image).unsqueeze(0).to(target))[0]
        print(json.dumps(result(patches)))
        return
    keys = [key for _, _, key in data_rows(args.dataset)]
    predictions = {key: result(patches)
                   for key, patches in zip(keys, model_features(backbone, target, args.dataset, keys), strict=True)}
    (args.dataset / "predictions.json").write_text(json.dumps(predictions) + "\n")
    print(f"Predicted {len(predictions)} masks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    fit = sub.add_parser("train")
    fit.add_argument("dataset", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--weights", type=Path)
    fit.add_argument("--epochs", type=int, default=18)
    fit.add_argument("--pseudo", action="store_true")
    fit.add_argument("--pseudo-threshold", type=float, default=0.92)
    fit.set_defaults(run=train)
    apply = sub.add_parser("predict")
    apply.add_argument("checkpoint", type=Path)
    source = apply.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--image", type=Path)
    apply.add_argument("--weights", type=Path)
    apply.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
