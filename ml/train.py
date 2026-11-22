"""Fine-tune a small metric head on labeled horse identities, then validate on unseen identities."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from model import HEAD_PATH, TRANSFORM, load_model


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect(folder: Path, limit: int) -> list[tuple[Path, str]]:
    samples = []
    for identity_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        # The Horse-30 archive also contains scene-named folders, including
        # sequences with two horses. Only Sample* folders are unambiguous here.
        if folder.name == "labeled-data" and not identity_dir.name.startswith("Sample"):
            continue
        paths = sorted(path for path in identity_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(paths) < 5:
            continue
        if len(paths) > limit:
            indexes = [round(i * (len(paths) - 1) / (limit - 1)) for i in range(limit)]
            paths = [paths[i] for i in indexes]
        samples += [(path, identity_dir.name) for path in paths]
    return samples


def collect_catalog(limit: int) -> list[tuple[Path, str]]:
    catalog = ROOT / "data" / "horses.json"
    if not catalog.exists():
        return []
    samples = []
    for horse in json.loads(catalog.read_text(encoding="utf-8")):
        paths = [ROOT / "data" / "images" / Path(photo["url"]).name
                 for photo in horse.get("photos", [])]
        paths = [path for path in paths if path.is_file()]
        if len(paths) >= 5:
            samples += [(path, horse["id"]) for path in paths[:limit]]
    return samples


class Photos(Dataset):
    def __init__(self, samples: list[tuple[Path, str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = TRANSFORM(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, label


@torch.inference_mode()
def extract(samples, model, target, batch_size):
    vectors, labels = [], []
    loader = DataLoader(Photos(samples), batch_size=batch_size, num_workers=0)
    for number, (images, names) in enumerate(loader, 1):
        output = model(images.to(target)).float().cpu()
        vectors.append(F.normalize(output, dim=1))
        labels.extend(names)
        if number % 20 == 0:
            print(f"Embedded {min(number * batch_size, len(samples))}/{len(samples)} photos", flush=True)
    return torch.cat(vectors), labels


def accuracy(features, labels, train_ids, test_ids, head=None):
    if head is not None:
        with torch.no_grad():
            features = F.normalize(head(features), dim=1)
    by_id = defaultdict(list)
    for index, label in enumerate(labels):
        if label in test_ids:
            by_id[label].append(index)
    gallery = [indexes[0] for indexes in by_id.values()]
    # Query from the last third of each sequence; avoid adjacent frames to the
    # gallery frame when measuring retrieval on video-derived images.
    queries = [index for indexes in by_id.values()
               for index in indexes[max(1, len(indexes) * 2 // 3)::2]]
    if not gallery or not queries:
        return 0.0
    scores = features[queries] @ features[gallery].T
    predicted = [labels[gallery[index]] for index in scores.argmax(dim=1).tolist()]
    return sum(prediction == labels[index] for prediction, index in zip(predicted, queries)) / len(queries)


def train(features, labels, train_ids, steps: int):
    dimension = features.shape[1]
    head = torch.nn.Linear(dimension, dimension, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(dimension))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
    by_id = defaultdict(list)
    for index, label in enumerate(labels):
        if label in train_ids:
            by_id[label].append(index)
    rng = random.Random(42)
    usable = [label for label, indexes in by_id.items() if len(indexes) >= 2]
    for step in range(steps):
        selected = rng.sample(usable, min(8, len(usable)))
        indexes = [index for label in selected for index in rng.sample(by_id[label], min(4, len(by_id[label])))]
        sample = features[indexes]
        names = [labels[index] for index in indexes]
        vectors = F.normalize(head(sample), dim=1)
        similarities = vectors @ vectors.T
        positive = torch.tensor([[a == b and i != j for j, b in enumerate(names)]
                                 for i, a in enumerate(names)])
        negative = torch.tensor([[a != b for b in names] for a in names])
        hardest_positive = similarities.masked_fill(~positive, 2).min(dim=1).values
        hardest_negative = similarities.masked_fill(~negative, -2).max(dim=1).values
        loss = F.relu(0.15 + hardest_negative - hardest_positive).mean()
        loss += 0.01 * (head.weight - torch.eye(dimension)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{steps}, loss {loss.item():.4f}", flush=True)
    return head.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "horse30" / "labeled-data")
    parser.add_argument("--catalog", action="store_true", help="Train from labeled individuals in the service catalog")
    parser.add_argument("--limit-per-id", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--no-promote", action="store_true", help="Evaluate only; never replace the deployed head")
    args = parser.parse_args()
    if not args.catalog and not args.data.is_dir():
        parser.error(f"Dataset folder not found: {args.data}")
    samples = collect_catalog(args.limit_per_id) if args.catalog else collect(args.data, args.limit_per_id)
    identities = sorted({label for _, label in samples})
    if len(identities) < 4:
        parser.error("Need at least four labeled individuals with five photos each")
    random.Random(42).shuffle(identities)
    test_count = max(2, round(len(identities) * 0.25))
    test_ids = set(identities[:test_count])
    train_ids = set(identities[test_count:])
    print(f"Photos: {len(samples)}, train identities: {len(train_ids)}, unseen test identities: {len(test_ids)}", flush=True)
    torch.set_num_threads(4)
    model, target, _ = load_model()
    features, labels = extract(samples, model, target, args.batch_size)
    baseline = accuracy(features, labels, train_ids, test_ids)
    print(f"Baseline rank-1 on unseen identities: {baseline:.3f}", flush=True)
    head = train(features, labels, train_ids, args.steps)
    trained = accuracy(features, labels, train_ids, test_ids, head)
    print(f"Trained rank-1 on unseen identities: {trained:.3f}", flush=True)
    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate = HEAD_PATH.with_name("horse_head_candidate.pt")
    torch.save({"state_dict": head.state_dict()}, candidate)
    # A tiny gain on five held-out horses is too weak to justify deployment.
    promoted = (not args.no_promote) and trained > baseline + 0.05
    if promoted:
        HEAD_PATH.write_bytes(candidate.read_bytes())
    elif not args.no_promote:
        HEAD_PATH.unlink(missing_ok=True)
    report = {
        "source": "service catalog" if args.catalog else str(args.data), "photos": len(samples), "train_identities": len(train_ids),
        "test_identities": len(test_ids), "baseline_rank1": baseline,
        "trained_rank1": trained, "promoted": promoted,
        "note": ("Catalog validation is small and may share camera backgrounds across photos."
                 if args.catalog else
                 "Horse-30 contains domestic horses; this is not an accuracy estimate for Przewalski horses."),
    }
    (HEAD_PATH.parent / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
