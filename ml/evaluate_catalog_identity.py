"""Closed-set identity accuracy on the current catalog.

Gallery = first photo of each horse. Queries = every catalog photo.
Reports top-1/3/5 hit rates and mean rank, plus per-horse top-1.
"""
from __future__ import annotations

import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from model import ROOT, load_model
from train import collect_catalog, extract


def main() -> None:
    torch.set_num_threads(4)
    samples = collect_catalog(60)
    if not samples:
        raise SystemExit("Catalog is empty")
    model, target, _ = load_model()
    features, labels = extract(samples, model, target, 8)
    features = F.normalize(features, dim=1)

    gallery = []
    gallery_labels = []
    seen = set()
    for index, label in enumerate(labels):
        if label not in seen:
            seen.add(label)
            gallery.append(index)
            gallery_labels.append(label)

    scores = features @ features[gallery].T
    order = scores.argsort(dim=1, descending=True).tolist()
    ranks = []
    per_horse = defaultdict(list)
    for index, label in enumerate(labels):
        predicted = order[index]
        gallery_pos = gallery_labels.index(label)
        rank = predicted.index(gallery_pos)
        ranks.append(rank)
        per_horse[label].append(rank)

    hits = {k: sum(rank < k for rank in ranks) / len(ranks) for k in (1, 3, 5)}
    report = {
        "photos": len(samples),
        "identities": len(gallery),
        "queries": len(ranks),
        "top1": round(hits[1], 4),
        "top3": round(hits[3], 4),
        "top5": round(hits[5], 4),
        "mean_rank": round(sum(ranks) / len(ranks), 2),
        "per_horse_top1": {horse: round(sum(r == 0 for r in ranks_list) / len(ranks_list), 3)
                           for horse, ranks_list in sorted(per_horse.items())},
    }
    (ROOT / "data" / "ml" / "catalog_identity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
