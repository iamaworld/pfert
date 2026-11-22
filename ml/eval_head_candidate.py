"""Validate an existing identity head candidate on the current catalog."""

from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F

from model import HEAD_PATH, ROOT, load_model
from train import accuracy, collect_catalog, extract

CANDIDATES = [
    HEAD_PATH.with_name("horse_head_reserve_crops_candidate.pt"),
    HEAD_PATH.with_name("horse_head_reserve_candidate.pt"),
    HEAD_PATH.with_name("horse_head_candidate.pt"),
]


def main() -> None:
    torch.set_num_threads(4)
    samples = collect_catalog(60)
    identities = sorted({label for _, label in samples})
    if len(identities) < 4:
        raise SystemExit("Need at least four labeled individuals")
    random.Random(42).shuffle(identities)
    test_count = max(2, round(len(identities) * 0.25))
    test_ids = set(identities[:test_count])
    train_ids = set(identities[test_count:])
    model, target, _ = load_model()
    features, labels = extract(samples, model, target, 8)
    baseline = accuracy(features, labels, train_ids, test_ids)
    report = {"photos": len(samples), "train_identities": len(train_ids),
              "test_identities": len(test_ids), "baseline_rank1": baseline, "candidates": {}}
    print(f"Baseline rank-1: {baseline:.3f}", flush=True)
    for candidate in CANDIDATES:
        if not candidate.is_file():
            continue
        try:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
            head = torch.nn.Linear(model.num_features, model.num_features, bias=False)
            head.load_state_dict(checkpoint["state_dict"])
            trained = accuracy(features, labels, train_ids, test_ids, head)
            report["candidates"][candidate.name] = trained
            print(f"{candidate.name}: {trained:.3f}", flush=True)
        except Exception as error:
            report["candidates"][candidate.name] = f"error: {error}"
            print(f"{candidate.name}: error {error}", flush=True)
    best = max((value for value in report["candidates"].values() if isinstance(value, float)), default=None)
    promote = None
    if best is not None and best > baseline + 0.02:
        promote = [name for name, value in report["candidates"].items() if value == best][0]
        source = HEAD_PATH.parent / promote
        HEAD_PATH.write_bytes(source.read_bytes())
        report["promoted"] = promote
        print(f"Promoted {promote}", flush=True)
    else:
        HEAD_PATH.unlink(missing_ok=True)
        report["promoted"] = None
    (HEAD_PATH.parent / "identity_head_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
