"""Check species and return cosine similarities for catalog photos, as JSON."""

from __future__ import annotations

import json
import os
import sys
import torch
from PIL import Image

from model import MODEL_ID, ROOT, embed_image, load_model
from species_quality import poor_quality


GATE_PATH = ROOT / "data" / "ml" / "species_gate.pt"
AUTO_PATH = ROOT / "data" / "ml" / "species_auto.pt"
GUARD_PATH = ROOT / "data" / "ml" / "species_hard_guard.pt"
CACHE_PATH = ROOT / "data" / "ml" / f"reference_cache_{MODEL_ID.rsplit('/', 1)[-1]}.pt"


def signature(path: str) -> str:
    stat = os.stat(path)
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    except Exception:
        return {}


def check_species(query: str, model: torch.nn.Module, target: torch.device) -> dict:
    if not GATE_PATH.exists():
        return {"decision": "unavailable"}
    checkpoint = torch.load(AUTO_PATH if AUTO_PATH.exists() else GATE_PATH,
                            map_location="cpu", weights_only=True)
    gate = torch.nn.Linear(model.num_features, 1).eval()
    gate.load_state_dict(checkpoint["state_dict"])
    # The gate was trained on the whole-image backbone vector, before the
    # optional identity head. Keep this preprocessing identical at inference.
    vectors = embed_image(query, model, target, query=False)
    with torch.inference_mode():
        score = torch.sigmoid(gate(vectors))[0].item()
    # Binary cutoff. The checkpoint's accept_threshold was calibrated for a
    # three-way (accept/review/reject) mode; using it as a binary cutoff was
    # too strict (7 false rejects on 57 photos). A sweep over benchmark_new
    # showed 0.5 keeps false accepts at zero with one false reject.
    accept_threshold = float(os.environ.get("TAKHI_SPECIES_THRESHOLD", 0.5))
    decision = "przewalski" if score >= accept_threshold else "not_przewalski"
    guard_score = None
    if decision == "przewalski" and GUARD_PATH.exists():
        guard_checkpoint = torch.load(GUARD_PATH, map_location="cpu", weights_only=True)
        guard = torch.nn.Linear(model.num_features, 1).eval()
        guard.load_state_dict(guard_checkpoint["state_dict"])
        with torch.inference_mode():
            guard_score = torch.sigmoid(guard(vectors))[0].item()
        if guard_score < 0.5:
            decision = "not_przewalski"
    with Image.open(query) as image:
        low_quality = poor_quality(image)
    result = {"decision": decision, "score": round(score, 4),
              "lowQuality": low_quality}
    if guard_score is not None:
        result["guardScore"] = round(guard_score, 4)
    return result


def photo_similarity(query_vectors: torch.Tensor, reference_vectors: torch.Tensor) -> float:
    scores = (query_vectors @ reference_vectors.T).flatten()
    if scores.numel() > 1:
        top = torch.topk(scores, min(2, scores.numel())).values
        return round(top.mean().item(), 5)
    return round(scores.max().item(), 5)


def main() -> None:
    query, *references = sys.argv[1:]
    model, target, head = load_model()
    species = check_species(query, model, target)
    if not references:
        print(json.dumps({"species": species, "matches": []}))
        return
    query_vectors = embed_image(query, model, target, head, query=True)
    cache = load_cache()
    results = []
    dirty = False
    for index, path in enumerate(references):
        try:
            key = signature(path)
            reference_vectors = cache.get(key)
            if reference_vectors is None:
                reference_vectors = embed_image(path, model, target, head)
                cache[key] = reference_vectors
                dirty = True
            score = photo_similarity(query_vectors, reference_vectors)
            results.append({"index": index, "similarity": score})
        except Exception as error:
            print(f"Skipped {path}: {error}", file=sys.stderr)
            results.append({"index": index, "similarity": None})
    if dirty:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, CACHE_PATH)
    print(json.dumps({"species": species, "matches": results}))


if __name__ == "__main__":
    main()

