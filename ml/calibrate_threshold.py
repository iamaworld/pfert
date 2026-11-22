"""Sweep species thresholds on benchmark_new using the deployed gate."""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from model import ROOT, embed_image, load_model
from species_quality import poor_quality

DATA = ROOT / "datasets" / "benchmark_new"
AUTO = ROOT / "data" / "ml" / "species_auto.pt"
GATE = ROOT / "data" / "ml" / "species_gate.pt"
GUARD = ROOT / "data" / "ml" / "species_hard_guard.pt"


def main() -> None:
    torch.set_num_threads(4)
    model, target, _ = load_model()
    checkpoint = torch.load(AUTO if AUTO.exists() else GATE, map_location="cpu", weights_only=True)
    gate = torch.nn.Linear(model.num_features, 1).eval()
    gate.load_state_dict(checkpoint["state_dict"])
    guard = None
    if GUARD.exists():
        guard = torch.nn.Linear(model.num_features, 1).eval()
        guard.load_state_dict(torch.load(GUARD, map_location="cpu", weights_only=True)["state_dict"])

    items = json.loads((DATA / "manifest.json").read_text())
    scores = []
    labels = []
    for index, item in enumerate(items, 1):
        vectors = embed_image(str(DATA / item["file"]), model, target, query=False)
        with torch.inference_mode():
            score = torch.sigmoid(gate(vectors))[0].item()
        scores.append(score)
        labels.append(item["label"] == "przewalski")
        if index % 10 == 0:
            print(f"Embedded {index}/{len(items)}", flush=True)

    rows = []
    best = None
    for threshold in [round(0.5 + step * 0.025, 3) for step in range(20)]:
        false_accept = false_reject = 0
        for score, positive in zip(scores, labels):
            decision = score >= threshold
            false_accept += bool(decision and not positive)
            false_reject += bool(not decision and positive)
        errors = false_accept + false_reject
        rows.append({"threshold": threshold, "false_accept": false_accept,
                     "false_reject": false_reject, "errors": errors})
        if best is None or errors < best["errors"]:
            best = rows[-1]
    report = {"photos": len(items), "positive": sum(labels), "negative": len(labels) - sum(labels),
              "deployed_threshold": float(checkpoint.get("accept_threshold", 0.5)),
              "sweep": rows, "best_on_this_set": best}
    (ROOT / "data" / "ml" / "threshold_sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
