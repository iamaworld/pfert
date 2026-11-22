"""Measure whether MegaDescriptor embeddings separate target species from hard negatives."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

from model import ROOT, embed_image, load_model


DATA = ROOT / "datasets" / "benchmark"
REPORT = ROOT / "data" / "ml" / "benchmark_report.json"
GATE = ROOT / "data" / "ml" / "species_gate.pt"


def fit_head(x: torch.Tensor, y: torch.Tensor) -> torch.nn.Linear:
    torch.manual_seed(42)
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.1)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(head(x).flatten(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head.eval()


def main() -> None:
    items = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    positives = [item for item in items if item["label"] == "przewalski"]
    negatives = [item for item in items if item["label"] == "not_przewalski"]
    if len(positives) != 30 or len(negatives) != 30:
        raise ValueError("Benchmark needs 30 positive and 30 negative photos")
    # Keep the entire second half of target photos out of training. Negatives
    # contain independent Horse-30 individuals and separate Commons examples.
    train_items = positives[:15] + negatives[:10] + negatives[20:23] + negatives[25:27]
    test_items = [item for item in items if item not in train_items]
    model, target, _ = load_model()
    vectors = {}
    for index, item in enumerate(items, 1):
        vectors[item["file"]] = embed_image(DATA / item["file"], model, target)[0]
        if index % 10 == 0:
            print(f"Embedded {index}/60", flush=True)
    x = torch.stack([vectors[item["file"]] for item in train_items])
    y = torch.tensor([float(item["label"] == "przewalski") for item in train_items])
    # Calibrate reject/accept bands using predictions for photos that were left
    # out of each training fold. The 30 final test photos remain untouched.
    oof = [0.0] * len(train_items)
    for fold in range(5):
        held = [i for i in range(len(train_items)) if i % 15 % 5 == fold]
        fit = [i for i in range(len(train_items)) if i not in held]
        fold_head = fit_head(x[fit], y[fit])
        with torch.no_grad():
            probabilities = torch.sigmoid(fold_head(x[held]).flatten()).tolist()
        for index, probability in zip(held, probabilities):
            oof[index] = probability
    accept_threshold = max(0.5, max(oof[i] for i in range(len(y)) if y[i] == 0) + 0.05)
    reject_threshold = min(0.5, min(oof[i] for i in range(len(y)) if y[i] == 1) - 0.05)
    accept_threshold = min(1.0, accept_threshold)
    reject_threshold = max(0.0, reject_threshold)
    head = fit_head(x, y)
    with torch.no_grad():
        scores = torch.sigmoid(head(torch.stack([vectors[item["file"]] for item in test_items])).flatten()).tolist()
    def decision(score):
        if score >= accept_threshold:
            return "przewalski"
        if score <= reject_threshold:
            return "not_przewalski"
        return "review"

    details = [{"file": item["file"], "label": item["label"], "score": round(score, 4),
                "predicted": "przewalski" if score >= 0.5 else "not_przewalski",
                "decision": decision(score)}
               for item, score in zip(test_items, scores)]
    tp = sum(row["label"] == "przewalski" and row["predicted"] == "przewalski" for row in details)
    fn = sum(row["label"] == "przewalski" and row["predicted"] != "przewalski" for row in details)
    tn = sum(row["label"] == "not_przewalski" and row["predicted"] == "not_przewalski" for row in details)
    fp = sum(row["label"] == "not_przewalski" and row["predicted"] != "not_przewalski" for row in details)
    false_accept = sum(row["label"] == "not_przewalski" and row["decision"] == "przewalski" for row in details)
    false_reject = sum(row["label"] == "przewalski" and row["decision"] == "not_przewalski" for row in details)
    review_count = sum(row["decision"] == "review" for row in details)
    deployed = false_accept == 0 and false_reject == 0 and review_count <= 15
    report = {"train_photos": len(train_items), "test_photos": len(test_items),
              "true_positive": tp, "false_negative": fn, "true_negative": tn, "false_positive": fp,
              "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
              "accept_threshold": accept_threshold, "reject_threshold": reject_threshold,
              "false_accept": false_accept, "false_reject": false_reject,
              "review_count": review_count, "deployed": deployed,
              "note": "Small, source-specific test set; any gate requires manual review of uncertain photos.",
              "details": details}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if deployed:
        torch.save({"state_dict": head.state_dict(), "accept_threshold": accept_threshold,
                    "reject_threshold": reject_threshold}, GATE)
    else:
        GATE.unlink(missing_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
