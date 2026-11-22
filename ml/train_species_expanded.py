"""Train an expanded species head and compare it with the deployed gate."""

from __future__ import annotations

import hashlib
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image

from evaluate_benchmark import fit_head
from model import ROOT, embed_image, load_model
from species_quality import poor_quality

GATE = ROOT / "data" / "ml" / "species_gate.pt"
VETO = ROOT / "data" / "ml" / "species_veto.pt"
AUTO = ROOT / "data" / "ml" / "species_auto.pt"
REPORT = ROOT / "data" / "ml" / "expanded_training_report.json"


def split() -> tuple[list[dict], list[dict]]:
    old = json.loads((ROOT / "datasets" / "benchmark" / "manifest.json").read_text())
    new = json.loads((ROOT / "datasets" / "benchmark_new" / "manifest.json").read_text())
    op = [x for x in old if x["label"] == "przewalski"]
    on = [x for x in old if x["label"] != "przewalski"]
    np = [x for x in new if x["label"] == "przewalski"]
    nn = [x for x in new if x["label"] != "przewalski"]
    train_old = op[:15] + on[:10] + on[20:23] + on[25:27]
    # Keep the original 30-photo holdout untouched. New negative identities are
    # partitioned as complete three-frame groups, not individual frames.
    train_new = np[:20] + nn[:18]
    train = [dict(row, dataset="benchmark") for row in train_old] + [dict(row, dataset="benchmark_new") for row in train_new]
    train_ids = {row["individual_id"] for row in nn[:18] if row.get("individual_id")}
    test = [dict(row, dataset="benchmark") for row in old
            if row not in train_old and row.get("individual_id") not in train_ids]
    test += [dict(row, dataset="benchmark_new") for row in np[20:] + nn[18:]]
    return train, test


def group(row: dict) -> str:
    if row.get("individual_id"):
        return row["dataset"] + ":" + row["individual_id"]
    title = row["title"].lower()
    title = re.sub(r"\([^)]*\)|\d+|[^a-zа-я]+", " ", title)
    return " ".join(title.split()[:4]) or row["title"]


def decision(scores: torch.Tensor, checkpoint: dict, quality: bool) -> str:
    whole = scores[0].item()
    if whole >= checkpoint["accept_threshold"]:
        result = "przewalski"
    elif whole <= checkpoint["reject_threshold"]:
        result = "not_przewalski"
    else:
        result = "review"
    if result == "not_przewalski" and scores[1:].max().item() > checkpoint["reject_threshold"]:
        result = "review"
    return "review" if quality else result


def metrics(rows: list[dict]) -> dict:
    return {"false_accept": sum(x["label"] == "not_przewalski" and x["decision"] == "przewalski" for x in rows),
            "false_reject": sum(x["label"] == "przewalski" and x["decision"] == "not_przewalski" for x in rows),
            "review": sum(x["decision"] == "review" for x in rows)}


def main() -> None:
    train, test = split()
    model, target, _ = load_model()
    current = torch.load(GATE, map_location="cpu", weights_only=True)
    views = {}
    quality = {}
    for index, row in enumerate(train + test, 1):
        key = row["dataset"] + "/" + row["file"]
        path = ROOT / "datasets" / key
        views[key] = embed_image(path, model, target, query=True)
        with Image.open(path) as image:
            quality[key] = poor_quality(image)
        if index % 20 == 0:
            print(f"Embedded {index}/{len(train) + len(test)}", flush=True)
    x = torch.stack([views[row["dataset"] + "/" + row["file"]][0] for row in train])
    y = torch.tensor([float(row["label"] == "przewalski") for row in train])
    groups = [group(row) for row in train]
    folds = [int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 5 for value in groups]
    oof = [0.0] * len(train)
    for fold in range(5):
        held = [i for i, number in enumerate(folds) if number == fold]
        fit = [i for i, number in enumerate(folds) if number != fold]
        head = fit_head(x[fit], y[fit])
        with torch.inference_mode():
            scores = torch.sigmoid(head(x[held]).flatten()).tolist()
        for index, score in zip(held, scores):
            oof[index] = score
    candidate = {"accept_threshold": min(1.0, max(0.5, max(oof[i] for i in range(len(y)) if y[i] == 0) + 0.05)),
                 "reject_threshold": max(0.0, min(0.5, min(oof[i] for i in range(len(y)) if y[i] == 1) - 0.05))}
    head = fit_head(x, y)
    candidate["state_dict"] = head.state_dict()
    old_head = torch.nn.Linear(x.shape[1], 1).eval()
    old_head.load_state_dict(current["state_dict"])
    details = []
    for row in test:
        key = row["dataset"] + "/" + row["file"]
        item = {"dataset": row["dataset"], "file": row["file"], "label": row["label"]}
        for name, checkpoint, fitted in (("current", current, old_head), ("candidate", candidate, head)):
            with torch.inference_mode():
                scores = torch.sigmoid(fitted(views[key])).flatten()
            item[name] = decision(scores, checkpoint, quality[key])
            item[name + "_score"] = round(scores[0].item(), 4)
        details.append(item)
    old_metrics = metrics([{"label": row["label"], "decision": row["current"]} for row in details])
    new_metrics = metrics([{"label": row["label"], "decision": row["candidate"]} for row in details])
    veto_rows = [{"label": row["label"], "decision":
                  "review" if row["current"] == "not_przewalski" and row["candidate"] != "not_przewalski"
                  else row["current"]} for row in details]
    veto_metrics = metrics(veto_rows)
    current_binary_errors = sum((row["current_score"] >= 0.5) != (row["label"] == "przewalski") for row in details)
    candidate_binary_errors = sum((row["candidate_score"] >= 0.5) != (row["label"] == "przewalski") for row in details)
    deploy_auto = candidate_binary_errors < current_binary_errors
    deploy = (new_metrics["false_accept"] <= old_metrics["false_accept"] and
              new_metrics["false_reject"] < old_metrics["false_reject"] and
              new_metrics["review"] <= old_metrics["review"] + 10)
    deploy_veto = (veto_metrics["false_accept"] <= old_metrics["false_accept"] and
                   veto_metrics["false_reject"] < old_metrics["false_reject"] and
                   veto_metrics["review"] <= old_metrics["review"] + 10)
    if deploy:
        torch.save(candidate, GATE)
        VETO.unlink(missing_ok=True)
    elif deploy_veto:
        torch.save(candidate, VETO)
    if deploy_auto:
        torch.save(candidate, AUTO)
    report = {"train": len(train), "test": len(test), "current": old_metrics,
              "candidate": new_metrics, "veto": veto_metrics,
              "candidate_thresholds": {key: candidate[key] for key in ("accept_threshold", "reject_threshold")},
              "deployed": deploy, "veto_deployed": deploy_veto,
              "auto_binary_errors": {"current": current_binary_errors, "candidate": candidate_binary_errors},
              "auto_deployed": deploy_auto,
              "note": "Exploratory split; some Przewalski photo series may overlap across train and test.",
              "details": details}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
