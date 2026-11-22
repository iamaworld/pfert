"""Try a stronger species guard using new domestic scenes; keep whole scenes held out."""

from __future__ import annotations

import collections
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F

from build_extra_audit import SCENES
from model import ROOT, load_model
from retrain_hard_negatives import embed
from train_species_expanded import split


DATA = ROOT / "datasets"
MODEL_DIR = ROOT / "data" / "ml"
AUTO = MODEL_DIR / "species_auto.pt"
GUARD = MODEL_DIR / "species_hard_guard.pt"
CANDIDATE = MODEL_DIR / "species_extra_candidate.pt"
REPORT = MODEL_DIR / "extra_training_report.json"


def head(path, features: int) -> torch.nn.Linear:
    model = torch.nn.Linear(features, 1).eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    return model


def fit(x: torch.Tensor, y: torch.Tensor, weight: float,
        anchor: torch.nn.Linear | None = None) -> torch.nn.Linear:
    torch.manual_seed(42)
    candidate = torch.nn.Linear(x.shape[1], 1)
    if anchor is not None:
        candidate.load_state_dict(anchor.state_dict())
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.01, weight_decay=0.1)
    positives = y.sum().item()
    negatives = len(y) - positives
    positive_weight = weight * negatives / positives
    for _ in range(300):
        logits = candidate(x).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, y,
                                                   pos_weight=torch.tensor(positive_weight))
        if anchor is not None:
            loss += 0.025 * sum((param - original).square().sum()
                                for param, original in zip(candidate.parameters(), anchor.parameters()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return candidate.eval()


def metrics(rows: list[dict], key: str) -> dict:
    false_accept = sum(row["label"] != "przewalski" and row[key] for row in rows)
    false_reject = sum(row["label"] == "przewalski" and not row[key] for row in rows)
    return {"count": len(rows), "false_accept": false_accept, "false_reject": false_reject,
            "errors": false_accept + false_reject}


def main() -> None:
    old_train, old_test = split()
    audit = [dict(row, dataset="large_audit") for row in
             json.loads((DATA / "large_audit" / "manifest.json").read_text())]
    extra = [dict(row, dataset="extra_audit") for row in
             json.loads((DATA / "extra_audit" / "manifest.json").read_text())]
    if len(extra) != 300 or {row["source_scene"] for row in extra} != set(SCENES):
        raise ValueError("The extra audit must contain 300 photos from ten scenes")
    previous = json.loads((MODEL_DIR / "large_training_report.json").read_text())
    held_zoos = set(previous["held_out_zoos"])
    positives = [row for row in audit if row["label"] == "przewalski"]
    positive_train = [row for row in positives if row["source_group"] not in held_zoos]
    positive_test = [row for row in positives if row["source_group"] in held_zoos]
    domestic = collections.defaultdict(list)
    for row in audit:
        if row["kind"] == "domestic_horse":
            domestic[row["individual_id"]].append(row)
    domestic_train = [row for number in range(1, 17) for row in domestic[f"Sample{number}"][:3]]
    domestic_test = [row for number in range(17, 21) for row in domestic[f"Sample{number}"]]
    zebra = [row for row in audit if row["kind"] == "zebra"]
    donkey = [row for row in audit if row["kind"] == "donkey"]
    hard_train = zebra[:5] + zebra[7:10] + donkey[:7]
    hard_test = zebra[5:7] + zebra[10:] + donkey[7:]
    extra_train = [row for row in extra if row["source_scene"] in SCENES[:5]]
    extra_test = [row for row in extra if row["source_scene"] in SCENES[5:]]
    train = old_train + positive_train + domestic_train + hard_train + extra_train
    test_groups = {"old_holdout": old_test, "zoo_holdout": positive_test,
                   "horse_identity_holdout": domestic_test,
                   "zebra_donkey_holdout": hard_test,
                   "new_scene_holdout": extra_test}
    test = [row for group in test_groups.values() for row in group]
    model, target, _ = load_model()
    vectors = embed(train + test, model, target)
    keys = lambda rows: [row["dataset"] + "/" + row["file"] for row in rows]
    x = torch.stack([vectors[key] for key in keys(train)])
    y = torch.tensor([float(row["label"] == "przewalski") for row in train])
    z = torch.stack([vectors[key] for key in keys(test)])
    primary = head(AUTO, x.shape[1])
    deployed_guard = head(GUARD, x.shape[1])
    variants = {"unweighted": fit(x, y, y.sum().item() / (len(y) - y.sum().item())),
                "balanced": fit(x, y, 1.0),
                "anchored": fit(x, y, 0.65, deployed_guard)}
    with torch.inference_mode():
        primary_scores = torch.sigmoid(primary(z)).flatten()
        guard_scores = torch.sigmoid(deployed_guard(z)).flatten()
        current = (primary_scores >= 0.5) & (guard_scores >= 0.5)
        choices = {}
        for name, candidate in variants.items():
            candidate_scores = torch.sigmoid(candidate(z)).flatten()
            choices[name + "_guard"] = (primary_scores >= 0.5) & (candidate_scores >= 0.5)
            choices[name + "_primary"] = (candidate_scores >= 0.5) & (guard_scores >= 0.5)
            choices[name + "_alone"] = candidate_scores >= 0.5
    details = []
    for index, row in enumerate(test):
        details.append({"group": next(group for group, rows in test_groups.items() if row in rows),
                        "dataset": row["dataset"], "file": row["file"],
                        "label": row["label"], "source_group": row.get("source_group"),
                        "current": bool(current[index]),
                        **{name: bool(scores[index]) for name, scores in choices.items()}})
    names = ["current", *choices]
    by_group = {group: {name: metrics([row for row in details if row["group"] == group], name)
                        for name in names} for group in test_groups}
    totals = {name: metrics(details, name) for name in names}
    baseline = totals["current"]
    eligible = [name for name in choices if
                totals[name]["errors"] < baseline["errors"] and
                totals[name]["false_reject"] <= baseline["false_reject"] and
                all(by_group[group][name]["false_accept"] <= by_group[group]["current"]["false_accept"]
                    for group in test_groups)]
    selected = min(eligible, key=lambda name: totals[name]["errors"]) if eligible else None
    if selected:
        variant_name, role = selected.rsplit("_", 1)
        torch.save({"state_dict": variants[variant_name].state_dict(), "trained_on": len(train),
                    "role": role}, CANDIDATE)
    else:
        CANDIDATE.unlink(missing_ok=True)
    report = {"train": len(train), "test": len(test), "new_photos": len(extra),
              "new_scene_train": sorted(SCENES[:5]), "new_scene_test": sorted(SCENES[5:]),
              "totals": totals, "by_group": by_group, "selected_candidate": selected,
              "deployed": False,
              "note": "New photos are correlated frames from ten Horse-30 videos. The positive holdout is reused from previous model selection; do not treat it as a fresh external test.",
              "details": details}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
