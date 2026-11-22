"""Train on expanded photos with zoo and horse identity groups held out."""

from __future__ import annotations

import collections
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from evaluate_benchmark import fit_head
from model import ROOT, load_model
from retrain_hard_negatives import embed, summary
from train_species_expanded import split

AUDIT = ROOT / "datasets" / "large_audit" / "manifest.json"
AUTO = ROOT / "data" / "ml" / "species_auto.pt"
CANDIDATE = ROOT / "data" / "ml" / "species_large_candidate.pt"
MULTICLASS = ROOT / "data" / "ml" / "species_multiclass_candidate.pt"
REPORT = ROOT / "data" / "ml" / "large_training_report.json"


def main() -> None:
    old_train, old_test = split()
    audit = [dict(row, dataset="large_audit") for row in json.loads(AUDIT.read_text())]
    positive = [row for row in audit if row["label"] == "przewalski"]
    groups = sorted({row["source_group"] for row in positive})
    held_groups = {name for index, name in enumerate(groups) if index % 4 == 0}
    positive_train = [row for row in positive if row["source_group"] not in held_groups]
    positive_test = [row for row in positive if row["source_group"] in held_groups]
    domestic = [row for row in audit if row["kind"] == "domestic_horse"]
    domestic_train = []
    domestic_test = []
    by_id = collections.defaultdict(list)
    for row in domestic:
        by_id[row["individual_id"]].append(row)
    for number in range(1, 21):
        rows = by_id[f"Sample{number}"]
        if number <= 16:
            domestic_train.extend(rows[:3])
        else:
            domestic_test.extend(rows)
    zebra = [row for row in audit if row["kind"] == "zebra"]
    donkey = [row for row in audit if row["kind"] == "donkey"]
    hard_train = zebra[:5] + zebra[7:10] + donkey[:7]
    hard_test = zebra[5:7] + zebra[10:] + donkey[7:]
    train = old_train + positive_train + domestic_train + hard_train
    test_groups = {"old_holdout": old_test, "zoo_holdout": positive_test,
                   "horse_identity_holdout": domestic_test, "zebra_donkey_holdout": hard_test}
    test = [row for rows in test_groups.values() for row in rows]
    model, target, _ = load_model()
    vectors = embed(train + test, model, target)
    x = torch.stack([vectors[row["dataset"] + "/" + row["file"]] for row in train])
    y = torch.tensor([float(row["label"] == "przewalski") for row in train])
    fitted = fit_head(x, y)
    classes = torch.tensor([0 if row["label"] == "przewalski" else
                            1 if row["kind"] == "domestic_horse" else 2 for row in train])
    counts = torch.bincount(classes, minlength=3).float()
    torch.manual_seed(42)
    multiclass = torch.nn.Linear(x.shape[1], 3)
    optimizer = torch.optim.AdamW(multiclass.parameters(), lr=0.01, weight_decay=0.1)
    weights = len(train) / (3 * counts)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(multiclass(x), classes, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    multiclass.eval()
    baseline = torch.nn.Linear(x.shape[1], 1).eval()
    baseline.load_state_dict(torch.load(AUTO, map_location="cpu", weights_only=True)["state_dict"])
    details = []
    for group_name, rows in test_groups.items():
        for row in rows:
            vector = vectors[row["dataset"] + "/" + row["file"]]
            item = {"group": group_name, "dataset": row["dataset"], "file": row["file"],
                    "label": row["label"], "source_group": row.get("source_group")}
            for name, head in (("current", baseline), ("candidate", fitted)):
                with torch.inference_mode():
                    score = torch.sigmoid(head(vector)).item()
                item[name] = "przewalski" if score >= 0.5 else "not_przewalski"
                item[name + "_score"] = round(score, 4)
            with torch.inference_mode():
                class_scores = torch.softmax(multiclass(vector), dim=0)
            item["multiclass"] = "przewalski" if class_scores.argmax().item() == 0 else "not_przewalski"
            item["multiclass_score"] = round(class_scores[0].item(), 4)
            details.append(item)
    results = {name: {model_name: summary([row for row in details if row["group"] == name], model_name)
                      for model_name in ("current", "candidate", "multiclass")}
               for name in test_groups}
    torch.save({"state_dict": fitted.state_dict(), "trained_on": len(train)}, CANDIDATE)
    torch.save({"state_dict": multiclass.state_dict(), "trained_on": len(train)}, MULTICLASS)
    report = {"train": len(train), "test": len(test), "positive_train": len(positive_train),
              "positive_test": len(positive_test), "held_out_zoos": sorted(held_groups),
              "results": results, "promoted": False,
              "note": "Group split protects new zoo/horse identities, but old training may share broader source context.",
              "details": details}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
