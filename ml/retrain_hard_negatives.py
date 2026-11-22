"""Try hard-negative training, preserving a zebra/donkey holdout."""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from evaluate_benchmark import fit_head
from model import ROOT, TRANSFORM, load_model
from train_species_expanded import split

AUDIT = ROOT / "datasets" / "large_audit"
AUTO = ROOT / "data" / "ml" / "species_auto.pt"
GUARD = ROOT / "data" / "ml" / "species_hard_guard.pt"
REPORT = ROOT / "data" / "ml" / "hard_negative_report.json"


def embed(rows: list[dict], model: torch.nn.Module, target: torch.device) -> dict[str, torch.Tensor]:
    vectors = {}
    for start in range(0, len(rows), 16):
        block = rows[start:start + 16]
        tensors = []
        for row in block:
            with Image.open(ROOT / "datasets" / row["dataset"] / row["file"]) as image:
                tensors.append(TRANSFORM(ImageOps.exif_transpose(image).convert("RGB")))
        with torch.inference_mode():
            batch = F.normalize(model(torch.stack(tensors).to(target)).float(), dim=1).cpu()
        for row, vector in zip(block, batch):
            vectors[row["dataset"] + "/" + row["file"]] = vector
        print(f"Embedded {min(start + 16, len(rows))}/{len(rows)}", flush=True)
    return vectors


def summary(rows: list[dict], name: str) -> dict:
    return {"count": len(rows),
            "false_accept": sum(row["label"] == "not_przewalski" and row[name] == "przewalski" for row in rows),
            "false_reject": sum(row["label"] == "przewalski" and row[name] == "not_przewalski" for row in rows)}


def main() -> None:
    train, old_test = split()
    audit = [dict(row, dataset="large_audit") for row in json.loads((AUDIT / "manifest.json").read_text())]
    zebra = [row for row in audit if row["kind"] == "zebra"]
    donkey = [row for row in audit if row["kind"] == "donkey"]
    # Keep the paired Amboseli, Connochaetes, and Giraffa photo series wholly
    # on one side of the split instead of testing a neighboring frame.
    hard_train = zebra[:5] + zebra[7:10] + donkey[:7]
    hard_test = zebra[5:7] + zebra[10:] + donkey[7:]
    audit_positive = [row for row in audit if row["label"] == "przewalski"]
    domestic = [row for row in audit if row["kind"] == "domestic_horse"]
    train += hard_train
    test = old_test + hard_test + audit_positive + domestic
    model, target, _ = load_model()
    vectors = embed(train + test, model, target)
    x = torch.stack([vectors[row["dataset"] + "/" + row["file"]] for row in train])
    y = torch.tensor([float(row["label"] == "przewalski") for row in train])
    candidate = fit_head(x, y)
    previous = torch.nn.Linear(x.shape[1], 1).eval()
    previous.load_state_dict(torch.load(AUTO, map_location="cpu", weights_only=True)["state_dict"])
    details = []
    for row in test:
        vector = vectors[row["dataset"] + "/" + row["file"]]
        item = {"dataset": row["dataset"], "file": row["file"], "label": row["label"], "kind": row["kind"]}
        for name, head in (("current", previous), ("candidate", candidate)):
            with torch.inference_mode():
                score = torch.sigmoid(head(vector)).item()
            item[name] = "przewalski" if score >= 0.5 else "not_przewalski"
            item[name + "_score"] = round(score, 4)
        details.append(item)
    groups = {"previous_holdout": details[:len(old_test)],
              "new_zebra_donkey": details[len(old_test):len(old_test) + len(hard_test)],
              "new_przewalski": details[len(old_test) + len(hard_test):len(old_test) + len(hard_test) + len(audit_positive)],
              "horse30_frames": details[-len(domestic):]}
    results = {group: {name: summary(rows, name) for name in ("current", "candidate")}
               for group, rows in groups.items()}
    for group, rows in groups.items():
        joined = [dict(row, ensemble="przewalski" if row["current"] == "przewalski"
                       and row["candidate"] == "przewalski" else "not_przewalski") for row in rows]
        results[group]["ensemble"] = summary(joined, "ensemble")
    before = results["previous_holdout"]["current"]
    after = results["previous_holdout"]["candidate"]
    hard_before = results["new_zebra_donkey"]["current"]
    hard_after = results["new_zebra_donkey"]["candidate"]
    deploy = (after["false_accept"] + after["false_reject"] <= before["false_accept"] + before["false_reject"] and
              hard_after["false_accept"] < hard_before["false_accept"] and
              results["new_przewalski"]["candidate"]["false_reject"] <= results["new_przewalski"]["current"]["false_reject"] and
              results["horse30_frames"]["candidate"]["false_accept"] <= results["horse30_frames"]["current"]["false_accept"])
    ensemble_deploy = all(
        results[group]["ensemble"]["false_accept"] <= results[group]["current"]["false_accept"] and
        results[group]["ensemble"]["false_reject"] <= results[group]["current"]["false_reject"]
        for group in results
    ) and any(results[group]["ensemble"]["false_accept"] < results[group]["current"]["false_accept"]
              for group in results)
    if deploy:
        torch.save({"state_dict": candidate.state_dict(), "trained_on": len(train)}, AUTO)
        GUARD.unlink(missing_ok=True)
    elif ensemble_deploy:
        torch.save({"state_dict": candidate.state_dict(), "trained_on": len(train)}, GUARD)
    else:
        GUARD.unlink(missing_ok=True)
    report = {"train": len(train), "hard_negative_train": len(hard_train),
              "results": results, "deployed": deploy, "ensemble_deployed": ensemble_deploy and not deploy,
              "note": "Horse-30 test frames share identities with earlier training; zebra/donkey holdout is small.",
              "details": details}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
