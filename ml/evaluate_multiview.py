"""Evaluate cropped-image species decisions on the grouped development holdout."""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image

from model import ROOT, TRANSFORM, crop_views, load_model


MODEL_DIR = ROOT / "data" / "ml"


def head(name: str, features: int) -> torch.nn.Linear:
    result = torch.nn.Linear(features, 1).eval()
    result.load_state_dict(torch.load(MODEL_DIR / name, map_location="cpu", weights_only=True)["state_dict"])
    return result


def summary(rows: list[dict], key: str) -> dict:
    false_accept = sum(row["label"] != "przewalski" and row[key] for row in rows)
    false_reject = sum(row["label"] == "przewalski" and not row[key] for row in rows)
    return {"count": len(rows), "false_accept": false_accept, "false_reject": false_reject,
            "errors": false_accept + false_reject}


def main() -> None:
    rows = json.loads((MODEL_DIR / "extra_training_report.json").read_text())["details"]
    model, device, _ = load_model()
    primary = head("species_auto.pt", model.num_features)
    guard = head("species_hard_guard.pt", model.num_features)
    details = []
    for start in range(0, len(rows), 16):
        block = rows[start:start + 16]
        tensors = []
        for row in block:
            with Image.open(ROOT / "datasets" / row["dataset"] / row["file"]) as image:
                tensors.extend(TRANSFORM(view) for view in crop_views(image, query=True))
        with torch.inference_mode():
            raw = model(torch.stack(tensors).to(device)).float()
            vectors = torch.nn.functional.normalize(raw, dim=1).cpu()
            primary_scores = torch.sigmoid(primary(vectors)).reshape(len(block), 6)
            guard_scores = torch.sigmoid(guard(vectors)).reshape(len(block), 6)
            approved = (primary_scores >= 0.5) & (guard_scores >= 0.5)
        for index, row in enumerate(block):
            views = approved[index].tolist()
            details.append({"group": row["group"], "dataset": row["dataset"],
                            "file": row["file"], "label": row["label"],
                            "whole": views[0], "center_or_whole": any(views[:2]),
                            "any_view": any(views), "two_views": sum(views) >= 2,
                            "view_votes": sum(views),
                            "primary_scores": [round(value, 4) for value in primary_scores[index].tolist()],
                            "guard_scores": [round(value, 4) for value in guard_scores[index].tolist()]})
        print(f"Checked {min(start + 16, len(rows))}/{len(rows)}", flush=True)
    names = ["whole", "center_or_whole", "any_view", "two_views"]
    groups = sorted({row["group"] for row in details})
    report = {"count": len(details),
              "totals": {name: summary(details, name) for name in names},
              "by_group": {group: {name: summary([row for row in details if row["group"] == group], name)
                                   for name in names} for group in groups},
              "note": "This grouped development holdout was used for prior model selection; crop rules need fresh validation.",
              "details": details}
    (MODEL_DIR / "multiview_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
