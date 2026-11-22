"""Run the frozen automatic species model on the 300+ photo audit set."""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from model import ROOT, TRANSFORM, load_model
from species_quality import poor_quality

DATA = ROOT / "datasets" / "large_audit"
REPORT = ROOT / "data" / "ml" / "large_audit_report.json"
GATE = ROOT / "data" / "ml" / "species_auto.pt"
GUARD = ROOT / "data" / "ml" / "species_hard_guard.pt"


def main() -> None:
    items = json.loads((DATA / "manifest.json").read_text())
    model, target, _ = load_model()
    checkpoint = torch.load(GATE, map_location="cpu", weights_only=True)
    head = torch.nn.Linear(model.num_features, 1).eval()
    head.load_state_dict(checkpoint["state_dict"])
    guard = None
    if GUARD.exists():
        guard = torch.nn.Linear(model.num_features, 1).eval()
        guard.load_state_dict(torch.load(GUARD, map_location="cpu", weights_only=True)["state_dict"])
    details = []
    for start in range(0, len(items), 16):
        block = items[start:start + 16]
        tensors = []
        qualities = []
        for row in block:
            with Image.open(DATA / row["file"]) as image:
                qualities.append(poor_quality(image))
                tensors.append(TRANSFORM(ImageOps.exif_transpose(image).convert("RGB")))
        with torch.inference_mode():
            vectors = F.normalize(model(torch.stack(tensors).to(target)).float(), dim=1).cpu()
            scores = torch.sigmoid(head(vectors)).flatten().tolist()
            guard_scores = torch.sigmoid(guard(vectors)).flatten().tolist() if guard else [1.0] * len(block)
        for row, score, guard_score, quality in zip(block, scores, guard_scores, qualities):
            details.append({"file": row["file"], "kind": row["kind"], "label": row["label"],
                            "individual_id": row.get("individual_id"),
                            "source_group": row.get("source_group"),
                            "decision": "przewalski" if score >= 0.5 and guard_score >= 0.5 else "not_przewalski",
                            "score": round(score, 4), "guard_score": round(guard_score, 4),
                            "low_quality": quality})
        print(f"Checked {min(start + 16, len(items))}/{len(items)}", flush=True)
    by_kind = {}
    for kind in sorted({row["kind"] for row in details}):
        rows = [row for row in details if row["kind"] == kind]
        by_kind[kind] = {"count": len(rows), "called_przewalski": sum(row["decision"] == "przewalski" for row in rows)}
    false_rows = [row for row in details if (row["decision"] == "przewalski") != (row["label"] == "przewalski")]
    summary = {"total": len(details), "by_kind": by_kind,
               "false_accept": sum(row["label"] == "not_przewalski" for row in false_rows),
               "false_reject": sum(row["label"] == "przewalski" for row in false_rows),
               "domestic_individuals": len({row["individual_id"] for row in details if row["kind"] == "domestic_horse"}),
               "note": "Horse-30 frames share 20 horse identities; counts are not independent trials."}
    REPORT.write_text(json.dumps({"summary": summary, "mistakes": false_rows, "details": details}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
