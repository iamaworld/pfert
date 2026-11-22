"""Probe the deployed species gate on degraded held-out benchmark images."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from model import ROOT, TRANSFORM, load_model
from species_quality import poor_quality

DATA = ROOT / "datasets" / "benchmark"
GATE = ROOT / "data" / "ml" / "species_gate.pt"
REPORT = ROOT / "data" / "ml" / "stress_report.json"


def variants(image: Image.Image) -> dict[str, Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    return {
        "dark": ImageEnhance.Brightness(image).enhance(0.45),
        "blur": image.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 120)),
        "center_crop": image.crop((width * 0.15, height * 0.15, width * 0.85, height * 0.85)),
    }


def main() -> None:
    baseline = json.loads((ROOT / "data" / "ml" / "benchmark_report.json").read_text())
    checkpoint = torch.load(GATE, map_location="cpu", weights_only=True)
    model, target, _ = load_model()
    gate = torch.nn.Linear(model.num_features, 1).eval()
    gate.load_state_dict(checkpoint["state_dict"])
    details = []
    for row in baseline["details"]:
        with Image.open(DATA / row["file"]) as source:
            altered = variants(source)
        with torch.inference_mode():
            batch = torch.stack([TRANSFORM(image) for image in altered.values()]).to(target)
            vectors = F.normalize(model(batch).float(), dim=1).cpu()
            scores = torch.sigmoid(gate(vectors)).flatten().tolist()
        for name, score in zip(altered, scores):
            decision = ("przewalski" if score >= checkpoint["accept_threshold"] else
                        "not_przewalski" if score <= checkpoint["reject_threshold"] else "review")
            if poor_quality(altered[name]):
                decision = "review"
            details.append({"file": row["file"], "label": row["label"], "variant": name,
                            "score": round(score, 4), "decision": decision})
    summary = {}
    for name in ("dark", "blur", "center_crop"):
        rows = [row for row in details if row["variant"] == name]
        summary[name] = {
            "false_accept": sum(row["label"] == "not_przewalski" and row["decision"] == "przewalski" for row in rows),
            "false_reject": sum(row["label"] == "przewalski" and row["decision"] == "not_przewalski" for row in rows),
            "review": sum(row["decision"] == "review" for row in rows),
        }
    REPORT.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
