"""Build and evaluate a local identity benchmark from labelled Horse-30 frames."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from model import ROOT, embed_image, load_model
from search import photo_similarity


def build_manifest() -> list[dict]:
    source = ROOT / "datasets" / "horse30"
    rows = []
    hashes = set()
    for number in range(1, 16):
        identity = f"Sample{number}"
        scene = source / "labeled-data" / identity
        with next(scene.glob("CollectedData_*.csv")).open(newline="") as handle:
            frames = sorted(Path(row[0]).name for row in list(csv.reader(handle))[3:]
                            if sum(bool(x and y) for x, y in zip(row[1::2], row[2::2])) >= 18)
        if len(frames) < 30:
            raise ValueError(f"Too few annotated frames for {identity}")
        for role, positions in (("reference", (.12, .23, .34)), ("query", (.63, .73, .83, .93))):
            for position in positions:
                file = scene / frames[int(position * len(frames))]
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f"Duplicate frame: {file}")
                hashes.add(digest)
                rows.append({"horse_id": identity, "role": role, "file": file.relative_to(ROOT).as_posix(),
                             "sha256": digest, "source": "https://huggingface.co/datasets/mwmathis/Horse-30",
                             "license": "CC BY-NC-SA 4.0"})
    return rows


def main() -> None:
    rows = build_manifest()
    output = ROOT / "data" / "ml"
    output.mkdir(parents=True, exist_ok=True)
    (output / "identity_demo_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    model, device, head = load_model()
    references = [(row, embed_image(ROOT / row["file"], model, device, head))
                  for row in rows if row["role"] == "reference"]
    details = []
    for row in rows:
        if row["role"] != "query":
            continue
        query = embed_image(ROOT / row["file"], model, device, head, query=True)
        best = {}
        for reference, vectors in references:
            identity = reference["horse_id"]
            score = photo_similarity(query, vectors)
            if identity not in best or score > best[identity]["similarity"]:
                best[identity] = {"horse_id": identity, "similarity": score, "file": reference["file"]}
        candidates = sorted(best.values(), key=lambda item: item["similarity"], reverse=True)[:5]
        details.append({"file": row["file"], "expected": row["horse_id"], "candidates": candidates})
    summary = {"identities": 15, "reference_photos": len(references), "queries": len(details)}
    for k in (1, 3, 5):
        summary[f"top_{k}_hits"] = sum(any(item["horse_id"] == row["expected"] for item in row["candidates"][:k])
                                        for row in details)
    report = {"summary": summary,
              "limitations": "Domestic horses, one series per ID, early references and later queries. "
                             "This checks the catalog matcher in isolation from the species classifier.",
              "details": details}
    (output / "identity_demo_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
