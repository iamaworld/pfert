"""Evaluate and train an identity head from the reserve archive, without extracting it."""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path, PurePosixPath

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from detect_horses import detect, load_detector
from model import ROOT, TRANSFORM, crop_views, load_model
from train import train


def inventory(archive):
    groups = defaultdict(list)
    skipped = Counter()
    entries = {}
    for member in archive.infolist():
        name = PurePosixPath(member.filename)
        if len(name.parts) != 3 or name.parts[0] != "Фотобанк_Хакатон" or name.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        horse = re.sub(r"\s*\(\d+ фото\)$", "", name.parts[1])
        entries.setdefault((member.CRC, member.file_size), []).append((horse, member))
    for copies in entries.values():
        if len({horse for horse, _ in copies}) > 1:
            skipped["conflicting_duplicates"] += len(copies)
            continue
        for horse, member in copies:
            try:
                with archive.open(member) as file, Image.open(file) as image:
                    exif = image.getexif()
                    stamp = exif.get_ifd(34665).get(36867) or exif.get(36867)
                    day = date.fromisoformat(str(stamp)[:10].replace(":", "-")).isoformat()
            except (ValueError, OSError, EOFError, SyntaxError):
                skipped["unreadable_or_undated"] += 1
                continue
            groups[horse, day].append(member)
            skipped["duplicate_copies"] += len(copies) - 1
            break
    return groups, skipped


def split(groups):
    horses = {horse for horse, _ in groups}
    days = sorted({day for _, day in groups})
    options = []
    for cutoff in days[1:]:
        early = [key for key in groups if key[1] < cutoff]
        late = [key for key in groups if key[1] >= cutoff]
        if {horse for horse, _ in early} == {horse for horse, _ in late} == horses:
            options.append((abs(len(late) / len(groups) - 0.25), cutoff, early, late))
    if not options:
        raise ValueError("No chronological cutoff represents all identities on both sides")
    _, cutoff, early, late = min(options)

    def select(keys):
        return [(horse, day, member) for horse, day in sorted(keys)
                for member in sorted(groups[horse, day], key=lambda item: item.filename)
                [::max(1, len(groups[horse, day]) - 1)][:2]]

    return cutoff, select(early), select(late)


@torch.inference_mode()
def embed(archive, rows, model, device, query, detector=None):
    result, kept, rejected = [], [], []
    for start in range(0, len(rows), 8):
        batches = []
        for row in rows[start:start + 8]:
            _, _, member = row
            with archive.open(member) as file, Image.open(file) as image:
                image.draft("RGB", (896, 896))
                if detector is not None:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    boxes = detect(image, detector)
                    if len(boxes) != 1:
                        rejected.append((member.filename, len(boxes)))
                        continue
                    box = boxes[0]
                    x, y, w, h = (box[key] for key in ("x", "y", "width", "height"))
                    image = image.crop((int(max(0, x - 0.08*w) * image.width),
                                        int(max(0, y - 0.08*h) * image.height),
                                        int(min(1, x + 1.08*w) * image.width),
                                        int(min(1, y + 1.08*h) * image.height)))
                batches.append(torch.stack([TRANSFORM(view) for view in crop_views(image, query)]))
                kept.append(row)
        if batches:
            batch = torch.stack(batches)
            vectors = model(batch.flatten(0, 1).to(device)).float().cpu()
            result.extend(F.normalize(vectors, dim=-1).reshape(*batch.shape[:2], -1))
        processed = min(start + 8, len(rows))
        if processed % 40 == 0 or processed == len(rows):
            print(f"Processed {processed}/{len(rows)} {'queries' if query else 'references'}", flush=True)
    if not result:
        raise ValueError("No usable photographs in this split")
    return torch.stack(result), kept, rejected


@torch.inference_mode()
def evaluate(reference_vectors, query_vectors, references, queries, head=None):
    if head is not None:
        reference_vectors = F.normalize(head(reference_vectors), dim=-1)
        query_vectors = F.normalize(head(query_vectors), dim=-1)
    scores = query_vectors @ reference_vectors.flatten(0, 1).T
    scores = scores.amax(dim=1).reshape(len(queries), len(references), -1).amax(dim=-1)
    labels = sorted({horse for horse, _, _ in references})
    by_horse = torch.stack([scores[:, [row[0] == horse for row in references]].amax(dim=1)
                            for horse in labels], dim=1)
    ranks = by_horse.argsort(dim=1, descending=True).tolist()
    return [[labels[index] for index in row[:5]] for row in ranks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--horse-crops", action="store_true", help="Use images with one detected horse")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    with zipfile.ZipFile(args.archive) as archive:
        groups, skipped = inventory(archive)
        cutoff, references, queries = split(groups)
        original_ref_count, original_query_count = len(references), len(queries)
        print(f"Cutoff {cutoff}, training/gallery {len(references)}, test {len(queries)}", flush=True)
        model, device, _ = load_model()
        detector = load_detector(ROOT / "data/ml/yolo26x.pt") if args.horse_crops else None
        reference_vectors, references, rejected_refs = embed(archive, references, model, device, False, detector)
        query_vectors, queries, rejected_queries = embed(archive, queries, model, device, True, detector)
    baseline = evaluate(reference_vectors, query_vectors, references, queries)
    labels = [horse for horse, _, _ in references]
    head = train(reference_vectors[:, 0], labels, set(labels), steps=100)
    trained = evaluate(reference_vectors, query_vectors, references, queries, head)
    expected = [horse for horse, _, _ in queries]
    metrics = {name: {f"top_{k}": sum(horse in row[:k] for horse, row in zip(expected, ranks))
                      for k in (1, 3, 5)} for name, ranks in (("baseline", baseline), ("trained", trained))}
    summary = {"identities": len(set(labels)), "cutoff": cutoff, "reference_photos": len(references),
               "query_photos": len(queries), "usable_dated_jpegs": sum(map(len, groups.values())),
               "initial_reference_photos": original_ref_count, "initial_query_photos": original_query_count,
               "rejected_query_box_counts": dict(Counter(count for _, count in rejected_queries)),
               "skipped": dict(skipped), "metrics": metrics}
    output = ROOT / "data" / "ml"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_crops" if args.horse_crops else ""
    torch.save({"state_dict": head.state_dict()}, output / f"horse_head_reserve{suffix}_candidate.pt")
    report = {**summary, "references": [{"horse": h, "date": day, "file": f.filename} for h, day, f in references],
              "queries": [{"horse": h, "date": day, "file": f.filename, "baseline": b, "trained": t}
                          for (h, day, f), b, t in zip(queries, baseline, trained)],
              "excluded": {"references": rejected_refs, "queries": rejected_queries},
              "scope": ("Identity retrieval on single detected crops from attributed JPEGs. No box-level labels. "
                        if args.horse_crops else "Identity retrieval from attributed full JPEGs. ") +
                       "Species gating is not evaluated."}
    (output / f"reserve_identity{suffix}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
