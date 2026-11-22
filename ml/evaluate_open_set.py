"""Learn ten known reserve identities and look for new ones by cosine distance."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import zipfile

import torch
from torch.nn.functional import normalize

from detect_horses import load_detector
from evaluate_reserve_zip import embed, inventory, split
from identity_distance import photo_distances
from model import ROOT, load_model
from train import train


def threshold(positive, negative):
    """Balance the rates of missed matches and false matches in calibration."""
    positive = sorted(positive)
    negative = sorted(negative)
    if not positive or not negative:
        raise ValueError("Calibration needs both matching and nonmatching distances")
    boundaries = sorted(set(positive + negative))
    candidates = [-1e-6] + [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])] + [boundaries[-1]]
    return min(candidates, key=lambda cut: (
        1 - bisect_right(positive, cut) / len(positive) + bisect_right(negative, cut) / len(negative), cut))


def clusters(distances, cutoff):
    """Complete-link grouping. All pairs in a group must pass the threshold."""
    matrix = distances.clone()
    matrix.fill_diagonal_(float("inf"))
    groups = {i: [i] for i in range(len(matrix))}
    while len(groups) > 1:
        flat = matrix.argmin().item()
        left, right = divmod(flat, len(matrix))
        if matrix[left, right].item() > cutoff:
            break
        groups[left].extend(groups.pop(right))
        updated = torch.maximum(matrix[left], matrix[right])
        matrix[left], matrix[:, left] = updated, updated
        matrix[right, :] = matrix[:, right] = float("inf")
    return list(groups.values())


def calibration_split(early, known):
    days_by_class = defaultdict(set)
    for horse, day, _ in early:
        if horse in known:
            days_by_class[horse].add(day)
    missing = sorted(horse for horse in known if len(days_by_class[horse]) < 2)
    if missing:
        raise ValueError(f"Need two early shooting dates per known identity after filtering: {missing}")
    calibration_dates = {horse: max(days) for horse, days in days_by_class.items()}
    fit = [i for i, (horse, day, _) in enumerate(early) if horse in known and day < calibration_dates[horse]]
    calibration = [i for i, (horse, day, _) in enumerate(early) if horse in known and day == calibration_dates[horse]]
    count = Counter(early[i][0] for i in fit)
    insufficient = sorted(horse for horse in known if count[horse] < 2)
    if insufficient:
        raise ValueError(f"Need two training photos per known identity after filtering: {insufficient}")
    return fit, calibration


def classify(early_vectors, early, queries, query_rows, known, steps):
    fit, calibration = calibration_split(early, known)
    missing = sorted(known - {row[0] for row in query_rows})
    if missing:
        raise ValueError(f"No retained test photos for known identities: {missing}")
    gallery_labels = [early[i][0] for i in fit]
    head = train(early_vectors[fit, 0], gallery_labels, known, steps)
    with torch.inference_mode():
        gallery = normalize(head(early_vectors[fit]), dim=-1)
        tune = normalize(head(early_vectors[calibration]), dim=-1)
        queries = normalize(head(queries), dim=-1)
    calibration_distances = photo_distances(tune, gallery)
    matching = torch.tensor([[early[i][0] == label for label in gallery_labels] for i in calibration])
    reject_cutoff = threshold(calibration_distances.masked_fill(~matching, torch.inf).amin(1).tolist(),
                              calibration_distances.masked_fill(matching, torch.inf).amin(1).tolist())
    cluster_cutoff = threshold(calibration_distances[matching].tolist(), calibration_distances[~matching].tolist())
    distances = photo_distances(queries, gallery)
    names = sorted(known)
    by_class = torch.stack([distances[:, [label == horse for label in gallery_labels]].amin(1)
                            for horse in names], dim=1)
    values, ranks = by_class.sort(dim=1)
    unknown = [i for i, distance in enumerate(values[:, 0].tolist()) if distance > reject_cutoff]
    grouped = clusters(photo_distances(queries[unknown], queries[unknown]), cluster_cutoff) if unknown else []
    cluster_by_row = {unknown[relative]: group_id for group_id, group in enumerate(grouped, 1) for relative in group}
    rows = []
    for i, (horse, day, member) in enumerate(query_rows):
        suggestions = [names[rank] for rank in ranks[i, :5].tolist()]
        rows.append({"expected": horse, "date": day, "file": member.filename,
                     "knownInTraining": horse in known, "nearestDistance": round(values[i, 0].item(), 5),
                     "candidates": suggestions, "cluster": cluster_by_row.get(i)})
    return rows, {"unknownDistance": reject_cutoff, "clusterDistance": cluster_cutoff,
                  "fitPhotos": len(fit), "calibrationPhotos": len(calibration),
                  "fitSeasons": dict(Counter(season(early[i][1]) for i in fit))}


def season(day):
    return ("winter", "spring", "summer", "autumn")[(int(day[5:7]) % 12) // 3]


def metrics(rows):
    novel = [row for row in rows if not row["knownInTraining"]]
    known = [row for row in rows if row["knownInTraining"]]
    groups = defaultdict(Counter)
    for row in rows:
        if row["cluster"] is not None:
            groups[row["cluster"]][row["expected"]] += 1
    totals = Counter(row["expected"] for row in novel)
    precision, recall = [], []
    for row in novel:
        group = groups.get(row["cluster"], Counter())
        matches = group[row["expected"]]
        precision.append(matches / sum(group.values()) if group else 0)
        recall.append(matches / totals[row["expected"]])
    return {
        "knownPhotos": len(known),
        "knownTop1Accepted": sum(row["cluster"] is None and row["candidates"][0] == row["expected"] for row in known),
        "knownTop3": sum(row["expected"] in row["candidates"][:3] for row in known),
        "knownIncorrectlyRejected": sum(row["cluster"] is not None for row in known),
        "novelPhotos": len(novel), "novelDetected": sum(row["cluster"] is not None for row in novel),
        "clustersFound": len(groups), "novelClusteringPrecision": round(sum(precision) / len(novel), 4),
        "novelClusteringRecall": round(sum(recall) / len(novel), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--known-classes", type=int, default=10)
    parser.add_argument("--expected-total", type=int, default=15)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-frame", action="store_true", help="Skip detection; group framing may affect identity")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ml/open_set_report.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        with zipfile.ZipFile(args.archive) as archive:
            groups, skipped = inventory(archive)
            labels = sorted({horse for horse, _ in groups})
            if len(labels) != args.expected_total or not 2 <= args.known_classes < len(labels):
                parser.error(f"Expected {args.expected_total} dated identities and 2 <= known classes < total; found {len(labels)}")
            known = set(random.Random(args.seed).sample(labels, args.known_classes))
            cutoff, early, late = split(groups)
            early = [row for row in early if row[0] in known]
            model, target, _ = load_model()
            detector = None if args.full_frame else load_detector(ROOT / "data/ml/yolo26x.pt")
            early_vectors, early, dropped_early = embed(archive, early, model, target, False, detector)
            query_vectors, late, dropped_late = embed(archive, late, model, target, True, detector)
    except zipfile.BadZipFile:
        parser.error("Expected a complete ZIP archive with its central directory. Download a fresh copy.")
    missing = sorted(set(labels) - {horse for horse, _, _ in late})
    if missing:
        parser.error(f"No retained test photos for {missing}. Review ambiguous frames or compare with --full-frame.")
    rows, calibration = classify(early_vectors, early, query_vectors, late, known, args.steps)
    report = {"knownClasses": sorted(known), "hiddenClasses": sorted(set(labels) - known),
              "testStarts": cutoff, "usesHorseCrops": not args.full_frame,
              "skippedArchiveEntries": dict(skipped), "droppedEarlyPhotos": len(dropped_early),
              "droppedTestPhotos": len(dropped_late), "calibration": calibration,
              "metrics": metrics(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
