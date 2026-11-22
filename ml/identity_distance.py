"""Cosine distance for identity retrieval and open-set evaluation."""

import torch
from torch.nn.functional import normalize

PART_WEIGHT = 0.35


def cosine_distances(query, reference):
    query = normalize(query.float(), dim=-1)
    reference = normalize(reference.float(), dim=-1)
    return 1 - (query @ reference.T).clamp(-1, 1)


def photo_distances(queries, references):
    """Minimum distance across the views of each pair of photographs."""
    distances = cosine_distances(queries.flatten(0, 1), references.flatten(0, 1))
    return distances.reshape(len(queries), queries.shape[1], len(references), references.shape[1]).amin((1, 3))


def with_part_distances(global_distance, query_parts, reference_parts):
    distances = {part: cosine_distances(vector[None], reference_parts[part][None]).item()
                 for part, vector in query_parts.items() if part in reference_parts}
    if not distances:
        return global_distance, distances
    part_distance = sum(distances.values()) / len(distances)
    return (1 - PART_WEIGHT) * global_distance + PART_WEIGHT * part_distance, distances
