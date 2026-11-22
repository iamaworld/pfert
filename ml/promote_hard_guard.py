"""Enable the second species head only if its joint held-out result improves."""

from __future__ import annotations

import json
import shutil

import torch

from model import ROOT


MODEL_DIR = ROOT / "data" / "ml"
REPORT = MODEL_DIR / "large_training_report.json"
CANDIDATE = MODEL_DIR / "species_large_candidate.pt"
GUARD = MODEL_DIR / "species_hard_guard.pt"
PROMOTION_REPORT = MODEL_DIR / "guard_promotion_report.json"
REQUIRED_GROUPS = {"old_holdout", "zoo_holdout", "horse_identity_holdout", "zebra_donkey_holdout"}


def count_errors(rows: list[dict], key: str) -> dict[str, int]:
    false_accept = sum(row["label"] != "przewalski" and row[key] == "przewalski" for row in rows)
    false_reject = sum(row["label"] == "przewalski" and row[key] != "przewalski" for row in rows)
    return {"false_accept": false_accept, "false_reject": false_reject,
            "total_errors": false_accept + false_reject}


def main() -> None:
    report = json.loads(REPORT.read_text())
    rows = report["details"]
    if len(rows) != report["test"] or {row["group"] for row in rows} != REQUIRED_GROUPS:
        raise ValueError("Held-out report is incomplete; guard was not enabled")
    for row in rows:
        row["joint"] = "przewalski" if (row["current"] == "przewalski"
                                         and row["candidate"] == "przewalski") else "not_przewalski"
    current = count_errors(rows, "current")
    joint = count_errors(rows, "joint")
    by_group = {group: {name: count_errors([row for row in rows if row["group"] == group], name)
                        for name in ("current", "joint")} for group in sorted(REQUIRED_GROUPS)}
    # Do not trade many missed Przewalski horses for fewer false accepts.
    approved = (joint["false_accept"] <= current["false_accept"] - 3
                and joint["false_reject"] <= current["false_reject"] + 1
                and joint["total_errors"] <= current["total_errors"] - 2)
    result = {"held_out_count": len(rows), "baseline": current, "joint": joint,
              "by_group": by_group, "approved": approved,
              "note": "Development holdout used for model selection; a fresh external set is still needed."}
    if not approved:
        raise ValueError(f"Guard did not improve held-out results: {result}")
    checkpoint = torch.load(CANDIDATE, map_location="cpu", weights_only=True)
    weights = checkpoint["state_dict"]
    if checkpoint.get("trained_on") != report["train"] or weights["weight"].ndim != 2 or weights["weight"].shape[0] != 1 or weights["bias"].shape != (1,) or not all(
            torch.isfinite(tensor).all() for tensor in weights.values()):
        raise ValueError("Guard checkpoint is invalid")
    shutil.copyfile(CANDIDATE, GUARD)
    PROMOTION_REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
