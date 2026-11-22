"""Evaluate the frozen deployed gate on newly collected photos only."""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from model import ROOT, load_model
from search import check_species

DATA = ROOT / "datasets" / "benchmark_new"
REPORT = ROOT / "data" / "ml" / "new_benchmark_report.json"


def main() -> None:
    items = json.loads((DATA / "manifest.json").read_text())
    model, target, _ = load_model()
    details = []
    for index, item in enumerate(items, 1):
        result = check_species(str(DATA / item["file"]), model, target)
        details.append({"file": item["file"], "label": item["label"], "kind": item["kind"],
                        "decision": result["decision"], "score": result.get("score")})
        if index % 10 == 0:
            print(f"Checked {index}/{len(items)}", flush=True)
    summary = {"positive": sum(x["label"] == "przewalski" for x in details),
               "negative": sum(x["label"] == "not_przewalski" for x in details),
               "false_accept": sum(x["label"] == "not_przewalski" and x["decision"] == "przewalski" for x in details),
               "false_reject": sum(x["label"] == "przewalski" and x["decision"] == "not_przewalski" for x in details),
               "review": sum(x["decision"] == "review" for x in details)}
    REPORT.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
