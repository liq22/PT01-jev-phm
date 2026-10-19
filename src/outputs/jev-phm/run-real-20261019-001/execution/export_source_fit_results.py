"""Extract the stopped P1 source-only fits; no checkpoint/model/data is loaded.

Run with the repository's Python. Missing RF optimizer steps remain empty. The
interrupted semantic refit and the pre-fit JSON failure have no result.json and
are not efficacy rows. Selection labels describe existing source artifacts only.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[3]
DESTINATION = ROOT / "paper/assets/tables/jev_p1_source_fit_results.csv"
FIELDS = ["family", "run", "seed", "parameters", "source_nll", "selected_step",
          "last_step", "termination", "qualification", "seconds", "checkpoint",
          "artifact", "selection_status"]
rows = []
for family in ("rf", "signal", "indexed", "semantic", "permuted", "token"):
    folder = RUN / "fits" / family
    artifacts = sorted(folder.glob("trial_*/result.json")) + sorted(folder.glob("refit_*/result.json"))
    if not artifacts:
        continue
    selected = None
    selection_file = folder / "selection.json"
    if selection_file.exists():
        selected = {item["path"] for item in json.loads(selection_file.read_text())["models"]}
    trials = [path for path in artifacts if path.parent.name.startswith("trial_")]
    winner = min(trials, key=lambda path: (json.loads(path.read_text())["source_nll"], path.parent.name))
    for artifact in artifacts:
        value = json.loads(artifact.read_text())
        if not math.isfinite(value["source_nll"]) or not math.isfinite(value["seconds"]):
            raise ValueError(f"Invalid source result: {artifact}")
        checkpoint = folder / value["path"]
        if not checkpoint.is_file() or checkpoint.parent != artifact.parent:
            raise ValueError(f"Missing/misaligned completed checkpoint: {artifact}")
        if selected is not None:
            status = "selected_in_completed_family" if value["path"] in selected else "not_selected_in_completed_family"
        else:
            status = "prospective_grid_winner_family_incomplete" if artifact == winner else "unselected_grid_family_incomplete"
        rows.append(dict(family=family, run=artifact.parent.name, seed=value["seed"],
            parameters=json.dumps(value["parameters"], sort_keys=True, separators=(",", ":")),
            source_nll=value["source_nll"], selected_step=value.get("selected_step"),
            last_step=value.get("last_step"), termination=value["termination"],
            qualification=value["qualification"], seconds=value["seconds"],
            checkpoint=str(checkpoint.relative_to(ROOT)), artifact=str(artifact.relative_to(ROOT)),
            selection_status=status))
counts = Counter(row["family"] for row in rows)
if counts != {"rf": 6, "signal": 8, "indexed": 8, "semantic": 6}:
    raise ValueError(f"Stopped-study extraction no longer has its 28 completed fits: {counts}")
if DESTINATION.exists():
    with DESTINATION.open(newline="") as handle:
        previous = list(csv.DictReader(handle))
    expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    if previous != expected:
        raise ValueError("Existing source-fit table differs; preserve it and inspect the artifacts")
else:
    with DESTINATION.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"rows": len(rows), "families": dict(counts), "table": str(DESTINATION.relative_to(ROOT)),
    "prospective_grid_winners": [{"family": r["family"], "run": r["run"], "source_nll": r["source_nll"],
        "qualification": r["qualification"]} for r in rows if r["selection_status"] == "prospective_grid_winner_family_incomplete"]}, indent=2))
