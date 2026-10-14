"""Check PU annotation semantics using real versioned metadata, without inference.

The published component locations below are transcribed from Lessmeier et al.
(2016), Tables 4, 5 and 7, with mixed-bearing dominance from Section 7.4. They are
explicit source facts, not classifications inferred from identifier prefixes.
Only primary rows are counted; the six repeated role rotations add no bearings.
This analysis does not modify data, define a train/test cohort, or relabel a run.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVISION = "d152c507d27ed2beb8cc2cca6ba4a04add9e439e"
UPSTREAM_REVISION = "cdc0669167ca25ac848e34c880875ca39c6aed89"
INPUT_PATH = "paper/assets/tables/jev_unit_roles.csv"
PUB_SITES = {
    **{k: "healthy" for k in ("K001", "K002", "K003", "K004", "K005", "K006")},
    **{k: "outer_ring_only" for k in (
        "KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09",
        "KA15", "KA16", "KA22", "KA30")},
    **{k: "inner_ring_only" for k in (
        "KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16",
        "KI17", "KI18", "KI21")},
    "KB23": "inner_and_outer", "KB24": "inner_and_outer", "KB27": "inner_and_outer",
}
SINGLE_SITE_LABEL = {"healthy": "healthy", "inner_ring_only": "inner-race-dominant",
                     "outer_ring_only": "outer-race-dominant"}
MIXED_DOMINANCE = {"KB23": "inner-race-dominant", "KB24": "inner-race-dominant",
                   "KB27": "outer-race-dominant"}


def analyze(path: Path) -> dict:
    """Return a descriptive join and a finite-map counterexample, not a score."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["protocol"] == "primary"]
    units = {}
    for row in rows:
        code = row["physical_unit"].rsplit(" ", 1)[-1]
        if code in units:
            raise ValueError(f"Repeated primary physical bearing: {code}")
        if code not in PUB_SITES:
            raise ValueError(f"Bearing not covered by inspected original tables: {code}")
        count = int(row["source_record_count"])
        if count <= 0:
            raise ValueError(f"Nonpositive recording count: {code}")
        sites = PUB_SITES[code]
        expected_label = MIXED_DOMINANCE[code] if code in MIXED_DOMINANCE else SINGLE_SITE_LABEL[sites]
        if row["class_name"] != expected_label:
            raise ValueError(f"Versioned label and original-paper interpretation disagree: {code}")
        units[code] = {"bearing": code, "published_component_state": sites,
                       "stored_label": row["class_name"], "raw_label": row["raw_label"],
                       "source_record_count": count, "pilot_role": row["role"]}
    if set(units) != set(PUB_SITES):
        raise ValueError("The primary inventory does not cover the inspected 32-bearing cohort")
    fibers = defaultdict(list)
    for row in units.values():
        fibers[row["published_component_state"]].append(row)
    conflicts = []
    for state, members in sorted(fibers.items()):
        labels = sorted({r["stored_label"] for r in members})
        if len(labels) > 1:
            conflicts.append({"component_state": state, "stored_labels": labels,
                              "bearings": sorted(r["bearing"] for r in members),
                              "recordings": sum(r["source_record_count"] for r in members)})
    # This checks the observed counterexample; it does not manufacture an outcome.
    if len(conflicts) != 1 or conflicts[0]["component_state"] != "inner_and_outer":
        raise ValueError("Expected native ambiguity was not reproduced; inspect sources, do not relabel")
    return {
        "analysis": "native_annotation_compatibility", "date": "2026-10-14",
        "evidence_kind": "original-source plus versioned-metadata analysis; no signal/model run",
        "input": {"repository": "liq22/PT01-jev-phm", "revision": SOURCE_REVISION,
                  "path": INPUT_PATH, "selection": "protocol == primary"},
        "upstream_revision": UPSTREAM_REVISION,
        "sources": {
            "PU": "https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf",
            "PU_locations": "Tables 4, 5 and 7; mixed sites Table 5 (p. 6); dominance Section 7.4 (p. 13)",
            "CWRU": "https://engineering.case.edu/bearingdatacenter/welcome",
            "CWRU_faults": "https://engineering.case.edu/bearingdatacenter/fault-specifications",
        },
        "population": {"bearings": len(units), "recordings": sum(r["source_record_count"] for r in units.values())},
        "by_published_state": {
            state: {"bearings": len(members), "recordings": sum(r["source_record_count"] for r in members)}
            for state, members in sorted(fibers.items())},
        "counterexamples": conflicts,
        "decisions": {
            "pure_site_H_I_O_cover_all_PU": "REFUTED: mixed observations are outside this event space",
            "H_I_O_plus_one_both_state_deterministic_coarsening": "REFUTED: one both state has two observed dominant labels",
            "single_ring_subset": "CONDITIONAL: 29 bearings / 2319 records describe a possible new task, not an authorized exclusion or native coarse-label experiment",
            "dominance_refined_states": "CONDITIONAL: may restore a deterministic map but change target semantics; need justified state definitions and identifiability",
            "PU_ball_label_from_KB_prefix": "REFUTED: KB23/24/27 have documented ring damage, not a rolling-element class",
            "CWRU_local_record_labels_channels_and_independent_units": "UNKNOWN: upstream reader and official vocabulary inspected, local metadata/H5 unavailable",
            "C4_general_method_effect": "UNTESTED", "C5_latency_and_backbone_portability": "UNTESTED",
        },
        "affected_units": [units[k] for k in sorted(MIXED_DOMINANCE)],
        "non_changes": ["pilot labels and results", "data cohort and split", "upstream loader", "model and training configuration"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, default=ROOT / INPUT_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.roles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"population": result["population"], "states": result["by_published_state"],
                      "conflicts": result["counterexamples"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
