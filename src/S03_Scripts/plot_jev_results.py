"""Recompute and plot the Jev-PHM study from saved predictions only.

No model is loaded or evaluated. All figures retain editable SVG text and include
PDF exports. Primary uncertainty resamples physical bearings; rotations remain
descriptive because their fitted models and training populations overlap.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/S01_Package"))
import jev_phm as jev

ARMS = ("Numerical-flat", "Numerical-factorized", "Qwen-flat", "Qwen-factorized",
        "Qwen-two-question-direct", "Qwen-shared-factorized")
VARIANTS = ("raw", "pooled", "unit")
PAIRS = (("Qwen-factorized", "Qwen-flat"),
         ("Qwen-factorized", "Qwen-shared-factorized"),
         ("Qwen-factorized", "Qwen-two-question-direct"),
         ("Numerical-factorized", "Numerical-flat"),
         ("Qwen-shared-factorized", "Qwen-flat"))
COLORS = ("#777777", "#262626", "#559ac1", "#045b91", "#c47332", "#8a5030")
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": False, "savefig.bbox": "tight"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_same(actual: Any, saved: Any, location: str) -> None:
    """Fail explicitly if saved metrics disagree with saved prediction arrays."""
    if isinstance(actual, dict):
        if set(actual) != set(saved):
            raise ValueError(f"Metric fields differ at {location}")
        for key in actual:
            check_same(actual[key], saved[key], f"{location}/{key}")
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(saved):
            raise ValueError(f"Saved result length differs at {location}")
        for index, (left, right) in enumerate(zip(actual, saved)):
            check_same(left, right, f"{location}/{index}")
    elif actual is None or isinstance(actual, str):
        if actual != saved:
            raise ValueError(f"Saved result differs at {location}: {actual!r} versus {saved!r}")
    elif not np.allclose(actual, saved, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Saved result differs at {location}: {actual!r} versus {saved!r}")


def read_evaluation(folder: Path, e2: bool) -> dict[str, Any]:
    """Recalculate held-out metrics and paired contrasts without fitting."""
    report, config = load_json(folder / "metrics.json"), load_json(folder / "config.json")
    arms = ARMS if e2 else ("numeric", "qwen")
    recomputed, curves, logits, unit_rows = {}, {}, {}, []
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        mask = archive["role"] == "test"
        y, units = archive["y"][mask], archive["unit"][mask]
        if not mask.any():
            raise ValueError(f"No held-out observations: {folder}")
        for arm in arms:
            prefix = arm.replace("-", "_")
            raw = archive[f"{prefix}_logits"][mask]
            recomputed[arm], curves[arm], logits[arm] = {}, {}, {}
            for variant in VARIANTS:
                z = raw / report[arm]["temperatures"][variant]
                values = jev.metrics(z, y, units, config["reject_cost"])
                check_same(values, report[arm][variant], f"{folder.name}/{arm}/{variant}")
                curve = jev.risk_coverage(z, y, units)
                for key, array in curve.items():
                    check_same(array, archive[f"{prefix}_{variant}_rc_{key}"],
                               f"{folder.name}/{arm}/{variant}/risk_coverage/{key}")
                recomputed[arm][variant], curves[arm][variant], logits[arm][variant] = values, curve, z
                unit_losses = []
                for unit in np.unique(units):
                    selected = units == unit
                    count = int(selected.sum())
                    loss = jev.nll(z[selected], y[selected], np.full(count, 1.0 / count))
                    unit_losses.append(loss)
                    unit_rows.append({"arm": arm, "calibration": variant, "unit": str(unit),
                                      "test_windows": count, "nll": loss})
                check_same(float(np.mean(unit_losses)), values["nll"],
                           f"{folder.name}/{arm}/{variant}/mean_unit_nll")
        pairs = PAIRS if e2 else tuple((arm, arm) for arm in arms)
        contrasts = {}
        for a, b in pairs:
            z_a = logits[a]["pooled" if e2 else "unit"]
            z_b = logits[b]["pooled"]
            key = f"{a}_minus_{b}" if e2 else f"{a}_unit_minus_pooled"
            delta = jev.paired_unit_nll(z_a, z_b, y, units,
                                        config.get("bootstrap_repetitions", 2000),
                                        config.get("bootstrap_seed", 20260922))
            saved = report["contrasts"][key] if e2 else report[a]["unit_minus_pooled_nll"]
            check_same(delta, saved, f"{folder.name}/{key}")
            if config["protocol"] != "primary":
                # Do not promote fold-conditioned intervals into repeated-fit uncertainty.
                delta = {k: v for k, v in delta.items() if k not in ("conditional_ci95", "repetitions", "seed")}
            contrasts[key] = delta
    origin = load_json(folder / "provenance.json")
    data_origin = json.loads(origin["data"])
    embedding_origin = json.loads(origin["embedding"])
    # Publish scientific provenance without copying local cache/data-root paths.
    provenance = {"data": {key: data_origin[key] for key in
                           ("phmfactory_commit", "window_plan", "channel", "normalization", "original_split", "evaluation_unit_roles")
                           if key in data_origin},
                  "embedding_archives": [Path(item["path"]).name for item in embedding_origin.get("archives", [])]}
    return {"config": config, "report": report, "metrics": recomputed, "provenance": provenance,
            "contrasts": contrasts, "curves": curves, "unit_nll": unit_rows, "test_windows": int(mask.sum()),
            "test_units": len(np.unique(units)), "source": str(folder.relative_to(ROOT))}


def save_figure(figure: plt.Figure, destination: Path) -> None:
    for suffix in ("svg", "pdf"):
        path = destination.with_suffix(f".{suffix}")
        figure.savefig(path)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def method_figure(destination: Path) -> None:
    """A formal vector diagram of the implemented information and action flow."""
    figure = plt.figure(figsize=(7.2, 3.15))
    ax = figure.add_axes((.01, .03, .98, .96))
    ax.set(xlim=(0, 10.7), ylim=(0, 3.6))
    ax.axis("off")
    boxes = [(0.1, 1.35, 1.6, 1.2, "Vibration\nwindow\nSix statistics\nShared state s"),
             (2.2, 2.2, 2.55, 1.2, "Fault presence\nFrozen Qwen\nBinary head\nr = P(fault | s)"),
             (2.2, 0.55, 2.55, 1.2, "Conditional fault type\nFrozen Qwen\nFault-only head\nu = P(type | fault, s)"),
             (5.45, 1.35, 2.35, 1.2, "Code composition\np(H) = 1 − r\np(k) = r uₖ, k ∈ F"),
             (8.45, 1.35, 2.1, 1.2, "Calibrate leaf p\nApply cost rule\nDiagnose / defer")]
    for x, y, width, height, label in boxes:
        color = "#e7f0f5" if x in (2.2, 5.45) else "#f3f3f3"
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor="#43515c", linewidth=0.8))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    for start, end in (((1.7, 2.05), (2.2, 2.8)), ((1.7, 1.85), (2.2, 1.15)),
                       ((4.75, 2.8), (5.45, 2.05)), ((4.75, 1.15), (5.45, 1.85)),
                       ((7.8, 1.95), (8.45, 1.95))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1, color="#43515c"))
    ax.text(3.475, 1.94, "No answer exchange", ha="center", fontsize=9, color="#43515c")
    ax.text(5.35, .08, "Diagnosis-derived labels, not independently measured physical-concept supervision.",
            ha="center", fontsize=9)
    save_figure(figure, destination)


def primary_figure(result: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), gridspec_kw={"height_ratios": [1, 1.15]})
    contrasts = [result["contrasts"][f"{a}_minus_{b}"] for a, b in PAIRS]
    labels = ["Question-factorized − flat", "Question-factorized − shared hierarchy",
              "Question-factorized − two direct", "Numerical hierarchy − numerical flat", "Shared hierarchy − flat"]
    for index, delta in enumerate(contrasts):
        mean, interval = delta["mean"], delta["conditional_ci95"]
        axes[0].plot(interval, [index, index], color="#045b91", linewidth=1.7)
        axes[0].scatter([mean], [index], s=27, color="#045b91", zorder=3)
    axes[0].axvline(0, color="#777777", linewidth=.8, linestyle="--")
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, xlabel="Paired bearing-weighted ΔNLL (nats)",
                title="(a) Primary held-out contrast")
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=9)
    for index, arm in enumerate(ARMS):
        curve = result["curves"][arm]["pooled"]
        # All prefixes use saved probabilities; no interpolation changes the risk definition.
        axes[1].plot(curve["coverage"], curve["risk"], label=arm, color=COLORS[index],
                     linewidth=1.2, linestyle="--" if "Numerical" in arm else "-")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Bearing-weighted coverage", ylabel="Accepted-window error rate",
                title="(b) Primary risk–coverage")
    axes[1].legend(fontsize=9, frameon=False, loc="upper right")
    figure.tight_layout(h_pad=2.6)
    save_figure(figure, destination)


def rotations_figure(results: dict[str, Any], selection: dict, destination: Path) -> None:
    figure, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(1, 7)
    for index, (a, b) in enumerate(PAIRS[:3]):
        values = [results[f"rotation_{fold}_raw_e2"]["contrasts"][f"{a}_minus_{b}"]["mean"] for fold in x]
        ax.plot(x, values, marker=("o", "s", "^")[index], color=COLORS[index + 3], linewidth=1.25,
                label=f"Question-factorized − {b.removeprefix('Qwen-')}")
    ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
    labels = [f"{fold}\nW={selection[f'rotation_{fold}']['window_size']}" for fold in x]
    ax.set(xticks=x, xticklabels=labels, xlabel="Predeclared rotation and train/tune-selected window length (samples)",
           ylabel="Bearing-weighted ΔNLL (nats)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    figure.tight_layout()
    save_figure(figure, destination)


def summarize(run_root: Path, figures: Path, tables: Path) -> None:
    expected = ["primary_raw_e2", "primary_raw_e1", "primary_normalized_e1"]
    expected += [f"rotation_{fold}_raw_e2" for fold in range(1, 7)]
    missing = [name for name in expected if not (run_root / "evaluations" / name / "predictions.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Real study outputs are incomplete: {', '.join(missing)}")
    results = {name: read_evaluation(run_root / "evaluations" / name, name.endswith("e2")) for name in expected}
    if results["primary_raw_e2"]["test_units"] != 8:
        raise ValueError("Primary analysis requires the frozen eight-bearing test population")
    selection = load_json(run_root / "inputs/selection.json")
    encoding = {}
    for path in sorted((run_root / "embeddings").glob("*.npz")):
        # The archive is canonical; opening just this member never loads hidden arrays.
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["provenance"]))
            encoding[path.stem] = {key: metadata[key] for key in
                                  ("model", "revision", "dtype", "batch_size", "mapping", "representation", "queries", "load_seconds", "total_seconds")
                                  if key in metadata}
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    compact = {"run": str(run_root.relative_to(ROOT)), "independent_unit": "physical bearing",
               "primary_interval": "2000 paired bearing resamples, conditional on primary fit and split",
               "rotation_interpretation": "descriptive refitted cases; overlapping fits are not independent replicates",
               "metric_definitions": {"brier": "sum of squared errors across all three classes",
                                      "ece": "10 equal-width confidence bins with equal-bearing weights",
                                      "risk_coverage": "stable saved observation-order ties; equal-bearing window weights"},
               "selection": selection, "encoding": encoding,
               "evaluations": {name: {**{key: result[key] for key in ("source", "test_windows", "test_units", "metrics", "contrasts", "config", "provenance")},
                                      "heads": {arm: {key: fields[key] for key in
                                                ("C", "tuning", "temperatures", "fit_evaluate_seconds", "head_parameters") if key in fields}
                                                for arm, fields in result["report"].items() if arm in result["metrics"]}}
                               for name, result in results.items()}}
    write_json(tables / "jev_study_summary.json", compact)
    rows = []
    for evaluation, result in results.items():
        for arm, variants in result["metrics"].items():
            for variant, values in variants.items():
                row = {"evaluation": evaluation, "arm": arm, "calibration": variant,
                       "test_units": result["test_units"], "test_windows": result["test_windows"],
                       "window_size": result["config"]["window_selection"]["window_size"],
                       "C": result["report"][arm]["C"], "temperature": result["report"][arm]["temperatures"][variant]}
                rows.append({**row, **values})
    with (tables / "jev_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unit_rows = [{"evaluation": name, **row} for name, result in results.items() for row in result["unit_nll"]]
    with (tables / "jev_unit_nll.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unit_rows[0]))
        writer.writeheader()
        writer.writerows(unit_rows)
    encoding_rows = []
    for archive, metadata in encoding.items():
        for query, fields in metadata["queries"].items():
            encoding_rows.append({"archive": archive, "query": query, "dtype": metadata["dtype"],
                                  "batch_size": metadata["batch_size"], "physical_gpu": metadata["mapping"].get("physical_index"),
                                  "load_seconds": metadata["load_seconds"], **fields})
    if not encoding_rows:
        raise ValueError("No saved encoding overhead metadata")
    with (tables / "jev_encoding.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(encoding_rows[0]))
        writer.writeheader()
        writer.writerows(encoding_rows)
    with plt.rc_context(STYLE):
        method_figure(figures / "jev_method")
        primary_figure(results["primary_raw_e2"], figures / "jev_primary_results")
        rotations_figure(results, selection, figures / "jev_rotation_contrasts")
    write_json(tables / "jev_window_selection.json", selection)
    print(f"Recomputed saved metrics and curves agree for {len(results)} evaluations; wrote figures and summary tables.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=ROOT / "paper/assets/figures")
    parser.add_argument("--tables", type=Path, default=ROOT / "paper/assets/tables")
    parser.add_argument("--method-only", action="store_true", help="Draw the implemented method without fabricating result figures")
    args = parser.parse_args()
    if args.method_only:
        args.figures.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(STYLE):
            method_figure(args.figures / "jev_method")
    else:
        summarize(args.run_root.resolve(), args.figures.resolve(), args.tables.resolve())


if __name__ == "__main__":
    main()
